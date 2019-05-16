#!/usr/bin/env python
# coding: utf-8
import csv
import pandas as pd
import numpy as np
import traceback
from scipy.integrate import solve_ivp
from os import path

DATA_DIR = path.join('..', 'data')
NP_TYPE = np.double
OUTPUT_HEADER = False

# Disease parameters
BETA = NP_TYPE(0.5 / 24)
GAMMA = NP_TYPE((1/3) / 24)

START_TIMES = (
    3,              # Monday morning
    24 * 4 + 3,     # Friday morning
    24 * 5 + 3,     # Saturday Morning
)
INITIAL_INFECTEDS = (1, 10, 10000)

NONE = 0
PROGRESS = 1
DETAIL = 2
DEBUG = 3
VERBOSITY = DETAIL

DAY_LOOKUP = {
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4,
    'Sat': 5,
    'Sun': 6,
}

def get_false_indices(arr):
    return [i for i, v in enumerate(arr) if not v]

def debug_print(level, msg, *vars):
    """Prints a message depending on current verbosity and message severity"""
    if level <= VERBOSITY:
        print(msg.format(*vars))

def get_pop_data():
    """Read csv population data

    Returns:
        Tuple (stations_pop, pop_values)
        stations_pop: pandas dataframe with columns detailing the stations and their boroughs
        pop_values: a numpy 1D array of population values for stations
    """
    boroughs = pd.read_csv(path.join(DATA_DIR, 'borough_pop.csv'))
    stations = pd.read_csv(path.join(DATA_DIR, 'station_borough.csv'))
    borough_count = stations['Local authority'].value_counts().to_frame()
    borough_count.columns = ['Station count']
    boroughs_pop_count = boroughs.merge(borough_count, left_on='Local authority',
                                        right_index=True, validate='one_to_one')
    boroughs_pop_count['Station population'] = \
        boroughs_pop_count['Population'] / boroughs_pop_count['Station count']
    stations_pop = stations.merge(boroughs_pop_count).sort_values('Station')
    pop_values = stations_pop['Station population'].values
    return stations_pop, pop_values

def get_movement_data():
    """Read csv files of TfL movement data

    Returns:
        pandas dataframe, each row a unique combination of start station, end
        stations, day of the week (0-6), and hour of the day (0-23), along with
        the corrosponding number of journeys
    """
    move_data = pd.read_csv(path.join(DATA_DIR, 'journey_count.csv'))
    move_data.columns = ['Start', 'End', 'Day', 'Hour', 'Journeys']
    # Make days numeric
    move_data['Day'].replace(DAY_LOOKUP, inplace=True)
    # Normalise when hours roll over
    move_data.loc[move_data['Hour'] > 23, 'Day'] = (move_data.loc[move_data['Hour'] > 23, 'Day'] + 1) % 7
    move_data.loc[move_data['Hour'] > 23, 'Hour'] -= 24
    assert move_data['Day'].max() < 7
    assert move_data['Hour'].max() < 24
    return move_data

def calc_hour(day, hour):
    """Convert day of week and hour of day into hour of week

    Params:
        day: int (0-6) representing day of week
        hour: int (0-23) representing hour of day

    Returns:
        hour of week: int (0-167)
    """
    return day * 24 + hour

def create_F_matrices(move_data, stations_pop):
    """Combine read in data to create F matrices needed for simulation"""
    assert frozenset(move_data['Start']) <= frozenset(stations_pop['Station'])
    assert frozenset(move_data['End']) <= frozenset(stations_pop['Station'])
    # numeric value of each station, i.e. their alphabetical position
    station_names = stations_pop['Station'].unique()
    station_names.sort()
    STATION_LOOKUP = {
        name: i for i, name in enumerate(station_names)
    }
    STATION_COUNT = len(STATION_LOOKUP)
    matrix_count = calc_hour(6, 23) + 1
    # hourly_F[h][i][j] is the number of people travelling from i to j in hour h
    # h is an hour of the week
    hourly_F = np.zeros((matrix_count, STATION_COUNT, STATION_COUNT), dtype=np.int32)
    for row in move_data.itertuples():
        start = STATION_LOOKUP[row.Start]
        end = STATION_LOOKUP[row.End]
        if start != end:
            hour = calc_hour(row.Day, row.Hour)
            hourly_F[hour][start][end] = 20 * row.Journeys
    return hourly_F

def np_geq(a, b):
    """ If a >= b using float comparison for ="""
    lt = a > b
    eq = np.isclose(a, b)
    return np.logical_or(lt, eq)

def np_leq(a, b):
    """ If a <= b using float comparison for ="""
    lt = a < b
    eq = np.isclose(a, b)
    return np.logical_or(lt, eq)

def check_state(expected_population, S, I, R, N):
    """Some basic checks on the system state.

    Everything should be non-negative, population is conserved, and N is the
    total population in each case.
    """
    assert np_geq(S, 0).all()
    assert np_geq(I, 0).all()
    assert np_geq(R, 0).all()
    assert np.isclose(N.sum(), expected_population)
    assert np.allclose(S + I + R, N)

def check_F(F, row_sum=1, all_positive=True):
    """Some row sums on a F matrix.

    Values should be in non-negative and row sums should be == row_sum
    """
    # Row sums
    assert np.allclose(F.sum(axis=1), row_sum)
    # Non-negative values
    if all_positive:
        assert np_geq(F, 0).all()

def update_state(F, tick_length, S, I, R, N):
    """Process one tick.

    Params:
        F: matrix to use for travel
        tick_length: how long this tick should be
        (S, I, R, N): starting state

    Returns:
        (S, I, R, N): state after this tick
    """
    start_population = N.sum()
    if VERBOSITY >= DEBUG:
        # Helps when debugging AssertionError
        start_state = (S.copy(), I.copy(), R.copy(), N.copy())
    S, I, R = step_SIR(tick_length, S, I, R, N)
    if VERBOSITY >= DEBUG:
        # Helps when debugging AssertionError
        SIR_state = (S.copy(), I.copy(), R.copy(), N.copy())
    check_state(start_population, S, I, R, N)
    S, I, R, N = step_travel(F, S, I, R, N)
    check_state(start_population, S, I, R, N)
    return (S, I, R, N)

def step_travel(F, S, I, R, N):
    """One timestep of travel"""
    assert np.allclose(F.sum(axis=1), 1)
    S = F.T.dot(S)
    I = F.T.dot(I)
    R = F.T.dot(R)
    Nnew = S + I + R
    assert np.allclose(Nnew, F.T.dot(N))
    return S, I, R, Nnew

def step_SIR(tick_length, S, I, R, N):
    """Move forward one SIR time step of tick_length"""
    effective_beta = tick_length * BETA
    effective_gamma = tick_length * GAMMA
    # Empty compartments have no change, ignore them to avoid dividing by 0
    idxs = ~np.isclose(N, 0)
    S_I_interaction = np.zeros_like(S)
    S_I_interaction[idxs] = effective_beta * S[idxs] * I[idxs] / N[idxs]
    Snew = S + -S_I_interaction
    Inew = I + S_I_interaction - effective_gamma * I
    Rnew = R + effective_gamma * I
    return Snew, Inew, Rnew

def get_normalised_F_matrix(t, N, hourly_F, tick_length=1, row_sum=1, positive_values=True):
    """Get the F matrix for a tick, normalised for current population size.

    Params:
        t: start of this tick
        N: current N vector (i.e. station populations)
        hourly_F: array of F matrices, indexed by t
        tick_length: how long this tick lasts
        row_sum: adjust the diagonal such that rows sum to this value.
                Common values are 0 or 1. 0 means that the values can be
                considered the rate of change of people per hour: diagonal
                is negative equal to rate of leaving. 1 means that values  can
                be considered the proportion of people ending up in each
                station, the diagonal is the proportion not moving.

    Returns:
        F matrix. F[i][j] is proportion of population at i that moves to j
        during this timestep
    """
    def set_row_sum(F):
        diags = np.diag_indices_from(F)
        F[diags] = 0
        mask = (F.sum(axis=1) > 1)
        if mask.any():
            F[mask] = F[mask,] / F[mask].sum(axis=1).reshape(F[mask].shape[0], 1)
            debug_print(DEBUG, 'Adjusting too high F at {}',
                        ', '.join(str(i) for i, val in enumerate(mask) if val))
        F[diags] = np.full_like(F[diags], row_sum) - F.sum(axis=1)
    # Calculate start and end indices for hourly_F
    start_idx = t % len(hourly_F)
    end_idx = (start_idx + tick_length) % len(hourly_F)
    F = np.array(hourly_F[start_idx:end_idx].sum(axis=0), dtype=NP_TYPE)
    assert F.shape == (N.shape[0], N.shape[0])
    # Perform (number moving) / (number in station) to get proportions
    empty_stations = np.isclose(N, 0)       # ignore 0s 
    normalisation = N.reshape((N.shape[0], 1))
    F[~empty_stations] = F[~empty_stations] / normalisation[~empty_stations]
    F[empty_stations] = 0
    set_row_sum(F)
    check_F(F, row_sum, positive_values)
    return F

def run_simulation(state, hourly_F, tick_length, start_time=0, timesteps=None):
    """Run a simulation from a given state using F matrices.

    Params:
        state: (S, I,, R, N) vectors to start simulation at
        hourly_F: list of F matrices to use for the travel component of the simulation
        tick_length: how long each tick (step) of the simulation should last
        start_time: the time at which to start the simulation. Affects the initial
                    index into the hourly_F array.
        timesteps: how many steps to run the simulation for. None (default) runs until there
                    is <0.5 infected people.

    Returns:
        tuple of three lists, representing that total number of people in each
        compartment for each step of the simulation.
    """
    t = start_time
    end_time = timesteps and t + (timesteps * tick_length)
    output = ([], [], [])

    def update_output(state):
        for out_row, state_row in zip(output, state):
            out_row.append(state_row.sum())
    
    Itotal = state[1].sum()
    while Itotal > 0.5 and (end_time is None or t < end_time):
        if t % 1000 == 0:
            debug_print(DETAIL, '{}: {} infected', t, Itotal)
        update_output(state)
        F = get_normalised_F_matrix(t, state[3], hourly_F, tick_length)
        if t / 24 < 6 and t % 24 == 17:
            debug_print(DEBUG, '{} infecteds move', (F.dot(state[1]).sum()))
        new_state = update_state(F, tick_length, *state)
        state = new_state
        t += tick_length
        Itotal = state[1].sum()
    return output

def run_one_config_ode(N0, hourly_F, station_index, I_count, t0, tick_length=1, return_raw=False):
    initial_population = N0.sum()
    get_derivs = create_derivative_func(hourly_F, initial_population)
    
    R0 = np.zeros_like(N0)
    I0 = np.zeros_like(N0)
    I0[station_index] = I_count
    S0 = N0 - I0 - R0
    y0 = np.concatenate((S0, I0, R0))
    result = solve_ivp(get_derivs, (t0, t0+7000), y0, method='Radau')
    for t, y in enumerate(result.y.T):
        state = np.array(np.split(np.array(y, dtype=NP_TYPE), 3), dtype=NP_TYPE)
        try:
            check_state(initial_population, *state, N=state.sum(axis=0))
        except AssertionError:
            debug_print(DETAIL, 'Possible ode issue at {} time', t)
    # Return timeseries of sum of S, I, R across all stations
    if return_raw:
        return result
    else:
        result = (i.sum(axis=1) for i in np.split(result.y, 3, axis=0))

def create_derivative_func(hourly_F, initial_population):
    def get_derivs(t, y):
        S, I, R = np.split(y, 3)
        N = S + I + R
        F = get_normalised_F_matrix(int(t), N, hourly_F, row_sum=0, positive_values=False)
        idxs = ~np.logical_or(np_leq(N, 0), np.logical_or(np_leq(S, 0), np_leq(I, 0)))
        S_I_interaction = np.zeros_like(S)
        S_I_interaction[idxs] = BETA * S[idxs] * I[idxs] / N[idxs]
        dSdt = -S_I_interaction + F.T.dot(S)
        dIdt = S_I_interaction - GAMMA * I + F.T.dot(I)
        dRdt = GAMMA * I + F.T.dot(R)
        dNdt = dSdt + dIdt + dRdt
        try:
            assert np.allclose(F.T.dot(N), dNdt, atol=0.1)
            assert np.isclose(0, dNdt.sum(), atol=0.1)
        except AssertionError:
            check_state(initial_population, S, I, R, N)
            raise
        return np.concatenate((dSdt, dIdt, dRdt))
    return get_derivs

def run_one_config(N, hourly_F, station_index, I_count, t, tick_length=1):
    debug_print(DEBUG, 'Shape of N is: {}', N.shape)
    R = np.zeros_like(N)
    I = np.zeros_like(N)
    I[station_index] = I_count
    S = N - I
    state = (S, I, R, N)
    return run_simulation(state, hourly_F, start_time=t, tick_length=tick_length)

def run_all_stations_times():
    np.seterr(all='raise', under='warn')
    HEADER = ('Init_station', 'Init_time', 'Init_count', 'Count_type')
    STATION_COUNT, INITIAL_N, hourly_F = setup()

    with open('results.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        if OUTPUT_HEADER:
            writer.writerow(HEADER)
        for station_index in range(STATION_COUNT):
            debug_print(PROGRESS, 'Station {} of {}', station_index, STATION_COUNT)
            for I_count in INITIAL_INFECTEDS:
                if INITIAL_N[station_index] < I_count:
                    debug_print(PROGRESS, 'Too small population to start with {} infections'
                            .format(I_count))
                    break
                for t in START_TIMES:
                    try:
                        result = run_one_config(INITIAL_N, hourly_F, station_index, I_count, t)
                        write_result(station_index, t, I_count, result, writer)
                    except Exception:
                        if VERBOSITY == DEBUG:
                            raise
                        else:
                            print('---------------------------------------------------------')
                            print('Error running with start state: station {}, t {}, I {}'.format(station_index, t, I_count))
                            traceback.print_exc()
                            print('---------------------------------------------------------')

def write_result(result, writer, *info):
    for i, name in enumerate(('S', 'I', 'R')):
        outrow = list(info) + [name]
        outrow.extend(result[i])
        writer.writerow(outrow)

def run_all_tick_lengths():
    np.seterr(all='raise', under='warn')
    _, INITIAL_N, hourly_F = setup()
    MAX_LENGTH = 25
    # Randomly chosen scenario
    STATION_INDEX = 355
    INITIAL_T = 108
    INITIAL_I = 1

    with open('results.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        debug_print(PROGRESS, 'Running ODE')
        result = run_one_config_ode(INITIAL_N, hourly_F, STATION_INDEX, INITIAL_I, INITIAL_T)
        write_result(result, writer, 0)
        for tick_length in range(1, MAX_LENGTH):
            debug_print(PROGRESS, 'Tick length {} of {}', tick_length, MAX_LENGTH)
            try:
                result = run_one_config(INITIAL_N, hourly_F, STATION_INDEX, INITIAL_I, INITIAL_T,
                                        tick_length=tick_length)
                write_result(result, writer, tick_length)
            except Exception:
                if VERBOSITY == DEBUG:
                    raise
                else:
                    print('---------------------------------------------------------')
                    print('Error running with tick length {}'.format(tick_length))
                    traceback.print_exc()
                    print('---------------------------------------------------------')

def setup():
    stations_pop, INITIAL_N = get_pop_data()
    move_data = get_movement_data()
    hourly_F = create_F_matrices(move_data, stations_pop)
    STATION_COUNT = len(stations_pop)
    return STATION_COUNT, INITIAL_N, hourly_F

if __name__ == '__main__':
    run_all_tick_lengths()