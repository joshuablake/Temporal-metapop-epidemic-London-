#!/usr/bin/env python
# coding: utf-8
import csv
import pandas as pd
import numpy as np
import traceback
from os import path

DATA_DIR = path.join('..', 'data')
NP_TYPE = np.double
OUTPUT_HEADER = False

# Disease parameters
BETA = NP_TYPE(0.5 / 24)
GAMMA = NP_TYPE((1/3) / 24)

START_TIMES = (
    3,              # Monday morning
    24 * 2 + 12,    # Midweek
    24 * 4 + 3,     # Friday morning
    24 * 4 + 12,    # Midday Friday
    24 * 5 + 3,     # Saturday Morning
)
INITIAL_INFECTEDS = (1, 10, 10000)

NONE = 0
PROGRESS = 1
DETAIL = 2
DEBUG = 3
VERBOSITY = DEBUG

DAY_LOOKUP = {
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4,
    'Sat': 5,
    'Sun': 6,
}

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
    hourly_F = np.zeros((matrix_count, STATION_COUNT, STATION_COUNT))
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
    assert np.isclose(S + I + R, N).all()

def check_F(F):
    """Some basic checks on a F matrix.

    Values should be in non-negative and row sums should be <= 1
    If VERBRO then 
    """
    # Row sums
    in_range = np_leq(F.sum(axis=1), 1)
    assert in_range.all()
    # Non-negative values
    in_range = np_geq(F, 0)
    assert in_range.all()

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
    Fdash = F.sum(axis=1)
    S = S + F.T.dot(S) - Fdash * S
    I = I + F.T.dot(I) - Fdash * I
    R = R + F.T.dot(R) - Fdash * R
    Nnew = S + I + R
    assert np.isclose(Nnew, N + F.T.dot(N) - Fdash * N).all()
    return S, I, R, Nnew

def step_SIR(tick_length, S, I, R, N):
    """Move forward one SIR time step of tick_length"""
    effective_beta = tick_length * BETA
    effective_gamma = tick_length * GAMMA
    S_I_interaction = np.zeros_like(S)
    # Empty compartments have no change, ignore them to avoid dividing by 0
    mask = ~np.isclose(N, 0)
    S_I_interaction[mask] = effective_beta * S[mask] * I[mask] / N[mask]
    S = S + -S_I_interaction
    I = I + S_I_interaction - effective_gamma * I
    R = R + effective_gamma * I
    return S, I, R

def get_normalised_F_matrix(t, N, hourly_F, tick_length):
    """Get the F matrix for a tick, normalised for current population size.

    Params:
        t: start of this tick
        N: current N vector (i.e. station populations)
        hourly_F: array of F matrices, indexed by t
        tick_length: how long this tick lasts

    Returns:
        F matrix. F[i][j] is proportion of population at i that moves to j
        during this timestep
    """
    def reduce_all_rows_to_one(F):
        mask = (F.sum(axis=1) > 1)
        F[mask] = F[mask,] / F[mask].sum(axis=1).reshape(F[mask].shape[0], 1)
        debug_print(DEBUG, 'Adjusting too high F at {}',
                    ', '.join(str(i) for i, val in enumerate(mask) if val))
    # Calculate start and end indices for hourly_F
    start_idx = t % len(hourly_F)
    end_idx = (start_idx + tick_length) % len(hourly_F)
    F = hourly_F[start_idx:end_idx].sum(axis=0)
    # Perform (number moving) / (number in station) to get proportions
    empty_stations = np.isclose(N, 0)       # ignore 0s 
    normalisation = N.reshape((N.shape[0], 1))
    F[~empty_stations] = F[~empty_stations] / normalisation[~empty_stations]
    F[empty_stations] = 0
    reduce_all_rows_to_one(F)
    check_F(F)
    return F

def run_simulation(state, hourly_F, tick_length, start_time=0, timesteps=None):
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
                    except Exception:
                        if VERBOSITY == DEBUG:
                            raise
                        else:
                            print('---------------------------------------------------------')
                            print('Error running with start state: station {}, t {}, I {}'.format(station_index, t, I_count))
                            traceback.print_exc()
                            print('---------------------------------------------------------')
                    else:
                        for i, name in enumerate(('S', 'I', 'R')):
                            outrow = [station_index, t, I_count, name]
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
        for tick_length in range(1, MAX_LENGTH):
            debug_print(PROGRESS, 'Tick length {} of {}', tick_length, MAX_LENGTH)
            try:
                result = run_one_config(INITIAL_N, hourly_F, STATION_INDEX, INITIAL_I, INITIAL_T,
                                        tick_length=tick_length)
            except Exception:
                if VERBOSITY == DEBUG:
                    raise
                else:
                    print('---------------------------------------------------------')
                    print('Error running with tick length {}'.format(tick_length))
                    traceback.print_exc()
                    print('---------------------------------------------------------')
            else:
                for i, name in enumerate(('S', 'I', 'R')):
                    outrow = [tick_length, name]
                    outrow.extend(result[i])
                    writer.writerow(outrow)

def setup():
    stations_pop, INITIAL_N = get_pop_data()
    move_data = get_movement_data()
    hourly_F = create_F_matrices(move_data, stations_pop)
    STATION_COUNT = len(stations_pop)
    return STATION_COUNT, INITIAL_N, hourly_F

if __name__ == '__main__':
    run_all_tick_lengths()
