#!/usr/bin/env python
# coding: utf-8
import csv
import pandas as pd
import numpy as np
import random
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
VERBOSITY = PROGRESS

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
    if level <= VERBOSITY:
        print(msg.format(*vars))

def get_pop_data():
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
    move_data = pd.read_csv(path.join(DATA_DIR, 'journey_count.csv'))
    move_data.columns = ['Start', 'End', 'Day', 'Hour', 'Journeys']
    # Make days numeric
    move_data['Day'].replace(DAY_LOOKUP, inplace=True)
    # Normalise when hours roll over
    move_data.loc[move_data['Hour'] > 23, 'Day'] += 1
    move_data.loc[move_data['Hour'] > 23, 'Hour'] -= 24
    return move_data

def calc_hour(day, hour):
    return day * 24 + hour

def create_F_matrices(move_data, stations_pop):
    STATION_POP = {
        row['Station']: row['Station population'] for _, row in stations_pop.iterrows()
    }
    STATION_COUNT = len(STATION_POP)
    STATION_LOOKUP = {
        name: i for i, name in enumerate(move_data['Start'].unique())
    }
    max_day = move_data['Day'].max()
    max_day_max_hour = move_data[move_data['Day'] == max_day]['Hour'].max()
    hourly_F = [
        np.zeros((STATION_COUNT, STATION_COUNT))
        for _ in range(calc_hour(max_day, max_day_max_hour) + 1)
    ]
    for row in move_data.itertuples():
        start = STATION_LOOKUP[row.Start]
        end = STATION_LOOKUP[row.End]
        hour = calc_hour(row.Day, row.Hour)
        if start != end:
            hourly_F[hour][start][end] = 20 * row.Journeys

    return hourly_F

def np_geq(a, b):
    """ If a >= b using float comparison for ="""
    lt = a > b
    eq = np.isclose(a, b)
    return np.logical_or(lt, eq)

def check_state(INITAL_POPULATION, S, I, R, N):
    assert np_geq(S, 0).all()
    assert np_geq(I, 0).all()
    assert np_geq(R, 0).all()
    assert np.isclose(N.sum(), INITAL_POPULATION)

def update_state(F, Fdash, S, I, R, N):
    # Progress disease
    S_I_interaction = np.zeros(S.shape)
    mask = ~np.isclose(N, 0)
    S_I_interaction[mask] = BETA * S[mask] * I[mask] / N[mask]
    Snew = -S_I_interaction + S
    Inew = S_I_interaction + (1-GAMMA) * I
    Rnew = GAMMA * I + R
    Nnew = Snew + Inew + Rnew
    # Add travel
    Snew += F.T.dot(Snew) - Fdash * Snew
    Inew += F.T.dot(Inew) - Fdash * Inew
    Rnew += F.T.dot(Rnew) - Fdash * Rnew
    return (Snew, Inew, Rnew, Nnew)

def run_simulation(state, hourly_F, start_time=0, timesteps=None):
    t = start_time
    end_time = timesteps and t + timesteps
    output = ([], [], [])

    def np_leq(a, b):
        """ If a <= b using float comparison for ="""
        lt = a < b
        eq = np.isclose(a, b)
        return np.logical_or(lt, eq)

    def update_output(state):
        for out_row, state_row in zip(output, state):
            out_row.append(state_row.sum())

    def get_matrices_and_normalise(t, N):
        def reduce_all_rows_to_one(F):
            mask = (F.sum(axis=1) > 1)
            if mask.any():
                F[mask] = F[mask,] / F[mask].sum(axis=1).reshape(F[mask].shape[0], 1)
                debug_print(DETAIL, 'Adjusting too high F at {}', (i for i, val in enumerate(mask) if val))
        def check_F(F):
            try:
                in_range = np_leq(F.sum(axis=1), 1)
                assert in_range.all()
            except AssertionError:
                if DEBUG:
                    print(F[~in_range])
                    print('Sum(s) are {}'.format(F.sum(axis=1)[~in_range]))
                    import pdb;pdb.set_trace()
                raise
            try:
                in_range = np_geq(F, 0)
                assert in_range.all()
            except AssertionError:
                if DEBUG:
                    print(F[~in_range])
                    import pdb;pdb.set_trace()
                raise
        F = hourly_F[t % len(hourly_F)]
        empty_stations = np.isclose(N, 0)
        normalisation = N.reshape((N.shape[0], 1))
        # prev_err = np.seterr(all='ignore')
        F[~empty_stations] = F[~empty_stations] / normalisation[~empty_stations]
        # np.seterr(**prev_err)
        F[empty_stations] = 0
        reduce_all_rows_to_one(F)
        check_F(F)
        Fdash = F.sum(axis=1)
        return F, Fdash

    INITIAL_POPULATION = state[3].sum()
    Itotal = state[1].sum()
    while Itotal > 0.5 and (end_time is None or t < end_time):
        if t % 1000 == 0:
            debug_print(DETAIL, '{}: {} infected', t, Itotal)
        update_output(state)
        F, Fdash = get_matrices_and_normalise(t, state[3])
        new_state = update_state(F, Fdash, *state)
        try:
            check_state(INITIAL_POPULATION, *new_state)
        except AssertionError:
            if DEBUG: import pdb; pdb.set_trace()
            raise
        state = new_state
        t += 1
        Itotal = state[1].sum()
    return output

def run_one_config(N, STATION_COUNT, hourly_F, station_index, I_count, t):
    I = np.zeros(STATION_COUNT)
    I[station_index] = I_count
    S = N - I
    R = np.zeros(STATION_COUNT)
    state = (S, I, R, N)
    return run_simulation(state, hourly_F, start_time=t)

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
                        result = run_one_config(INITIAL_N, STATION_COUNT, hourly_F, station_index, I_count, t)
                    except Exception as e:
                        print('Error running with start state: station {}, t {}, I {}'.format(station_index, t, I_count))
                        traceback.print_exc()
                    else:
                        for i, name in enumerate(('S', 'I', 'R')):
                            outrow = [station_index, t, I_count, name]
                            outrow.extend(result[i])
                            writer.writerow(outrow)

def setup():
    stations_pop, INITIAL_N = get_pop_data()
    move_data = get_movement_data()
    hourly_F = create_F_matrices(move_data, stations_pop)
    STATION_COUNT = len(stations_pop)
    return STATION_COUNT, INITIAL_N, hourly_F

if __name__ == '__main__':
    run_all_stations_times()
