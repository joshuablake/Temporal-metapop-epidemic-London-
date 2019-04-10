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

# Disease parameters
BETA = NP_TYPE(0.5 / 24)
GAMMA = NP_TYPE((1/3) / 24)
START_TIMES = [24*i for i in range(7)]
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
    def check_F(F):
        assert F.shape == (STATION_COUNT, STATION_COUNT)
        assert (F.sum(axis=1) < 1).all()
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
        hourly_F[hour][start][end] = row.Journeys / STATION_POP[row.Start]

    for F in hourly_F:
        check_F(F)

    return hourly_F




def update_state(F, Fdash, S, I, R, N):
    S_I_interaction = BETA * S * I * 1/N
    Snew = -S_I_interaction + F.T.dot(S) - Fdash * S + S
    Inew = S_I_interaction + F.T.dot(I) - Fdash * I + (1-GAMMA) * I
    Rnew = GAMMA * I + F.T.dot(R) - Fdash * R + R
    Nnew = Snew + Inew + Rnew
    return (Snew, Inew, Rnew, Nnew)

def run_simulation(state, hourly_F, hourly_Fdash, start_time=0, timesteps=None):
    t = start_time
    end_time = timesteps and t + timesteps
    output = ([], [], [])

    def update_output(state):
        for out_row, state_row in zip(output, state):
            out_row.append(state_row.sum())

    def get_matrices(t):
        F = hourly_F[t % len(hourly_F)]
        Fdash = hourly_Fdash[t % len(hourly_F)]
        return F, Fdash

    Itotal = sum(state[1])
    while Itotal > 0.5 and (end_time is None or t < end_time):
        if t % 1000 == 0:
            debug_print(DETAIL, '{}: {} infected', (t, Itotal))
        update_output(state)
        debug_print(DEBUG, state)
        F, Fdash = get_matrices(t)
        state = update_state(F, Fdash, *state)
        t += 1
        Itotal = sum(state[1])
    return output

def run_one_config(N, STATION_COUNT, hourly_F, hourly_Fdash, station_index, I_count, t):
    I = np.zeros(STATION_COUNT)
    I[station_index] = I_count
    S = N - I
    R = np.zeros(STATION_COUNT)
    state = (S, I, R, N)
    return run_simulation(state, hourly_F, hourly_Fdash, start_time=t)

def run_all_stations_times():
    np.seterr(all='raise', under='ignore')
    HEADER = ('Init_station', 'Init_time', 'Init_count', 'Count_type')
    HEADER_PRINT = 'Starting at: station {}, time {}, count {}'
    stations_pop, INITIAL_N = get_pop_data()
    move_data = get_movement_data()
    hourly_F = create_F_matrices(move_data, stations_pop)
    hourly_Fdash = [F.sum(axis=1) for F in hourly_F]
    STATION_COUNT = len(stations_pop)

    with open('results.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(HEADER)
        for station_index in range(STATION_COUNT):
            debug_print(PROGRESS, 'Station {} of {}', station_index, STATION_COUNT)
            for I_count in INITIAL_INFECTEDS:
                if INITIAL_N[station_index] < I_count:
                    print('Too small population to start with {} infections'
                            .format(I_count))
                    break
                for t in START_TIMES:
                    result = run_one_config(INITIAL_N, STATION_COUNT, hourly_F, hourly_Fdash, station_index, I_count, t)
                    for i, name in enumerate(('S', 'I', 'R')):
                        outrow = [station_index, t, I_count, name]
                        outrow.extend(result[i])
                        writer.writerow(outrow)

if __name__ == '__main__':
    run_all_stations_times()
