from joblib import Parallel, delayed
from simulate import *
import timeit

STATIONS_TO_TEST = 20

np.seterr(all='raise', under='ignore')
stations_pop, INITIAL_N = get_pop_data()
move_data = get_movement_data()
hourly_F = create_F_matrices(move_data, stations_pop)
hourly_Fdash = [F.sum(axis=1) for F in hourly_F]
STATION_COUNT = len(stations_pop)

def run_once(station_index=0):
    run_one_config(INITIAL_N, STATION_COUNT, hourly_F, hourly_Fdash, station_index, 100, 0)

def iterate():
    for i in range(STATIONS_TO_TEST):
        run_once(i)

def parallel(backend, jobs):
    Parallel(n_jobs=jobs, backend=backend)(delayed(run_once)(i,) for i in range(STATIONS_TO_TEST))

if __name__ == '__main__':
    for stmt in (
            'iterate()',
            'parallel("threading", 2)',
            'parallel("threading", 4)',
            'parallel("loky", 2)',
            'parallel("loky", 4)'
        ):
        print('{}: {}'.format(stmt, timeit.timeit(stmt, number=1, setup='from __main__ import iterate, parallel')))

