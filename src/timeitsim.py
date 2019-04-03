from simulate import *

np.seterr(all='raise', under='ignore')
stations_pop, INITIAL_N = get_pop_data()
move_data = get_movement_data()
hourly_F = create_F_matrices(move_data, stations_pop)
hourly_Fdash = [F.sum(axis=1) for F in hourly_F]
STATION_COUNT = len(stations_pop)

def run_once():
    result = run_one_config(INITIAL_N, STATION_COUNT, hourly_F, hourly_Fdash, 0, 100, 0)
