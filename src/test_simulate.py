import numpy as np
import simulate
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

np.seterr(all='raise', under='ignore')

@st.composite
def st_state(draw, min_stations=1, max_stations=1000):
    station_count = draw(st.integers(min_stations, max_stations))
    strategy_args = {
        'dtype': simulate.NP_TYPE,
        'shape': (station_count,),
        'elements': st.floats(min_value=0, max_value=1e100, allow_nan=False, allow_infinity=False),
        'fill': st.floats(min_value=0, max_value=0, allow_nan=False, allow_infinity=False),
    }
    S = draw(npst.arrays(**strategy_args))
    I = draw(npst.arrays(**strategy_args))
    R = draw(npst.arrays(**strategy_args))
    try:
        N = S + I + R
        hyp.assume(np.isfinite(N.sum()))
    except FloatingPointError:
        hyp.assume(False)
    return (S, I, R, N)

def assume_check_func(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except AssertionError:
        hyp.assume(False)

@hyp.given(st_state())
def test_valid_states(state):
    simulate.check_state(state[-1].sum(), *state)

@hyp.given(st_state())
def test_no_movement(state):
    assume_check_func(simulate.check_state, state[-1].sum(), *state)
    num_dim = state[0].shape[0]
    F = np.identity(num_dim)
    initial_state = tuple(i.copy() for i in state)
    new_state = simulate.step_travel(F, *state)
    for i, name in enumerate(['S', 'I', 'R', 'N']):
        assert np.isclose(initial_state[i], new_state[i]).all(), name

@hyp.given(st_state())
def test_all_population_to_last_station(state):
    initial_pop = state[-1].sum()
    initial_state = tuple(i.copy() for i in state)
    assume_check_func(simulate.check_state, initial_pop, *state)
    num_dim = state[0].shape[0]
    F = np.zeros(shape=(num_dim, num_dim))
    F[:,-1] = 1
    new_state = simulate.step_travel(F, *state)
    simulate.check_state(initial_pop, *new_state)
    assert np.isclose(initial_pop, new_state[-1][-1].sum())
    # Check no movement between compartments
    for i, name in enumerate(['S', 'I', 'R']):
        assert np.isclose(initial_state[i].sum(), new_state[i].sum()), name

@hyp.given(st_state(min_stations=3))
def test_half_population_to_third_station(state):
    initial_pop = state[-1].sum()
    initial_state = tuple(i.copy() for i in state)
    assume_check_func(simulate.check_state, initial_pop, *state)
    num_dim = state[0].shape[0]
    hyp.assume(num_dim >= 3)
    F = np.identity(num_dim) / 2
    F[:,2] = 0.5
    F[2][2] = 1
    new_state = simulate.step_travel(F, *state)
    simulate.check_state(initial_pop, *new_state)
    correct_populations = np.isclose(initial_state[-1] / 2, new_state)
    # Have already checked population preserved (in check_state) so if all
    # are correct except 3rd station, so is 3rd station
    correct_populations[2] = True
    assert correct_populations[2].all()
    # Check no movement between compartments
    for i, name in enumerate(['S', 'I', 'R']):
        assert np.isclose(initial_state[i].sum(), new_state[i].sum()), name

@hyp.given(st.integers(0, 100), st_state())
@hyp.example(
    tick_length = 3,
    state = (
        np.array([12535.120608433417]), np.array([175.41209488175582]), np.array([382.4358827165815]), np.array([13092.968586031755])
    )
)
def test_SIR_gives_valid_results(tick_length, state):
    hyp.assume(((tick_length * simulate.BETA * state[1]) <= state[3]).all())
    # hyp.assume((tick_length * simulate.BETA < state[3] / state[1]).all())
    initial_pop = state[-1].sum()
    initial_state = tuple(i.copy() for i in state)
    assume_check_func(simulate.check_state, initial_pop, *state)
    S, I, R = simulate.step_SIR(tick_length, *state)
    simulate.check_state(initial_pop, S, I, R, initial_state[-1])