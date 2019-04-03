#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


import pandas as pd
import numpy as np
import random
import traceback
from os import path
DATA_DIR = path.join('..', 'data')
NP_TYPE = np.double
np.seterr(all='raise')


# # Initial state

# In[2]:


def check_state(S, I, R, N):
    assert (N == S + I + R).all() # Checks both value and shapes
    assert (S >= 0).all()
    assert (R >= 0).all()
    assert (I >= 0).all()
    assert np.isclose(N.sum(), POP_SIZE)


# In[3]:


boroughs = pd.read_csv(path.join(DATA_DIR, 'borough_pop.csv'))
stations = pd.read_csv(path.join(DATA_DIR, 'station_borough.csv'))
POP_SIZE = boroughs['Population'].sum()
POP_SIZE


# In[4]:


borough_count = stations['Local authority'].value_counts().to_frame()
borough_count.columns = ['Station count']
boroughs_pop_count = boroughs.merge(borough_count, left_on='Local authority',
                                    right_index=True, validate='one_to_one')
boroughs_pop_count['Station population'] = boroughs_pop_count['Population'] / boroughs_pop_count['Station count']
boroughs_pop_count.head()


# In[5]:


stations_pop = stations.merge(boroughs_pop_count).sort_values('Station')
INITIAL_N = stations_pop['Station population'].values


# # Transition matrices

# ## Load data

# In[6]:


DAY_LOOKUP = {
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4,
    'Sat': 5,
    'Sun': 6,
}


# In[7]:


move_data = pd.read_csv(path.join(DATA_DIR, 'journey_count.csv'))
move_data['Day'].replace(DAY_LOOKUP, inplace=True)
move_data.columns = ['Start', 'End', 'Day', 'Hour', 'Journeys']
move_data.loc[move_data['Hour'] > 23, 'Day'] += 1
move_data.loc[move_data['Hour'] > 23, 'Hour'] -= 24
move_data.head()


# In[8]:


STATION_LOOKUP = {
    name: i for i, name in enumerate(move_data['Start'].unique())
}


# ## Create matrices

# In[9]:


def calc_hour(day, hour):
    return day * 24 + hour

max_day = move_data['Day'].max()
max_day_max_hour = move_data[move_data['Day'] == max_day]['Hour'].max()
hourly_F = [
    np.zeros((len(STATION_LOOKUP), len(STATION_LOOKUP)))
    for _ in range(calc_hour(max_day, max_day_max_hour) + 1)
]

STATION_POP = {
    row['Station']: row['Station population'] for _, row in stations_pop.iterrows()
}

for row in move_data.itertuples():
    start = STATION_LOOKUP[row.Start]
    end = STATION_LOOKUP[row.End]
    hourly_F[calc_hour(row.Day, row.Hour)][start][end] = row.Journeys / STATION_POP[row.Start]


# In[10]:


def check_F(F):
    assert F.shape == (len(INITIAL_N), len(INITIAL_N))
    assert (F.sum(axis=1) < 1).all()
for F in hourly_F:
    check_F(F)


# # Constants

# In[11]:


BETA = NP_TYPE(0.5 / 24)
GAMMA = NP_TYPE((1/3) / 24)
assert np.isclose(BETA / GAMMA, 1.5)


# # Main

# In[13]:


def update_state(F, Fdash, S, I, R, N):
    S_I_interaction = BETA * S * I * 1/N
    Snew = -S_I_interaction + F.T.dot(S) - Fdash * S + S
    Inew = S_I_interaction + F.T.dot(I) - Fdash * I + (1-GAMMA) * I
    Rnew = GAMMA * I + F.T.dot(R) - Fdash * R + R
    Nnew = Snew + Inew + Rnew
    return (Snew, Inew, Rnew, Nnew)

def run_simulation(state, start_time=0, timesteps=None):
    old_err = np.seterr(under='ignore')
    hourly_Fdash = [F.sum(axis=1) for F in hourly_F]
    t = 0
    Stotals = Itotals = Rtotals = []
    Itotal = sum(I)
    while Itotal > 0.5 and (timesteps == None or t < timesteps):
#         if t % 1000 == 0:
#             print('{}: {} infected'.format(t, Itotal))
        Stotals.append(S.sum())
        Itotals.append(Itotal)
        Rtotals.append(R.sum())
#         print(state)
        F = hourly_F[t % len(hourly_F)]
        Fdash = hourly_Fdash[t % len(hourly_F)]
        new_state = update_state(F, Fdash, *state)
#         check_state(*new_state)
        state = new_state
        t += 1
#         assert len(states) == t
        Itotal = sum(state[1])
    np.seterr(**old_err)
    return (Stotals, Itotals, Rtotals)


# In[ ]:


# %timeit run_simulation(1000)


# In[14]:


INITIAL_N[3]


# In[ ]:


START_TIMES = [24*i for i in range(7)]
INITIAL_INFECTEDS = (1, 10, 10000)
HEADER = ('Init_station', 'Init_time', 'Init_count', 'Count_type')
HEADER_PRINT = 'Starting at: station {}, time {}, count {}'
import csv
with open('results.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(HEADER)
    for station_index in range(len(INITIAL_N)):
        print('Station {} of {}'.format(station_index, len(INITIAL_N)))
        for I_count in INITIAL_INFECTEDS:
            if INITIAL_N[station_index] < I_count:
                print('Too small population to start with {}'.format(I_count))
                break
            for t in START_TIMES:
                N = INITIAL_N
                I = np.zeros(len(N))
                R = np.zeros(len(N))
                I[station_index] = I_count
                S = N - I
                state = (S, I, R, N)
                try:
                    check_state(*state)
                    result = run_simulation(state=state, start_time=t)
                except Exception as err:
                    print('ERROR: start time is {}, infecteds is {}'.format(t, I_count))
                    raise
                else:
                    for i, name in enumerate(('S', 'I', 'R')):
                        outrow = [station_index, t, I_count, name]
                        outrow.extend(result[i])
                        writer.writerow(outrow)

