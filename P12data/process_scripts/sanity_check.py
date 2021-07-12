import numpy as np

P_list = np.load('../processed_data/P_list.npy', allow_pickle=True)

arr_outcomes = np.load('../processed_data/arr_outcomes.npy', allow_pickle=True)

PTdict_list = np.load('../processed_data/PTdict_list.npy', allow_pickle=True)

print(P_list.shape, P_list[0])
print(arr_outcomes.shape, arr_outcomes[0])
print(PTdict_list.shape, PTdict_list[0])