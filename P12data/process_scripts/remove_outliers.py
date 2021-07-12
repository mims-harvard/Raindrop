import numpy as np

"""Remove 12 patients at blacklist"""
PTdict_list = np.load('../processed_data/PTdict_list.npy', allow_pickle=True)
arr_outcomes = np.load('../processed_data/arr_outcomes.npy', allow_pickle=True)


# remove blacklist patients
blacklist = ['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731', '150309', '155655', '156254']

i = 0
n = len(PTdict_list)
while i<n:
    pid = PTdict_list[i]['id']
    if pid in blacklist:
        PTdict_list = np.delete(PTdict_list, i)
        arr_outcomes = np.delete(arr_outcomes, i, axis=0)
        n -= 1
    i += 1
print(len(PTdict_list), arr_outcomes.shape)

np.save('../processed_data/PTdict_list.npy', PTdict_list)
np.save('../processed_data/arr_outcomes.npy', arr_outcomes)