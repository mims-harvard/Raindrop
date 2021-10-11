
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


df_outcomes_a = pd.read_csv('../rawdata/Outcomes-a.txt', sep=",", header=0,
                            names=["RecordID","SAPS-I","SOFA","Length_of_stay","Survival","In-hospital_death"])
df_outcomes_b = pd.read_csv('../rawdata/Outcomes-b.txt', sep=",", header=0,
                            names=["RecordID","SAPS-I","SOFA","Length_of_stay","Survival","In-hospital_death"])
df_outcomes_c = pd.read_csv('../rawdata/Outcomes-c.txt', sep=",", header=0,
                            names=["RecordID","SAPS-I","SOFA","Length_of_stay","Survival","In-hospital_death"])
print(df_outcomes_a.head(n=5))
print(df_outcomes_b.head(n=5))
print(df_outcomes_c.head(n=5))

arr_outcomes_a = np.array(df_outcomes_a)
arr_outcomes_b = np.array(df_outcomes_b)
arr_outcomes_c = np.array(df_outcomes_c)

n_a = arr_outcomes_a.shape[0]
n_b = arr_outcomes_b.shape[0]
n_c = arr_outcomes_c.shape[0]
print('n_a = %d, n_b = %d, n_c = %d' % (n_a,n_b,n_c))

# merge dataframes
arr_outcomes = np.concatenate([arr_outcomes_a, arr_outcomes_b, arr_outcomes_c], axis=0)
n = arr_outcomes.shape[0]
print(arr_outcomes.shape)

y_inhospdeath = arr_outcomes[:,-1]
print("Percentage of in-hosp death: %.2f%%" % (np.sum(y_inhospdeath)/n*100))
print(y_inhospdeath.shape)

# Store outcomes in npy format
np.save('../processed_data/arr_outcomes.npy', arr_outcomes)
print('arr_outcomes.npy saved')

# arr_outcomes = np.load('phy12_outcomes.npy')
# print(arr_outcomes.shape)

# extract all parameters encountered across all patients
def extract_unq_params(path):
    cnt = 0
    for f in os.listdir(path):
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '.txt':
            df_temp = pd.read_csv(path+file_name+'.txt', sep=",", header=1, names=["time", "param", "value"])
            arr_data_temp = np.array(df_temp)
            params_temp = arr_data_temp[:, 1]
            if cnt==0:
                params_all = params_temp
            else:
                params_all = np.concatenate([params_all, params_temp], axis=0)
            cnt += 1
    params_all = list(params_all)
    params_all = [p for p in params_all if str(p) != 'nan']
    param_list = list(np.unique(np.array(params_all)))
    return param_list

param_list_a = extract_unq_params('../rawdata/set-a/')
param_list_b = extract_unq_params('../rawdata/set-b/')
param_list_c = extract_unq_params('../rawdata/set-c/')

param_list = param_list_a + param_list_b + param_list_c
param_list = list(np.unique(param_list))

# remove 5 fields
param_list.remove("Gender")
param_list.remove("Height")
param_list.remove("Weight")
param_list.remove("Age")
param_list.remove("ICUType")

print("Parameters: ", param_list)
print("Number of total parameters:", len(param_list))

# save variable names
np.save('../processed_data/ts_params.npy', param_list)
print('ts_params.npy: the names of 36 variables')

static_param_list = ['Age','Gender','Height','ICUType','Weight']
np.save('../processed_data/static_params.npy', static_param_list)
print('save names of static descriptors: static_params.npy')


def parse_all(path):
    P_list = []
    cnt = 0
    allfiles = os.listdir(path)
    allfiles.sort()
    for f in allfiles:
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '.txt':
            df = pd.read_csv(path+file_name+'.txt', sep=",", header=1, names=["time", "param", "value"])
            df_demogr = df.iloc[0:5]
            df_data = df.iloc[5:]
            
            arr_demogr = np.array(df_demogr)
            arr_data = np.array(df_data)

            my_dict = {'id': file_name}
            my_dict['static'] = (arr_demogr[0,2], arr_demogr[1,2], arr_demogr[2,2], arr_demogr[3,2], arr_demogr[4,2])

            # time-series
            n_pts = arr_data.shape[0]
            ts_list = []
            for i in range(n_pts):  # for each line
                param = arr_data[i, 1]  # the name of variables
                if param in param_list:
                    ts = arr_data[i, 0]  # time stamp
                    hrs, mins = float(ts[0:2]), float(ts[3:5])
                    value = arr_data[i, 2]  # value of variable
                    totalmins = 60.0*hrs + mins
                    ts_list.append((hrs,mins,totalmins,param,value))
            my_dict['ts'] = ts_list
            
            # append patient dictionary in master dictionary
            P_list.append(my_dict)
            cnt += 1
    return P_list

# Merge lists of patients into master list
p_list_a = parse_all('../rawdata/set-a/')
p_list_b = parse_all('../rawdata/set-b/')
p_list_c = parse_all('../rawdata/set-c/')
P_list = p_list_a + p_list_b + p_list_c
print('Length of P_list', len(P_list))

np.save('../processed_data/P_list.npy', P_list)
print('P_list.npy saved')
