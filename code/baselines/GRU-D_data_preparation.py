import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt


inputpath_1 = '../../P12data/rawdata/set-a/'
inputpath_2 = '../../P12data/rawdata/set-b/'
inputpath_3 = '../../P12data/rawdata/set-c/'

inputdict = {
    "ALP" : 0,             # o
    "ALT" : 1,             # o
    "AST" : 2,             # o
    "Albumin" : 3,         # o
    "BUN" : 4,             # o
    "Bilirubin" : 5,       # o
    "Cholesterol" : 6,     # o
    "Creatinine" : 7,      # o
    "DiasABP" : 8,         # o
    "FiO2" : 9,            # o
    "GCS" : 10,            # o
    "Glucose" : 11,        # o
    "HCO3" : 12,           # o
    "HCT" : 13,            # o
    "HR" : 14,             # o
    "K" : 15,              # o
    "Lactate" : 16,        # o
    "MAP" : 17,            # o
    "Mg" : 18,             # o
    "Na" : 19,             # o
    "PaCO2" : 20,          # o
    "PaO2" : 21,           # o
    "Platelets" : 22,      # o
    "RespRate" : 23,       # o
    "SaO2" : 24,           # o
    "SysABP" : 25,         # o
    "Temp" : 26,           # o
    "Tropl" : 27,          # o
    "TroponinI" : 27,      # temp: regarded same as Tropl
    "TropT" : 28,          # o
    "TroponinT" : 28,      # temp: regarded same as TropT
    "Urine" : 29,          # o
    "WBC" : 30,            # o
    "Weight" : 31,         # o
    "pH" : 32,             # o
    "NIDiasABP" : 33,      # unused variable
    "NIMAP" : 34,          # unused variable
    "NISysABP" : 35,       # unused variable
    "MechVent" : 36,       # unused variable
    "RecordID" : 37,       # unused variable
    "Age" : 38,            # unused variable
    "Gender" :39,          # unused variable
    "ICUType" : 40,        # unused variable
    "Height": 41           # unused variable
}


# functions to process the time in the data
def timeparser(time):
    return pd.to_timedelta(time + ':00')


def timedelta_to_day_figure(timedelta):
    return timedelta.days + (timedelta.seconds/86400)  # (24*60*60)


# group the data by time
def df_to_inputs(df, inputdict, inputs):
    grouped_data = df.groupby('Time')

    for row_index, value in df.iterrows():

        # t = colum ~ time frame
        # agg_no = row ~ variable

        # print(value.Parameter, type(value.Parameter))
        if isinstance(value.Parameter, str) or (isinstance(value.Parameter, float) and not math.isnan(value.Parameter)):
            agg_no = inputdict[value.Parameter]

            #print('agg_no : {}\t  value : {}'.format(agg_no, value.Value))
            inputs[agg_no].append(value.Value)

    return inputs


inputs = []

# prepare empty list to put data
# len(inputdict)-2: two items has same agg_no
for i in range(len(inputdict)-2):
    t = []
    inputs.append(t)

# read all the files in the input folder
for filename in os.listdir(inputpath_1):
    df = pd.read_csv(inputpath_1 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)

    inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

for filename in os.listdir(inputpath_2):
    df = pd.read_csv(inputpath_2 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)

    inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

for filename in os.listdir(inputpath_3):
    df = pd.read_csv(inputpath_3 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)

    inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

print(inputs[0][0])

# save inputs just in case
np.save('saved/inputs.npy', inputs, allow_pickle=True)
loaded_inputs = np.load('saved/inputs.npy', allow_pickle=True)
print(loaded_inputs[0][0])



# make input items list
input_columns = list(inputdict.keys())


# remove two overlaped items
# "TroponinI" : 27, #temp
# "TroponinT" : 28, #temp

input_columns.remove("TroponinI")
input_columns.remove("TroponinT")
print(input_columns)
print(len(input_columns))


# describe the data
# print count, min, max, mean, median, std, var and histogram if hist == True
# return values as a list
def describe(inputs, input_columns, inputdict, hist=False):

    desc = []

    for i in range(len(inputdict)-2):
        input_arr = np.asarray(inputs[i])

        des = []

        des.append(input_arr.size)
        des.append(np.amin(input_arr))
        des.append(np.amax(input_arr))
        des.append(np.mean(input_arr))
        des.append(np.median(input_arr))
        des.append(np.std(input_arr))
        des.append(np.var(input_arr))

        desc.append(des)

        # print histgram
        if hist:
            a = np.hstack(input_arr)
            plt.hist(a, bins='auto')
            plt.title("Histogram about {}".format(input_columns[i]))
            plt.show()

        print('count: {}, min: {}, max: {}'.format(des[0], des[1], des[2]))
        print('mean: {}, median: {}, std: {}, var: {}'.format(des[3], des[4], des[5], des[6]))

    return desc


desc = describe(loaded_inputs, input_columns, inputdict, hist=False)
desc = np.asarray(desc)
print(desc.shape)

# save desc
# 0: count, 1: min, 2: max, 3: mean, 4: median, 5: std, 6: var
np.save('saved/desc.npy', desc)
loaded_desc = np.load('saved/desc.npy')


# def normalization(desc, inputs):
#     # for each catagory
#     for i in range(desc.shape[0]):
#         # for each value
#         for j in range(len(inputs[i])):
#             inputs[i][j] = (inputs[i][j] - desc[i][3])/desc[i][5]
#     return inputs



# dataframe to dataset

def df_to_x_m_d(df, inputdict, size, id_posistion, split):
    grouped_data = df.groupby('Time')

    #generate input vectors
    x = np.zeros((len(inputdict)-2, grouped_data.ngroups))
    masking = np.zeros((len(inputdict)-2, grouped_data.ngroups))
    delta = np.zeros((split, size))
    timetable = np.zeros(grouped_data.ngroups)
    id = 0

    all_x = np.zeros((split,1))

    s_dataset = np.zeros((3, split, size))

    if grouped_data.ngroups > size:

        # fill the x and masking vectors
        pre_time = pd.to_timedelta(0)
        t = 0
        for row_index, value in df.iterrows():

            # t = colum, time frame
            # agg_no = row, variable

            #print(value)
            if isinstance(value.Parameter, str) or (isinstance(value.Parameter, float) and not math.isnan(value.Parameter)):
                agg_no = inputdict[value.Parameter]

            # same timeline check.
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                timetable[t] = timedelta_to_day_figure(value.Time)

            #print('agg_no : {}\t t : {}\t value : {}'.format(agg_no, t, value.Value))
            x[agg_no, t] = value.Value
            masking[agg_no, t] = 1

        # # generate random index array
        # ran_index = np.random.choice(grouped_data.ngroups, size=size, replace=False)
        # ran_index.sort()
        # ran_index[0] = 0
        # ran_index[size-1] = grouped_data.ngroups-1

        # generate index that has most parameters and first/last one.
        ran_index = grouped_data.count()
        ran_index = ran_index.reset_index()
        ran_index = ran_index.sort_values('Value', ascending=False)
        ran_index = ran_index[:size]
        ran_index = ran_index.sort_index()
        ran_index = np.asarray(ran_index.index.values)
        ran_index[0] = 0
        ran_index[size-1] = grouped_data.ngroups-1

        #print(ran_index)

        # take id for outcome comparing
        id = x[id_posistion, 0]

        # remove unnesserly parts(rows)
        x = x[:split, :]
        masking = masking[:split, :]

        # coulme(time) sampling
        x_sample = np.zeros((split, size))
        m_sample = np.zeros((split, size))
        time_sample = np.zeros(size)

        t_x_sample = x_sample.T
        t_marsking = m_sample.T
        #t_time = t_sample.T

        t_x = x.T
        t_m = masking.T
        #t_t = t.T

        it = np.nditer(ran_index, flags=['f_index'])
        while not it.finished:
            #print('it.index = {}, it[0] = {}, ran_index = {}'.format(it.index, it[0], ran_index[it.index]))
            t_x_sample[it.index] = t_x[it[0]]
            t_marsking[it.index] = t_m[it[0]]
            time_sample[it.index] = timetable[it[0]]
            it.iternext()

        x = x_sample
        masking = m_sample
        timetable = time_sample

        # # normalize the X
        # nor_x = x/max_input[:, np.newaxis]

        # fill the delta vectors
        for index, value in np.ndenumerate(masking):

            # index[0] = row, agg
            # index[1] = col, time

            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1]-1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1] + delta[index[0], index[1]-1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1]

    else:

        # fill the x and masking vectors
        pre_time = pd.to_timedelta(0)
        t = 0
        for row_index, value in df.iterrows():

            # t = colum, time frame
            # agg_no = row, variable

            #print(value)
            if isinstance(value.Parameter, str) or (isinstance(value.Parameter, float) and not math.isnan(value.Parameter)):
                agg_no = inputdict[value.Parameter]

            # same timeline check.
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                timetable[t] = timedelta_to_day_figure(value.Time)

            #print('agg_no : {}\t t : {}\t value : {}'.format(agg_no, t, value.Value))
            x[agg_no, t] = value.Value
            masking[agg_no, t] = 1

        # take id for outcome comparing
        id = x[id_posistion, 0]

        # remove unnesserly parts(rows)
        x = x[:split, :]
        masking = masking[:split, :]

        x = np.pad(x, ((0,0), (size-grouped_data.ngroups, 0)), 'constant')
        masking = np.pad(masking, ((0,0), (size-grouped_data.ngroups, 0)), 'constant')
        timetable = np.pad(timetable, (size-grouped_data.ngroups, 0), 'constant')

        # # normalize the X
        # nor_x = x/max_input[:, np.newaxis]

        # fill the delta vectors
        for index, value in np.ndenumerate(masking):

            # index[0] = row, agg
            # index[1] = col, time

            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1]-1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1] + delta[index[0], index[1]-1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1]


    all_x = np.concatenate((all_x, x), axis=1)
    all_x = all_x[:,1:]

    s_dataset[0] = x
    s_dataset[1] = masking
    s_dataset[2] = delta

    return s_dataset, all_x, id


# def df_to_x_m_d(df, inputdict, mean, std, size, id_posistion, split):
size = 49  # steps ~ from the paper
id_posistion = 37
input_length = 33  # input variables ~ from the paper
dataset = np.zeros((1, 3, input_length, size))

all_x_add = np.zeros((input_length, 1))

for filename in os.listdir(inputpath_1):
    df = pd.read_csv(inputpath_1 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)
    s_dataset, all_x, id = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion, split=input_length)

    dataset = np.concatenate((dataset, s_dataset[np.newaxis, :,:,:]))
    all_x_add = np.concatenate((all_x_add, all_x), axis=1)

for filename in os.listdir(inputpath_2):
    df = pd.read_csv(inputpath_2 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)
    s_dataset, all_x, id = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                       split=input_length)

    dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
    all_x_add = np.concatenate((all_x_add, all_x), axis=1)

for filename in os.listdir(inputpath_3):
    df = pd.read_csv(inputpath_3 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)
    s_dataset, all_x, id = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                       split=input_length)

    dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
    all_x_add = np.concatenate((all_x_add, all_x), axis=1)


dataset = dataset[1:, :,:,:]
# (total datasets, kind of data(x, masking, and delta), input length, num of varience)
# (4000, 3, 33, 49)
print(dataset.shape)
print(dataset[0].shape)
print(dataset[0][0][0])

print(all_x_add.shape)
all_x_add = all_x_add[:, 1:]
print(all_x_add.shape)

train_proportion = 0.8
train_index = int(all_x_add.shape[1] * train_proportion)
train_x = all_x_add[:, :train_index]
print(train_x.shape)


def get_mean(x):
    x_mean = []
    for i in range(x.shape[0]):
        mean = np.mean(x[i])
        x_mean.append(mean)
    return x_mean


def get_median(x):
    x_median = []
    for i in range(x.shape[0]):
        median = np.median(x[i])
        x_median.append(median)
    return x_median


def get_std(x):
    x_std = []
    for i in range(x.shape[0]):
        std = np.std(x[i])
        x_std.append(std)
    return x_std


def get_var(x):
    x_var = []
    for i in range(x.shape[0]):
        var = np.var(x[i])
        x_var.append(var)
    return x_var


x_mean = get_mean(train_x)
print(x_mean)
print(len(x_mean))

x_std = get_std(train_x)
print(x_std)
print(len(x_std))


# dataset shape : (4000, 3, 33, 49)
def dataset_normalize(dataset, mean, std):
    for i in range(dataset.shape[0]):
        dataset[i][0] = (dataset[i][0] - mean[:, None])
        dataset[i][0] = dataset[i][0]/std[:, None]

    return dataset


x_mean = np.asarray(x_mean)
x_std = np.asarray(x_std)

dataset = dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)
print(dataset[0][0][0])


def normalize_chk(dataset):
    all_x_add = np.zeros((dataset[0][0].shape[0],1))
    for i in range(dataset.shape[0]):
        all_x_add = np.concatenate((all_x_add, dataset[i][0]), axis=1)

    mean = get_mean(all_x_add)
    median = get_median(all_x_add)
    std = get_std(all_x_add)
    var = get_var(all_x_add)

    print('mean')
    print(mean)
    print('median')
    print(median)
    print('std')
    print(std)
    print('var')
    print(var)

    return mean, median, std, var


nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

np.save('saved/x_mean_aft_nor', nor_mean)
np.save('saved/x_median_aft_nor', nor_median)
np.save('saved/dataset.npy', dataset)


t_dataset = np.load('saved/dataset.npy')
print(t_dataset.shape)


# only check In-hospital_death
def df_to_y1(df):
    output = df.values
    output = output[:, 5:]

    return output


A_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-a.txt')
B_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-b.txt')
C_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-c.txt')

y_a_outcomes = df_to_y1(A_outcomes)
y_b_outcomes = df_to_y1(B_outcomes)
y_c_outcomes = df_to_y1(C_outcomes)

y1_outcomes = np.concatenate((y_a_outcomes, y_b_outcomes, y_c_outcomes))
print(y1_outcomes.shape)
np.save('saved/y1_out', y1_outcomes)

