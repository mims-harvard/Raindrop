import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt


# functions to process the time in the data
def timeparser(time):
    return pd.to_timedelta(time + ':00')


def timedelta_to_day_figure(timedelta):
    return timedelta.days + (timedelta.seconds/86400)


def df_to_inputs(df, inputdict, inputs):    # group the data by time
    grouped_data = df.groupby('Time')
    for row_index, value in df.iterrows():
        if isinstance(value.Parameter, str) or (isinstance(value.Parameter, float) and not math.isnan(value.Parameter)):
            agg_no = inputdict[value.Parameter]
            inputs[agg_no].append(value.Value)
    return inputs


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

        if hist:
            a = np.hstack(input_arr)
            plt.hist(a, bins='auto')
            plt.title("Histogram about {}".format(input_columns[i]))
            plt.show()

        print('count: {}, min: {}, max: {}'.format(des[0], des[1], des[2]))
        print('mean: {}, median: {}, std: {}, var: {}'.format(des[3], des[4], des[5], des[6]))

    return desc


def df_to_x_m_d(df, inputdict, size, id_posistion, split, dataset_name='P12'):
    grouped_data = df.groupby('Time')

    #generate input vectors
    if dataset_name == 'P12':
        x = np.zeros((len(inputdict) - 2, grouped_data.ngroups))
        masking = np.zeros((len(inputdict) - 2, grouped_data.ngroups))
    elif dataset_name == 'P19' or dataset_name == 'eICU':
        x = np.zeros((len(inputdict), grouped_data.ngroups))
        masking = np.zeros((len(inputdict), grouped_data.ngroups))
    elif dataset_name == 'PAM':
        x = np.zeros((len(inputdict), 600))
        masking = np.zeros((len(inputdict), 600))

    delta = np.zeros((split, size))
    if dataset_name == 'PAM':
        timetable = np.zeros(600)
    else:
        timetable = np.zeros(grouped_data.ngroups)
    id = 0

    all_x = np.zeros((split,1))

    s_dataset = np.zeros((3, split, size))

    if grouped_data.ngroups > size:
        # fill the x and masking vectors
        if dataset_name == 'P12':
            pre_time = pd.to_timedelta(0)
        elif dataset_name == 'P19' or dataset_name == 'eICU' or dataset_name == 'PAM':
            pre_time = 0

        t = 0
        for row_index, value in df.iterrows():
            if isinstance(value.Parameter, str) or (isinstance(value.Parameter, float) and not math.isnan(value.Parameter)):
                agg_no = inputdict[value.Parameter]

            # same timeline check.
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                if dataset_name == 'P12':
                    timetable[t] = timedelta_to_day_figure(value.Time)
                elif dataset_name == 'P19' or dataset_name == 'eICU' or dataset_name == 'PAM':
                    timetable[t] = value.Time

            x[agg_no, t] = value.Value
            masking[agg_no, t] = 1

        # generate index that has most parameters and first/last one.
        ran_index = grouped_data.count()
        ran_index = ran_index.reset_index()
        ran_index = ran_index.sort_values('Value', ascending=False)
        ran_index = ran_index[:size]
        ran_index = ran_index.sort_index()
        ran_index = np.asarray(ran_index.index.values)
        ran_index[0] = 0
        ran_index[size-1] = grouped_data.ngroups-1

        # take id for outcome comparing
        id = x[id_posistion, 0]

        x = x[:split, :]
        masking = masking[:split, :]

        x_sample = np.zeros((split, size))
        m_sample = np.zeros((split, size))
        time_sample = np.zeros(size)

        t_x_sample = x_sample.T
        t_marsking = m_sample.T

        t_x = x.T
        t_m = masking.T

        it = np.nditer(ran_index, flags=['f_index'])
        while not it.finished:
            t_x_sample[it.index] = t_x[it[0]]
            t_marsking[it.index] = t_m[it[0]]
            time_sample[it.index] = timetable[it[0]]
            it.iternext()

        x = x_sample
        masking = m_sample
        timetable = time_sample

        # fill the delta vectors
        for index, value in np.ndenumerate(masking):
            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1]-1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1] + delta[index[0], index[1]-1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1]

    else:
        # fill the x and masking vectors
        if dataset_name == 'P12':
            pre_time = pd.to_timedelta(0)
        elif dataset_name == 'P19' or dataset_name == 'eICU' or dataset_name == 'PAM':
            pre_time = 0

        t = 0
        for row_index, value in df.iterrows():
            if isinstance(value.Parameter, str) or (isinstance(value.Parameter, float) and not math.isnan(value.Parameter)):
                agg_no = inputdict[value.Parameter]

            # same timeline check.
            if pre_time != value.Time:
                pre_time = value.Time
                t += 1
                if dataset_name == 'P12':
                    timetable[t] = timedelta_to_day_figure(value.Time)
                elif dataset_name == 'P19' or dataset_name == 'eICU' or dataset_name == 'PAM':
                    timetable[t] = value.Time

            x[agg_no, t] = value.Value
            masking[agg_no, t] = 1

        # take id for outcome comparing
        id = x[id_posistion, 0]

        x = x[:split, :]
        masking = masking[:split, :]

        x = np.pad(x, ((0,0), (size-grouped_data.ngroups, 0)), 'constant')
        masking = np.pad(masking, ((0,0), (size-grouped_data.ngroups, 0)), 'constant')
        timetable = np.pad(timetable, (size-grouped_data.ngroups, 0), 'constant')

        # fill the delta vectors
        for index, value in np.ndenumerate(masking):
            if index[1] == 0:
                delta[index[0], index[1]] = 0
            elif masking[index[0], index[1]-1] == 0:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1] + delta[index[0], index[1]-1]
            else:
                delta[index[0], index[1]] = timetable[index[1]] - timetable[index[1]-1]

    all_x = np.concatenate((all_x, x), axis=1)
    all_x = all_x[:, 1:]

    s_dataset[0] = x
    s_dataset[1] = masking
    s_dataset[2] = delta

    return s_dataset, all_x, id


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


def dataset_normalize(dataset, mean, std):
    for i in range(dataset.shape[0]):
        dataset[i][0] = (dataset[i][0] - mean[:, None])
        dataset[i][0] = dataset[i][0]/std[:, None]
    return dataset


def normalize_chk(dataset):
    all_x_add = np.zeros((dataset[0][0].shape[0],1))
    for i in range(dataset.shape[0]):
        all_x_add = np.concatenate((all_x_add, dataset[i][0]), axis=1)

    mean = get_mean(all_x_add)
    median = get_median(all_x_add)
    std = get_std(all_x_add)
    var = get_var(all_x_add)

    return mean, median, std, var


def df_to_y1(df):
    output = df.values
    output = output[:, 5:]  # for mortality

    # # for LoS
    # output = output[:, 3]
    # output = np.array(list(map(lambda los: 0 if los <= 3 else 1, output)))[..., np.newaxis]
    return output


if __name__ == '__main__':
    dataset_name = 'P12'  # possible values: 'P12', 'P19', 'eICU', 'PAM'
    print('Dataset used: ', dataset_name)

    if dataset_name == 'P12':
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

        # save inputs just in case
        np.save('saved/inputs.npy', inputs, allow_pickle=True)
        loaded_inputs = np.load('saved/inputs.npy', allow_pickle=True)

        # make input items list
        input_columns = list(inputdict.keys())

        input_columns.remove("TroponinI")
        input_columns.remove("TroponinT")

        desc = describe(loaded_inputs, input_columns, inputdict, hist=False)
        desc = np.asarray(desc)
        print(desc.shape)

        # 0: count, 1: min, 2: max, 3: mean, 4: median, 5: std, 6: var
        np.save('saved/desc.npy', desc)
        loaded_desc = np.load('saved/desc.npy')

        size = 49  # steps ~ from the paper
        id_posistion = 37
        input_length = 33  # input variables ~ from the paper
        dataset = np.zeros((1, 3, input_length, size))

        all_x_add = np.zeros((input_length, 1))

        for filename in os.listdir(inputpath_1):
            df = pd.read_csv(inputpath_1 + filename, header=0, parse_dates=['Time'], date_parser=timeparser)
            s_dataset, all_x, id = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion, split=input_length)

            dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
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

        dataset = dataset[1:, :, :, :]
        all_x_add = all_x_add[:, 1:]

        train_proportion = 0.8
        train_index = int(all_x_add.shape[1] * train_proportion)
        train_x = all_x_add[:, :train_index]

        x_mean = get_mean(train_x)
        x_std = get_std(train_x)

        x_mean = np.asarray(x_mean)
        x_std = np.asarray(x_std)

        dataset = dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)

        nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

        np.save('saved/x_mean_aft_nor', nor_mean)
        np.save('saved/x_median_aft_nor', nor_median)
        np.save('saved/dataset.npy', dataset)

        t_dataset = np.load('saved/dataset.npy')
        print(t_dataset.shape)

        A_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-a.txt')
        B_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-b.txt')
        C_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-c.txt')

        y_a_outcomes = df_to_y1(A_outcomes)
        y_b_outcomes = df_to_y1(B_outcomes)
        y_c_outcomes = df_to_y1(C_outcomes)

        y1_outcomes = np.concatenate((y_a_outcomes, y_b_outcomes, y_c_outcomes))
        print(y1_outcomes.shape)
        np.save('saved/y1_out', y1_outcomes)

    elif dataset_name == 'P19':
        data = np.load('../../P19data/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        labels_ts = np.load('../../P19data/processed_data/labels_ts.npy', allow_pickle=True)
        labels_static = np.load('../../P19data/processed_data/labels_demogr.npy', allow_pickle=True)

        all_labels = np.concatenate((labels_static, labels_ts))

        inputdict = {label: i for i, label in enumerate(all_labels)}

        inputs = [[] for i in range(len(inputdict))]
        i = 0
        for patient in data:
            if i % 1000 == 0:
                print(i)
            i += 1

            length = patient['length']
            arr = patient['arr']
            time = patient['time']

            # static to df
            time_df = [0] * len(labels_static)
            parameter_df = list(labels_static)
            value_df = list(patient['extended_static'])

            # time series to df
            observations_indices = np.nonzero(arr)
            for x, y in zip(*observations_indices):
                time_df.append(x)
                parameter_df.append(labels_ts[y])
                value_df.append(arr[x, y])

            df = pd.DataFrame(data={'Time': time_df, 'Parameter': parameter_df, 'Value': value_df})
            inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

        # save inputs just in case
        np.save('saved/P19_inputs.npy', inputs, allow_pickle=True)

        loaded_inputs = np.load('saved/P19_inputs.npy', allow_pickle=True)

        # make input items list
        input_columns = list(inputdict.keys())

        size = 49  # steps ~ from the paper
        id_posistion = 37  # not used
        input_length = len(all_labels)  # input variables
        dataset = np.zeros((1, 3, input_length, size))
        all_x_add = np.zeros((input_length, 1))

        i = 0
        for patient in data:
            if i % 1000 == 0:
                print(i)
            i += 1
            length = patient['length']
            arr = patient['arr']
            time = patient['time']

            # static to df
            time_df = [0] * len(labels_static)
            parameter_df = list(labels_static)
            value_df = list(patient['extended_static'])

            # time series to df
            observations_indices = np.nonzero(arr)
            for x, y in zip(*observations_indices):
                time_df.append(x)
                parameter_df.append(labels_ts[y])
                value_df.append(arr[x, y])

            df = pd.DataFrame(data={'Time': time_df, 'Parameter': parameter_df, 'Value': value_df})

            s_dataset, all_x, _ = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                              split=input_length, dataset_name=dataset_name)

            dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
            all_x_add = np.concatenate((all_x_add, all_x), axis=1)

        dataset = dataset[1:, :, :, :]

        all_x_add = all_x_add[:, 1:]

        train_proportion = 0.8
        train_index = int(all_x_add.shape[1] * train_proportion)
        train_x = all_x_add[:, :train_index]

        x_mean = get_mean(train_x)
        x_std = get_std(train_x)

        x_mean = np.asarray(x_mean)
        x_std = np.asarray(x_std)

        dataset = dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)
        dataset = dataset[:, :, :-1, :]

        nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

        np.save('saved/P19_x_mean_aft_nor', nor_mean)
        np.save('saved/P19_x_median_aft_nor', nor_median)
        np.save('saved/P19_dataset.npy', dataset)

        t_dataset = np.load('saved/P19_dataset.npy')

        # labels
        y1_outcomes = np.load('../../P19data/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        np.save('saved/' + dataset_name + '_y1_out', y1_outcomes)

    elif dataset_name == 'eICU':
        data = np.load('../../eICUdata/processed_data/PTdict_list.npy', allow_pickle=True)
        labels_ts = np.load('../../eICUdata/processed_data/eICU_ts_vars.npy', allow_pickle=True)
        labels_static = np.load('../../eICUdata/processed_data/eICU_static_vars.npy', allow_pickle=True)[-2:]   # only height and weight

        all_labels = np.concatenate((labels_static, labels_ts))

        inputdict = {label: i for i, label in enumerate(all_labels)}

        # prepare empty list to put data
        inputs = [[] for i in range(len(inputdict))]
        i = 0
        for patient in data:
            if i % 1000 == 0:
                print(i)
            i += 1

            arr = patient['arr']
            time = patient['time']

            # static to df
            time_df = [0] * len(labels_static)
            parameter_df = list(labels_static)
            value_df = list(patient['extended_static'][-2:])

            # time series to df
            observations_indices = np.nonzero(arr)
            for x, y in zip(*observations_indices):
                time_df.append(x)
                parameter_df.append(labels_ts[y])
                value_df.append(arr[x, y])

            df = pd.DataFrame(data={'Time': time_df, 'Parameter': parameter_df, 'Value': value_df})
            inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

        # save inputs just in case
        np.save('saved/eICU_inputs.npy', inputs, allow_pickle=True)

        loaded_inputs = np.load('saved/eICU_inputs.npy', allow_pickle=True)

        # make input items list
        input_columns = list(inputdict.keys())

        size = 49  # steps ~ from the paper
        id_posistion = 0  # not used
        input_length = len(all_labels)  # input variables
        dataset = np.zeros((1, 3, input_length, size))
        all_x_add = np.zeros((input_length, 1))

        i = 0
        for patient in data:
            if i % 1000 == 0:
                print(i)
            i += 1

            arr = patient['arr']
            time = patient['time']

            # static to df
            time_df = [0] * len(labels_static)
            parameter_df = list(labels_static)
            value_df = list(patient['extended_static'][-2:])

            # time series to df
            observations_indices = np.nonzero(arr)
            for x, y in zip(*observations_indices):
                time_df.append(x)
                parameter_df.append(labels_ts[y])
                value_df.append(arr[x, y])

            df = pd.DataFrame(data={'Time': time_df, 'Parameter': parameter_df, 'Value': value_df})

            s_dataset, all_x, _ = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                              split=input_length, dataset_name=dataset_name)

            dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
            all_x_add = np.concatenate((all_x_add, all_x), axis=1)

        dataset = dataset[1:, :, :, :]

        all_x_add = all_x_add[:, 1:]

        train_proportion = 0.8
        train_index = int(all_x_add.shape[1] * train_proportion)
        train_x = all_x_add[:, :train_index]

        x_mean = get_mean(train_x)

        x_std = get_std(train_x)

        x_mean = np.asarray(x_mean)
        x_std = np.asarray(x_std)

        dataset = dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)

        nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

        np.save('saved/eICU_x_mean_aft_nor', nor_mean)
        np.save('saved/eICU_x_median_aft_nor', nor_median)
        np.save('saved/eICU_dataset.npy', dataset)

        t_dataset = np.load('saved/eICU_dataset.npy')
        print(t_dataset.shape)

        # labels
        y1_outcomes = np.load('../../eICUdata/processed_data/arr_outcomes.npy', allow_pickle=True)
        y1_outcomes = y1_outcomes[..., np.newaxis]
        np.save('saved/' + dataset_name + '_y1_out', y1_outcomes)

    elif dataset_name == 'PAM':
        data = np.load('../../PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)

        n_sensors = 17
        all_labels = np.array(['sensor_%d' % i for i in range(n_sensors)])
        inputdict = {label: i for i, label in enumerate(all_labels)}

        # prepare empty list to put data
        inputs = [[] for i in range(n_sensors)]
        i = 0
        for patient in data:
            if i % 1000 == 0:
                print(i)
            i += 1

            arr = patient
            time = np.arange(arr.shape[0])
            time = time[:, np.newaxis]

            # time series to df
            time_df = []
            parameter_df = []
            value_df = []
            observations_indices = np.nonzero(arr)
            for x, y in zip(*observations_indices):
                time_df.append(x)
                parameter_df.append(all_labels[y])
                value_df.append(arr[x, y])

            df = pd.DataFrame(data={'Time': time_df, 'Parameter': parameter_df, 'Value': value_df})
            inputs = df_to_inputs(df=df, inputdict=inputdict, inputs=inputs)

        # save inputs just in case
        np.save('saved/PAM_inputs.npy', inputs, allow_pickle=True)

        loaded_inputs = np.load('saved/PAM_inputs.npy', allow_pickle=True)

        # make input items list
        input_columns = list(inputdict.keys())

        size = 49  # steps ~ from the paper
        id_posistion = 0  # not used
        input_length = len(all_labels)  # input variables
        dataset = np.zeros((1, 3, input_length, size))
        all_x_add = np.zeros((input_length, 1))

        i = 0
        for patient in data:
            if i % 1000 == 0:
                print(i)
            i += 1

            arr = patient
            time = np.arange(arr.shape[0])
            time = time[:, np.newaxis]

            # time series to df
            time_df = []
            parameter_df = []
            value_df = []
            observations_indices = np.nonzero(arr)
            for x, y in zip(*observations_indices):
                time_df.append(x)
                parameter_df.append(all_labels[y])
                value_df.append(arr[x, y])

            df = pd.DataFrame(data={'Time': time_df, 'Parameter': parameter_df, 'Value': value_df})

            s_dataset, all_x, _ = df_to_x_m_d(df=df, inputdict=inputdict, size=size, id_posistion=id_posistion,
                                              split=input_length, dataset_name=dataset_name)

            dataset = np.concatenate((dataset, s_dataset[np.newaxis, :, :, :]))
            all_x_add = np.concatenate((all_x_add, all_x), axis=1)

        dataset = dataset[1:, :, :, :]

        all_x_add = all_x_add[:, 1:]

        train_proportion = 0.8
        train_index = int(all_x_add.shape[1] * train_proportion)
        train_x = all_x_add[:, :train_index]

        x_mean = get_mean(train_x)
        x_std = get_std(train_x)

        x_mean = np.asarray(x_mean)
        x_std = np.asarray(x_std)

        dataset = dataset_normalize(dataset=dataset, mean=x_mean, std=x_std)

        nor_mean, nor_median, nor_std, nor_var = normalize_chk(dataset)

        np.save('saved/PAM_x_mean_aft_nor', nor_mean)
        np.save('saved/PAM_x_median_aft_nor', nor_median)
        np.save('saved/PAM_dataset.npy', dataset)

        t_dataset = np.load('saved/PAM_dataset.npy')
        print(t_dataset.shape)

        # labels
        y1_outcomes = np.load('../../PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)
        np.save('saved/' + dataset_name + '_y1_out', y1_outcomes)

