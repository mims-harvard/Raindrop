"""
Code originates from GRUD_mean.ipynb from gitHub repository https://github.com/Han-JD/GRU-D.
"""

import torch
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import average_precision_score
from utils_baselines import random_sample

'''
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
np.save('saved/dataset', dataset)


t_dataset = np.load('saved/dataset.npy')
print(t_dataset.shape)



# # Y values
# 
# def df_to_y3(df):
#     
#     # RecordID  SAPS-I  SOFA  Length_of_stay  Survival  In-hospital_death
#     
#     output = np.zeros((4000,3))
#     
#     for row_index, value in df.iterrows():
#         los = value[3]  # Length_of_stay
#         sur = value[4]  # Survival
#         ihd = value[5]  # In-hospital_death
# 
#         output[row_index][0] = ihd
#         output[row_index][1] = ihd
# 
#         # length-of-stay less than 3 yes/no 1/0
#         if los < 3:
#             output[row_index][2] = 0
#         else:
#             output[row_index][2] = 1
# 
#     return output


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
'''

# def df_to_y2(df):
#     
#     # RecordID  SAPS-I  SOFA  Length_of_stay  Survival  In-hospital_death
#     
#     output = np.zeros((4000,2))
#     
#     for row_index, value in df.iterrows():
#         ihd = value[5] # In-hospital_death
# 
#         output[row_index][0] = ihd
#         output[row_index][1] = ihd
#         
#     return output
# 
# 
# A_outcomes = pd.read_csv('../../P12data/rawdata/Outcomes-a.txt')
# y2_outcomes = df_to_y2(A_outcomes)
# print(y2_outcomes.shape)
# np.save('saved/y2_out', y2_outcomes)


# define model


class GRUD(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0,
                 bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0.0):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.zeros = torch.autograd.Variable(torch.zeros(input_size))
        # self.x_mean = torch.autograd.Variable(torch.tensor(x_mean))
        self.x_mean = x_mean.clone().detach().requires_grad_(True)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
        
        ################################
        gate_size = 1 # not used
        ################################
        
        self._all_weights = []

        '''
        w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
        w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
        b_ih = Parameter(torch.Tensor(gate_size))
        b_hh = Parameter(torch.Tensor(gate_size))
        layer_params = (w_ih, w_hh, b_ih, b_hh)
        '''
        # decay rates gamma
        w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # z
        w_xz = torch.nn.Parameter(torch.Tensor(input_size))
        w_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mz = torch.nn.Parameter(torch.Tensor(input_size))

        # r
        w_xr = torch.nn.Parameter(torch.Tensor(input_size))
        w_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mr = torch.nn.Parameter(torch.Tensor(input_size))

        # h_tilde
        w_xh = torch.nn.Parameter(torch.Tensor(input_size))
        w_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
        w_mh = torch.nn.Parameter(torch.Tensor(input_size))

        # y (output)
        w_hy = torch.nn.Parameter(torch.Tensor(output_size, hidden_size))

        # bias
        b_dg_x = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_h = torch.nn.Parameter(torch.Tensor(hidden_size))
        b_y = torch.nn.Parameter(torch.Tensor(output_size))

        layer_params = (w_dg_x, w_dg_h,
                        w_xz, w_hz, w_mz,
                        w_xr, w_hr, w_mr,
                        w_xh, w_hh, w_mh,
                        w_hy,
                        b_dg_x, b_dg_h, b_z, b_r, b_h, b_y)

        param_names = ['weight_dg_x', 'weight_dg_h',
                       'weight_xz', 'weight_hz','weight_mz',
                       'weight_xr', 'weight_hr','weight_mr',
                       'weight_xh', 'weight_hh','weight_mh',
                       'weight_hy']
        if bias:
            param_names += ['bias_dg_x', 'bias_dg_h',
                            'bias_z',
                            'bias_r',
                            'bias_h',
                            'bias_y']
        
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()
        
    def flatten_parameters(self):
        """
        Resets parameter data pointer so that they can use faster code paths.
        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(GRUD, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        
        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)
    
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(GRUD, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []

        weights = ['weight_dg_x', 'weight_dg_h',
                   'weight_xz', 'weight_hz','weight_mz',
                   'weight_xr', 'weight_hr','weight_mr',
                   'weight_xh', 'weight_hh','weight_mh',
                   'weight_hy',
                   'bias_dg_x', 'bias_dg_h',
                   'bias_z', 'bias_r', 'bias_h','bias_y']

        if self.bias:
            self._all_weights += [weights]
        else:
            self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]
    
    def forward(self, input):
        # input.size = (3, 33,49) : num_input or num_hidden, num_layer or step
        X = torch.squeeze(input[0])  # .size = (33,49)
        Mask = torch.squeeze(input[1])  # .size = (33,49)
        Delta = torch.squeeze(input[2])  # .size = (33,49)
        Hidden_State = torch.autograd.Variable(torch.zeros(self.input_size))
        
        # step_size = X.size(1)  # 49
        #print('step size : ', step_size)
        
        output = None
        h = Hidden_State

        # decay rates gamma
        w_dg_x = getattr(self, 'weight_dg_x')
        w_dg_h = getattr(self, 'weight_dg_h')

        #z
        w_xz = getattr(self, 'weight_xz')
        w_hz = getattr(self, 'weight_hz')
        w_mz = getattr(self, 'weight_mz')

        # r
        w_xr = getattr(self, 'weight_xr')
        w_hr = getattr(self, 'weight_hr')
        w_mr = getattr(self, 'weight_mr')

        # h_tilde
        w_xh = getattr(self, 'weight_xh')
        w_hh = getattr(self, 'weight_hh')
        w_mh = getattr(self, 'weight_mh')

        # bias
        b_dg_x = getattr(self, 'bias_dg_x')
        b_dg_h = getattr(self, 'bias_dg_h')
        b_z = getattr(self, 'bias_z')
        b_r = getattr(self, 'bias_r')
        b_h = getattr(self, 'bias_h')
        
        for layer in range(self.num_layers):
            
            x = torch.squeeze(X[:,layer:layer+1])
            m = torch.squeeze(Mask[:,layer:layer+1])
            d = torch.squeeze(Delta[:,layer:layer+1])

            #(4)
            gamma_x = torch.exp(-torch.max(self.zeros, (w_dg_x * d + b_dg_x)))
            gamma_h = torch.exp(-torch.max(self.zeros, (w_dg_h * d + b_dg_h)))

            #(5)
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean)

            #(6)
            if self.dropout == 0:
                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))

                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''

                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = dropout(h_tilde)

                h = (1 - z)* h + z*h_tilde

            else:
                h = gamma_h * h

                z = torch.sigmoid((w_xz*x + w_hz*h + w_mz*m + b_z))
                r = torch.sigmoid((w_xr*x + w_hr*h + w_mr*m + b_r))
                h_tilde = torch.tanh((w_xh*x + w_hh*(r * h) + w_mh*m + b_h))

                h = (1 - z) * h + z * h_tilde
            
        w_hy = getattr(self, 'weight_hy')
        b_y = getattr(self, 'bias_y')

        output = torch.matmul(w_hy, h) + b_y
        output = torch.sigmoid(output)
        
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_dataloader(dataset, outcomes, upsampling_batch, batch_size, split_type, train_proportion=0.8, dev_proportion=0.1):
    # 80% train, 10% validation, 10% test
    # test data is the remaining part after training and validation set (default=0.1)

    if split_type == 'random':
        # shuffle data
        # np.random.seed(77)   # if you want the same permutation for each run
        permuted_idx = np.random.permutation(dataset.shape[0])
        dataset = dataset[permuted_idx]
        outcomes = outcomes[permuted_idx]

        train_index = int(np.floor(dataset.shape[0] * train_proportion))
        dev_index = int(np.floor(dataset.shape[0] * (train_proportion + dev_proportion)))

        # split dataset to train/dev/test set
        if upsampling_batch:
            train_data = []
            train_label = []
            idx_0 = np.where(outcomes[:train_index, :] == 0)[0]
            idx_1 = np.where(outcomes[:train_index, :] == 1)[0]
            for i in range(train_index // batch_size):
                indices = random_sample(idx_0, idx_1, batch_size)
                train_data.extend(dataset[indices, :, :, :])
                train_label.extend(outcomes[indices, :])
            train_data = np.array(train_data)
            train_label = np.array(train_label)
        else:
            train_data, train_label = dataset[:train_index, :, :, :], outcomes[:train_index, :]

        dev_data, dev_label = dataset[train_index:dev_index, :, :, :], outcomes[train_index:dev_index, :]
        test_data, test_label = dataset[dev_index:, :, :, :], outcomes[dev_index:, :]
    elif split_type == 'age' or split_type == 'gender':
        # # calculate and save statistics
        # idx_under_65 = []
        # idx_over_65 = []
        # idx_male = []
        # idx_female = []
        #
        # P_list = np.load('../../P12data/processed_data/P_list.npy', allow_pickle=True)
        #
        # for i in range(len(P_list)):
        #     age, gender, _, _, _ = P_list[i]['static']
        #     if age > 0:
        #         if age < 65:
        #             idx_under_65.append(i)
        #         else:
        #             idx_over_65.append(i)
        #     if gender == 0:
        #         idx_female.append(i)
        #     if gender == 1:
        #         idx_male.append(i)
        #
        # np.save('saved/grud_idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
        # np.save('saved/grud_idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
        # np.save('saved/grud_idx_male.npy', np.array(idx_male), allow_pickle=True)
        # np.save('saved/grud_idx_female.npy', np.array(idx_female), allow_pickle=True)

        if split_type == 'age':
            idx_train = np.load('saved/grud_idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load('saved/grud_idx_over_65.npy', allow_pickle=True)
        else:   # split_type == 'gender':
            idx_train = np.load('saved/grud_idx_male.npy', allow_pickle=True)
            idx_vt = np.load('saved/grud_idx_female.npy', allow_pickle=True)

        if upsampling_batch:
            train_data = []
            train_label = []
            idx_0 = idx_train[np.where(outcomes[idx_train, :] == 0)[0]]
            idx_1 = idx_train[np.where(outcomes[idx_train, :] == 1)[0]]
            for i in range(len(idx_train) // batch_size):   # last small batch is dropped
                indices = random_sample(idx_0, idx_1, batch_size)
                train_data.extend(dataset[indices, :, :, :])
                train_label.extend(outcomes[indices, :])
            train_data = np.array(train_data)
            train_label = np.array(train_label)
        else:
            train_data, train_label = dataset[idx_train, :, :, :], outcomes[idx_train, :]

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

        dev_data, dev_label = dataset[idx_val, :, :, :], outcomes[idx_val, :]
        test_data, test_label = dataset[idx_test, :, :, :], outcomes[idx_test, :]

    # ndarray to tensor
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    # tensor to dataset
    train_dataset = utils.TensorDataset(train_data, train_label)
    dev_dataset = utils.TensorDataset(dev_data, dev_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    # dataset to dataloader 
    train_dataloader = utils.DataLoader(train_dataset)
    dev_dataloader = utils.DataLoader(dev_dataset)
    test_dataloader = utils.DataLoader(test_dataset)
    
    print("train_data.shape : {}\t train_label.shape : {}".format(train_data.shape, train_label.shape))
    print("dev_data.shape : {}\t dev_label.shape : {}".format(dev_data.shape, dev_label.shape))
    print("test_data.shape : {}\t test_label.shape : {}".format(test_data.shape, test_label.shape))
    
    return train_dataloader, dev_dataloader, test_dataloader


'''
in the paper : 49 layers, 33 input, 18838 parameters
input : 10-weights(*input), 6 - biases
Y: 1 weight(hidden*output), 1 bias(output)
Input : hidden : output : layer  = # of parameters : len(para)
1:1:1:1 = 18 : 18
2:1:1:1 = 25 : 18  // +7 as expected
1:1:1:2 = 34 : 18 // 34 = 16*2 + 2
33:33:1:1 = 562 : 18 // 16*33(528) + 33*1 +1 = 562
33:33:5:1 = 698 : 18 // 16*33(528) + 33*5(165) +5 = 698
33:33:5:49 = 26042 : 18 // 16*33*49(25872) + 33*5(165) +5 = 698
weights = 10*33*49(16170) + 33*5(165) = 16335 gap : 2503

'''


def train_gru_d(num_runs, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, n_epochs, batch_size, upsampling_batch, split_type):
    model_path = 'saved/grud_model_best.pt'

    acc_all = []
    auc_all = []
    aupr_all = []

    for r in range(num_runs):
        t_dataset = np.load('saved/dataset.npy')
        t_out = np.load('saved/y1_out.npy')
        if r == 0:
            print(t_dataset.shape, t_out.shape)

        train_dataloader, dev_dataloader, test_dataloader = data_dataloader(t_dataset, t_out, upsampling_batch, batch_size,
                                                                            split_type, train_proportion=0.8, dev_proportion=0.1)

        x_mean = torch.Tensor(np.load('saved/x_mean_aft_nor.npy'))
        print(x_mean.shape)
        # x_median = torch.Tensor(np.load('saved/x_median_aft_nor.npy'))

        model = GRUD(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout,
                     dropout_type='mloss', x_mean=x_mean, num_layers=num_layers)
        # print('number of parameters : ', count_parameters(model))
        # print(list(model.parameters())[0].requires_grad)


        epoch_losses = []

        # to check the update
        old_state_dict = {}
        for key in model.state_dict():
            old_state_dict[key] = model.state_dict()[key].clone()

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=1, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

        print('\n------------------\nRUN %d: Training started\n------------------' % r)
        best_aupr_val = 0
        for epoch in range(n_epochs):
            # train the model
            losses, acc = [], []
            label, pred = [], []
            y_pred_col = []
            model.train()
            for train_data, train_label in train_dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
                train_data = torch.squeeze(train_data)
                # train_label = torch.squeeze(train_label)
                train_label = torch.squeeze(train_label, dim=0)

                # Forward pass : Compute predicted y by passing train data to the model
                y_pred = model(train_data)

                #print(y_pred.shape)
                #print(train_label.shape)

                # Save predict and label
                y_pred_col.append(y_pred.item())
                pred.append(y_pred.item() > 0.5)
                label.append(train_label.item())

                #print('y_pred: {}\t label: {}'.format(y_pred, train_label))

                # Compute loss
                loss = criterion(y_pred, train_label)
                acc.append(
                    torch.eq(
                        # (torch.sigmoid(y_pred).data > 0.5).float(),   # sigmoid is already in 'forward' function
                        (y_pred.data > 0.5).float(),
                        train_label)
                )
                losses.append(loss.item())

                # perform a backward pass, and update the weights.
                loss.backward()
                optimizer.step()

            train_acc = torch.mean(torch.cat(acc).float())
            train_loss = np.mean(losses)

            train_pred_out = pred
            train_label_out = label

            # save new params
            new_state_dict = {}
            for key in model.state_dict():
                new_state_dict[key] = model.state_dict()[key].clone()

            # compare params
            for key in old_state_dict:
                if (old_state_dict[key] == new_state_dict[key]).all():
                    print('Not updated in {}'.format(key))

            # validation loss
            losses, acc = [], []
            label, pred = [], []
            model.eval()
            for dev_data, dev_label in dev_dataloader:
                # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
                dev_data = torch.squeeze(dev_data)
                # dev_label = torch.squeeze(dev_label)
                dev_label = torch.squeeze(dev_label, dim=0)

                # Forward pass : Compute predicted y by passing train data to the model
                y_pred = model(dev_data)

                # Save predict and label
                pred.append(y_pred.item())
                label.append(dev_label.item())

                # Compute loss
                loss = criterion(y_pred, dev_label)
                acc.append(
                    torch.eq(
                        # (torch.sigmoid(y_pred).data > 0.5).float(),   # sigmoid is already in 'forward' function
                        (y_pred.data > 0.5).float(),
                        dev_label)
                )
                losses.append(loss.item())

            auc_val = roc_auc_score(label, pred)
            aupr_val = average_precision_score(label, pred)
            scheduler.step(aupr_val)  # reduce learning rate when this metric has stopped improving

            if aupr_val > best_aupr_val:
                best_aupr_val = aupr_val
                torch.save(model, model_path)

            dev_acc = torch.mean(torch.cat(acc).float())
            dev_loss = np.mean(losses)

            dev_pred_out = pred
            dev_label_out = label

            print('Non-zero predictions = ', np.count_nonzero((np.array(pred) > 0.5).astype(int)))
            print("VALIDATION: Epoch %d, val_acc: %.2f, val_loss: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                  (epoch, dev_acc * 100, dev_loss.item(), aupr_val * 100, auc_val * 100))

            print(confusion_matrix(label, (np.array(pred) > 0.5).astype(int), labels=[0, 1]))

            # test loss
            losses, acc = [], []
            label, pred = [], []
            model.eval()
            for test_data, test_label in test_dataloader:
                # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
                test_data = torch.squeeze(test_data)
                # test_label = torch.squeeze(test_label)
                test_label = torch.squeeze(test_label, dim=0)

                # Forward pass : Compute predicted y by passing train data to the model
                y_pred = model(test_data)

                # Save predict and label
                pred.append(y_pred.item())
                label.append(test_label.item())

                # Compute loss
                loss = criterion(y_pred, test_label)
                acc.append(
                    torch.eq(
                        # (torch.sigmoid(y_pred).data > 0.5).float(),   # sigmoid is already in 'forward' function
                        (y_pred.data > 0.5).float(),
                        test_label)
                )
                losses.append(loss.item())

            test_acc = torch.mean(torch.cat(acc).float())
            test_loss = np.mean(losses)

            test_pred_out = pred
            test_label_out = label

            epoch_losses.append([
                 train_loss, dev_loss, test_loss,
                 train_acc, dev_acc, test_acc,
                 train_pred_out, dev_pred_out, test_pred_out,
                 train_label_out, dev_label_out, test_label_out,
             ])

            # pred = np.asarray(pred)
            # label = np.asarray(label)
            # auc_score = roc_auc_score(label, pred)
            # aupr_score = average_precision_score(label, pred)
            # print('Non-zero predictions = ', np.count_nonzero((pred > 0.5).astype(int)))
            # print("Epoch: {} Train loss: {:.4f}, Validation loss: {:.4f}, Test loss: {:.4f}, Test Acc: {:.4f}, Test AUROC: {:.4f}, Test AUPRC: {:.4f}".format(
            #         epoch, train_loss, dev_loss, test_loss, test_acc * 100, auc_score * 100, aupr_score * 100))

            # # save the parameters
            # train_log = []
            # train_log.append(model.state_dict())
            # torch.save(model.state_dict(), 'saved/grud_mean_grud_para.pt')
            # #print(train_log)

        print('\n------------------\nRUN %d: Training finished\n------------------' % r)

        # Test set
        model = torch.load(model_path)

        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            test_data = torch.squeeze(test_data)
            # test_label = torch.squeeze(test_label)
            test_label = torch.squeeze(test_label, dim=0)

            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)

            # Save predict and label
            pred.append(y_pred.item())
            label.append(test_label.item())

            # Compute loss
            loss = criterion(y_pred, test_label)
            acc.append(
                torch.eq(
                    # (torch.sigmoid(y_pred).data > 0.5).float(),   # sigmoid is already in 'forward' function
                    (y_pred.data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.item())

        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)

        pred = np.asarray(pred)
        label = np.asarray(label)

        auc_score = roc_auc_score(label, pred)
        aupr_score = average_precision_score(label, pred)

        print("\nTEST: test_acc: %.2f aupr_test: %.2f, auc_test: %.2f\n" %
              (test_acc * 100, aupr_score * 100, auc_score * 100))

        print(confusion_matrix(label, (np.array(pred) > 0.5).astype(int), labels=[0, 1]))

        acc_all.append(test_acc * 100)
        auc_all.append(auc_score * 100)
        aupr_all.append(aupr_score * 100)

    # print mean and std of all metrics
    acc_all, auc_all, aupr_all = np.array(acc_all), np.array(auc_all), np.array(aupr_all)
    mean_acc, std_acc = np.mean(acc_all), np.std(acc_all)
    mean_auc, std_auc = np.mean(auc_all), np.std(auc_all)
    mean_aupr, std_aupr = np.mean(aupr_all), np.std(aupr_all)
    print('------------------------------------------')
    print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
    print('AUROC    = %.1f +/- %.1f' % (mean_auc, std_auc))
    print('AUPRC    = %.1f +/- %.1f' % (mean_aupr, std_aupr))

    # show AUROC on test data for last trained epoch
    test_preds, test_labels = epoch_losses[-1][8], epoch_losses[-1][11]
    plot_roc_and_auc_score(test_preds, test_labels, 'GRU-D')


def plot_roc_and_auc_score(outputs, labels, title):
    false_positive_rate, true_positive_rate, threshold = roc_curve(labels, outputs)
    auc_score = roc_auc_score(labels, outputs)
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve, AREA = {:.4f}'.format(auc_score))
    plt.plot([0,1], [0,1], 'red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    num_runs = 5
    input_size = 33   # num of variables base on the paper
    hidden_size = 33  # same as inputsize
    output_size = 1
    num_layers = 49  # num of step or layers base on the paper / number of hidden states
    dropout = 0.0    # dropout_type : Moon, Gal, mloss
    learning_rate = 0.001
    n_epochs = 20
    batch_size = 128
    upsampling_batch = True
    split_type = 'gender'  # possible values: 'random', 'age', 'gender'

    train_gru_d(num_runs, input_size, hidden_size, output_size, num_layers, dropout, learning_rate, n_epochs,
                batch_size, upsampling_batch, split_type)

