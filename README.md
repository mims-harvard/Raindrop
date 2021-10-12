<!-- # GRAPH-GUIDED NETWORK FOR IRREGULARLY SAMPLED MULTIVARIATE TIME SERIES -->
# Graph-Guided Network For Irregularly Sampled Multivariate Time Series

## Overview

This repository contains processed datasets and implementation code for manuscript *Graph-Guided Network For Irregularly Sampled Multivariate Time Series*.
We propose, Raindrop, a graph-guided network, to learn representations of irregularly sampled multivariate time series. 
We use Raindrop to classify time series of three healthcare and human activity datasets in four different settings. 


## Key idea of Raindrop

The proposed Raindrop models dependencies between sensors using neural message passing and temporal self attention.
Raindrop represents every sample (e.g., patient) as a graph, where nodes indicate sensors and edges represent dependencies between them. 
Raindrop takes samples as input, each sample containing multiple sensors and each sensor consisting of irregularly recorded observations (e.g., in clinical data, an individual patient’s state of health, recorded at irregular time intervals with different subsets of sensors observed at different times). Raindrop model is inspired by the idea of raindrops falling into a pool at sequential but nonuniform time intervals and thereby creating ripple effects that propagate across the pool.

![Raindrop idea](images/fig1.png "Idea of Raindrop.")

The main idea of Raindrop is to generate observation (a) and sensor (b) embeddings. Calculated sensor
embeddings then serve as the basis for sample embeddings that can fed into a downstream task 
such as classification. 

![Raindrop observations](images/fig3.png "Raindrop observations.")

**(a)** Raindrop generates observation embedding based on observed value, passes
message to neighbor sensors, and generates observation embedding through inter-sensor dependencies. 

**(b)** An illustration of generating sensor embedding. We apply the message
passing in (a) to all timestamps and produce corresponding observation embeddings. 
We aggregate arbitrary number of observation embeddings into a fixed-length sensor embedding,
while paying distinctive attentions to different observations. 
We independently apply the processing procedure to all sensors.

<br />
We evaluate our model in comparison with the baselines in four different settings:

**Setting 1: Classic time series classification.** We randomly split the dataset into training
(80%), validation (10%), and test (10%) set. The indices of these splits are fixed across all methods.

**Setting 2: Leave-fixed-sensors-out.** In this setting, we select a proportion of
sensors, and set all their observations as zero in validation and testing set 
(training samples are not changed). We mask out the most informative sensors and 
the selected sensors are fixed across samples and models.

**Setting 3: Leave-random-sensors-out.** Setting 3 is similar to Setting 2 except that the
missing sensors in this setting are randomly selected instead of fixed. In each test sample, 
we select a subset of sensors and regard them as missing through replacing all of their 
observations with zeros.

**Setting 4: Group-wise time series classification.** In this setting we split the data 
into two groups, based on a specific static attribute. The first split attribute is *age*, 
where we classify people into young (< 65 years) and old (>= 65 years) groups. 
We also split patients into male and female by *gender* attribute. Given the split attribute, 
we use one group as a train set and randomly split the other group into equally sized
validation and test set.

## Datasets

We prepared to run our code for Raindrop as well as the baseline methods with two healthcare and 
one human activity dataset.

**(1)** P19 (PhysioNet Sepsis Early Prediction Challenge 2019) includes 38,803 patients that are monitored by 34 sensors. Each patient is associated with a
binary label representing the occurrence of sepsis. 

**(2)** P12 (PhysioNet Mortality Prediction Challenge 2012) records temporal measurements 
of 36 sensors of 11,988 patients in the first 48-hour stay in ICU. 
The samples are labeled based on hospitalization length. 

**(3)** PAM (PAMAP2 Physical Activity Monitoring) contains 5,333 segments from 8 activities 
of daily living that are measured by 17 sensors.

The preprocessing scripts with data are available in folders *P19data*, 
*P12data* and *PAMdata*. 

Let's look at the content of *P12data* folder, though, 
the structure is the same for all three folders with data. 
Some datasets may exclude raw data and do not have preprocessing scripts.
Inside the *\*data* folder, we have the following structure:

- *process_scripts*
    - Inside we have preprocessing scripts and *readme* with the instructions how to run them.
- *processed_data*
    - *P\_list.npy*: Array of dictionaries, which is created from raw data. Array has a length of 
    number of samples and each dictionary has keys 'id', 'static' variables and 'ts' time series data.
    - *PTdict\_list.npy*: Processed array of dictionaries. Array has a length of number of samples and 
    each dictionary includes keys, such as 'id', 'static' attributes, 'arr' time series data 
    and 'time' of observations.    
    - *arr\_outcomes.npy*: The content has the shape (number of samples, outcomes). 
    For each sample (patient) there are target outputs, such as length of hospital stay or mortality.
    - *ts\_params.npy*: Array with names of all sensors.
    - *static\_params.npy*: Array with names of static attributes.
    - *extended\_static\_params.npy*: Array with names of extended static attributes 
    (with more attributes than in *static\_params.npy*).
    - *readme.md*: Short description of the files.
- *rawdata*
    - *set-a*: Data in the form of 4,000 .txt files, each containing time series observations. 
    - *set-b*: Data in the form of 4,000 .txt files, each containing time series observations.
    - *set-c*: Data in the form of 4,000 .txt files, each containing time series observations.
    - *Outcomes-a*: Text file, including target values (e.g., length of hospital stay, mortality) for all 4,000 samples from *set-a*.
    - *Outcomes-b*: Text file, including target values (e.g., length of hospital stay, mortality) for all 4,000 samples from *set-b*.
    - *Outcomes-c*: Text file, including target values (e.g., length of hospital stay, mortality) for all 4,000 samples from *set-c*.
- *splits*
    - Includes 5 different splits of data indices (train, validation, test) to use them 
    when running an algorithm five times to measure mean and standard deviation of the performance.


## Requirements

Raindrop was tested using Python 3.6 and 3.9.

To have consistent libraries and their versions, you can install needed dependencies 
for this project running the following command:

```
pip install -r requirements.txt
```


## Running the code

We provide ready-to-run code for our Raindrop model and the following baselines: 
Transformer, Trans-mean, GRU-D, SeFT and mTAND. 
Starting from root directory *Raindrop*, you can run models as follows:

- Raindrop
```
cd code
python Raindrop.py
```

- Transformer
```
cd code/baselines
python Transformer_baseline.py
```

- Trans-mean
```
cd code/baselines
python Transformer_baseline.py --imputation mean
```

- GRU-D
```
cd code/baselines
python GRU-D_baseline.py
```

- SeFT
```
cd code/baselines
python SEFT_baseline.py
```

- mTAND
```
cd code/baselines/mTAND
python mTAND_baseline.py
```

All algorithms can be run with named arguments, which allow the use of different settings from the paper:
- *dataset*: Choose which dataset to use. Options: [P12, P19, PAM].
- *withmissingratio*: If True, missing ratio of sensors in test set ranges from 0.1 to 0.5. 
If False, missing ratio is 0. Used in setting 2 and 3. Options: [True, False].
- *splittype*: Choose how the data is split into train, validation and test set.
Used in setting 4. Options: [random, age, gender]. 
- *reverse*: Choose the order in setting 4. If True, use female/old for training. 
If False, use male/young for training. Options: [True, False].
- *feature_removal_level*: Choose between setting 1 (no_removal), 2 (set) and 3 (sample). 
Options: [no_removal, set, sample]. 
- *predictive_label*: Choose which label is predicted. Only for P12 dataset. Options: [mortality, LoS].
- *imputation*: Imputation method to choose to fill in missing values. Only used in Transformer.
Options: [no_imputation, mean, forward, cubic_spline].


#### Examples

In all cases beware the directory from which you run these commands (see *cd* commands above).

Run Raindrop model on P12 dataset in setting 1 (standard time series classification) 
for predicting length of hospital stay, which is binarized with the threshold of 3 days:

```
python Raindrop.py --dataset P12 --withmissingratio False --splittype random --feature_removal_level no_removal --predictive_label LoS
```

Run Transformer baseline on P19 dataset in setting 2 (leave-fixed-sensors-out):

```
python Transformer_baseline.py --dataset P19 --withmissingratio True --splittype random --feature_removal_level set
```

Run SeFT baseline on PAM dataset in setting 3 (leave-random-sensors-out):

```
python SEFT_baseline.py --dataset PAM --withmissingratio True --splittype random --feature_removal_level sample
```

Run GRU-D baseline on P12 dataset in setting 4, where you train on younger than 65 and 
test on aged 65 or more.

```
python GRU-D_baseline.py --dataset P12 --withmissingratio False --splittype age --feature_removal_level no_removal --reverse False
```


## License

Raindrop is licensed under the MIT License.
