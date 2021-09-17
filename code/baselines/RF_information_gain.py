import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale


dataset = 'eICU'     # possible values: 'P12', 'P19', 'eICU'
print('Dataset used: ', dataset)

if dataset == 'P12':
    labels_path = '../../P12data/processed_data/arr_outcomes.npy'
    labels_names_path = '../../P12data/processed_data/ts_params.npy'
    features_path = '../../P12data/processed_data/PTdict_list.npy'
elif dataset == 'P19':
    labels_path = '../../P19data/processed_data/arr_outcomes_6.npy'
    labels_names_path = '../../P19data/processed_data/labels_ts.npy'
    features_path = '../../P19data/processed_data/PT_dict_list_6.npy'
elif dataset == 'eICU':
    labels_path = '../../eICUdata/processed_data/arr_outcomes.npy'
    labels_names_path = '../../eICUdata/processed_data/eICU_ts_vars.npy'
    features_path = '../../eICUdata/processed_data/PTdict_list.npy'

labels = np.load(labels_path, allow_pickle=True)
if dataset == 'P12' or dataset == 'P19':
    labels_np = labels[:, -1].reshape([-1, 1])  # shape  (11988, 1)
elif dataset == 'eICU':
    labels_np = labels[..., np.newaxis]

features = np.load(features_path, allow_pickle=True)

T, F = features[0]['arr'].shape
feature_np = np.zeros((len(features), T, F))
for i in range(len(features)):
    feature_np[i] = features[i]['arr']
print(feature_np.shape, labels_np.shape)

n_sensors = feature_np.shape[-1]
print('n-sensors: {}'.format(n_sensors))
AUC_list = []
# select the first feature, 11988 samples, 215 row, each row has 36 features
for f in range(n_sensors):
    feature_ji = f

    feature_1 = feature_np[:, :, feature_ji]
    data_fea_label = np.hstack((feature_1, labels_np))

    n_seg = data_fea_label.shape[0]
    np.random.shuffle(data_fea_label)

    train_data = data_fea_label[: int(0.8*n_seg)]
    test_data = data_fea_label[int(0.8*n_seg):]

    no_fea_long = train_data.shape[-1] - 1  # here is - 2, because has two IDs

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    # print(train_data.shape, test_data.shape)

    feature_train = train_data[:, :no_fea_long]
    feature_test = test_data[:, :no_fea_long]
    """Normalization"""
    feature_train = scale(feature_train, axis=0)
    feature_test = scale(feature_test, axis=0)

    label_train = train_data[:, no_fea_long:no_fea_long + 1].squeeze(-1)
    label_test = test_data[:, no_fea_long:no_fea_long + 1].squeeze(-1)
    # random forest
    rf = RandomForestClassifier(n_estimators=10).fit(feature_train, label_train)

    rf_acc = rf.score(feature_test, label_test)
    rf_acc_train = rf.score(feature_train, label_train)

    rf_result = rf.predict(feature_test)

    auc_score = roc_auc_score(label_test, rf_result)
    print('Feature:', feature_ji, '| ACC: %.4f' % rf_acc, '| AUC: %.4f' % auc_score)
    AUC_list.append(auc_score)

sorted_sensor = np.argsort(np.array(AUC_list))  # sensor importance ascendingly ordered
sensor_descending = sorted_sensor[::-1]
print(sensor_descending)

labels_names = np.load(labels_names_path, allow_pickle=True)

indices_with_names = np.array([[ind, labels_names[ind]] for ind in sensor_descending])

# np.save('saved/IG_density_scores_' + dataset, indices_with_names, allow_pickle=True)
