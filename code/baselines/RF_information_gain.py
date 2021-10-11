import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale


dataset = 'PAM'     # possible values: 'P12', 'P19', 'eICU', 'PAM'
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
elif dataset == 'PAM':
    labels_path = '../../PAMdata/processed_data/arr_outcomes.npy'
    labels_names_path = ''   # not applicable
    features_path = '../../PAMdata/processed_data/PTdict_list.npy'

labels = np.load(labels_path, allow_pickle=True)
if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM':
    labels_np = labels[:, -1].reshape([-1, 1])
elif dataset == 'eICU':
    labels_np = labels[..., np.newaxis]

features = np.load(features_path, allow_pickle=True)

if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
    T, F = features[0]['arr'].shape
    feature_np = np.zeros((len(features), T, F))
    for i in range(len(features)):
        feature_np[i] = features[i]['arr']
elif dataset == 'PAM':
    feature_np = features

n_sensors = feature_np.shape[-1]
AUC_list = []
for f in range(n_sensors):
    feature_ji = f

    feature_1 = feature_np[:, :, feature_ji]
    data_fea_label = np.hstack((feature_1, labels_np))

    n_seg = data_fea_label.shape[0]
    np.random.shuffle(data_fea_label)

    train_data = data_fea_label[: int(0.8*n_seg)]
    test_data = data_fea_label[int(0.8*n_seg):]

    no_fea_long = train_data.shape[-1] - 1

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    feature_train = train_data[:, :no_fea_long]
    feature_test = test_data[:, :no_fea_long]
    feature_train = scale(feature_train, axis=0)
    feature_test = scale(feature_test, axis=0)

    label_train = train_data[:, no_fea_long:no_fea_long + 1].squeeze(-1)
    label_test = test_data[:, no_fea_long:no_fea_long + 1].squeeze(-1)
    rf = RandomForestClassifier(n_estimators=10).fit(feature_train, label_train)

    rf_acc = rf.score(feature_test, label_test)
    rf_acc_train = rf.score(feature_train, label_train)

    rf_result = rf.predict(feature_test)

    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
        auc_score = roc_auc_score(label_test, rf_result)
    elif dataset == 'PAM':
        # actually accuracy of the predictions, since it's multiclass and we don't have probabilities for AUC
        auc_score = rf_acc

    print('Feature:', feature_ji, '| ACC: %.4f' % rf_acc, '| AUC: %.4f' % auc_score)
    AUC_list.append(auc_score)

sorted_sensor = np.argsort(np.array(AUC_list))
sensor_descending = sorted_sensor[::-1]
print(sensor_descending)

if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
    labels_names = np.load(labels_names_path, allow_pickle=True)
elif dataset == 'PAM':
    labels_names = ['sensor_%s' % str(i) for i in range(17)]

indices_with_names = np.array([[ind, labels_names[ind]] for ind in sensor_descending])

# np.save('saved/IG_density_scores_' + dataset, indices_with_names, allow_pickle=True)
