# It is an experiment script of Section 2.4.1 to extrapolate ZTs of the materials from unknown material groups.

import numpy
import pandas
import joblib
from itertools import chain
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from simd.data import load_dataset
from simd.mat_system import MatSysDict


# Experiment setting.
n_folds = 3
random_seed = 0


# Load target dataset.
dataset = load_dataset(path_dataset='dataset/example_dataset.xlsx',
                       idx_form=0,
                       idx_num_feats=1,
                       idx_target=6)

# Load source dataset for transfer learning.
dataset_src = load_dataset(path_dataset='dataset/example_src_dataset.xlsx',
                           idx_form=0,
                           idx_num_feats=1,
                           idx_target=2)
k_folds = dataset.get_k_fold_chem(k=n_folds, random_seed=random_seed)

results_total = list()
mae = list()
r2 = list()

# Perform k-fold cross validation.
for i in range(0, n_folds):
    forms = list()
    temps = list()
    targets = list()
    preds = list()

    # Load training and test datasets.
    dataset_train = k_folds[i][0]
    dataset_test = k_folds[i][1]

    # Data augmentation of the source dataset.
    dataset_train.merge(dataset_src)

    # Generate the system-identified material descriptor.
    sys_dict = MatSysDict(dataset_train)
    dataset_train.set_sys_info(sys_dict)
    dataset_test.set_sys_info(sys_dict)

    # Train a prediction model.
    model = XGBRegressor(max_depth=7, n_estimators=700, objective='reg:squaredlogerror')
    model.fit(dataset_train.x(), dataset_train.y())

    # Predict the target properties of the test materials.
    for d in dataset_test.data:
        forms.append(d.form)
        temps.append(d.temp)
    targets = dataset_test.y().reshape(-1, 1)
    preds = model.predict(dataset_test.x()).reshape(-1, 1)

    # Calculate evaluation metrics.
    mae.append(mean_absolute_error(targets, preds))
    r2.append(r2_score(targets, preds))

    # Save the training and evaluation results.
    results = list()
    for j in range(0, len(forms)):
        results.append([forms[j], temps[j], targets[j, 0], preds[j, 0], numpy.abs(targets[j, 0] - preds[j, 0])])
    pandas.DataFrame(results).to_excel('results/results_extrapol_' + str(i) + '.xlsx', index=None, header=None)
    joblib.dump(model, 'results/model_extrapol_' + str(i) + '.joblib')
    joblib.dump(sys_dict, 'results/sdict_extrapol_' + str(i) + '.joblib')
    results_total.append(results)

    print(mean_absolute_error(targets, preds), r2_score(targets, preds))

# Print the mean and the standard deviation of the trained model in k-fold cross validation.
print('-------------------------------------------------')
print('Evaluation results of the k-fold cross validation')
print(numpy.mean(mae), numpy.std(mae))
print(numpy.mean(r2), numpy.std(r2))

# Save the prediction results of the train models for the test datasets.
results_total = list(chain.from_iterable(results_total))
pandas.DataFrame(results_total).to_excel('results/preds_sxgb.xlsx', index=None, header=None)
