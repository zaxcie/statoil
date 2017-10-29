import numpy as np
import pandas as pd

from datetime import datetime
import json

from src.models.model_def import get_callbacks, get_model
from src.features.prepare_data import prepare_data

from statistics import mean, stdev

from sklearn.model_selection import KFold

#TODO port loading + processing data to function
# Create file for model info and validation metrics

def train_model(X_train, X_valid, X_test,
                X_angle_train, X_angle_valid, X_angle_test,
                y_train, y_valid,
                train_id, val_id, test_id,
                nb_fold=10):

    model_info = dict()
    bs = 32

    folds = KFold(n_splits=nb_fold)
    model_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    model_path = str(MODEL_PATH + model_name + ".model_weights.hdf5")

    model_info['model_name'] = model_name
    model_info['nb_folds'] = nb_fold
    model_info['model_weight_path'] = model_path
    model_info['metrics_per_fold'] = []
    model_info['metrics_cv'] = {}
    model_info['metrics_val'] = {}

    prob_train_ho = None

    callbacks = get_callbacks(filepath=model_path, patience=3)

    for trainset, valset in folds.split(train):
        print(len(X_train[trainset]))
        print(len(X_train[valset]))

        model.fit([X_train[trainset], X_angle_train[trainset]], y_train[trainset], epochs=1000,
                  validation_data=([X_train[valset], X_angle_train[valset]], y_train[valset]),
                  batch_size=bs,
                  callbacks=callbacks)

        preds = model.predict([X_train[trainset], X_angle_train[trainset]], verbose=1, batch_size=bs)

        eval_model = model.evaluate([X_train[valset], X_angle_train[valset]], y_train[valset],
                                    verbose=1, batch_size=bs)
        eval_model_dict = dict()

        eval_model_dict['loss'] = eval_model[0]
        eval_model_dict['accuracy'] = eval_model[1]

        model_info['metrics_per_fold'].append(eval_model_dict)

        if prob_train_ho is None:
            prob_train_ho = pd.DataFrame({'id': train_id[trainset],
                                          'is_iceberg': preds.reshape((preds.shape[0]))})

        else:
            temp_df = pd.DataFrame({'id': train_id[trainset],
                                    'is_iceberg': preds.reshape((preds.shape[0]))})

            prob_train_ho.append(temp_df)

    eval_model_dict = dict()


    loss = []
    accuracy = []

    for fold_result in model_info['metrics_per_fold']:
        loss.append(fold_result['accuracy'])
        accuracy.append(fold_result['loss'])

    eval_model_dict['loss_mean'] = mean(loss)
    eval_model_dict['loss_std'] = stdev(loss)

    eval_model_dict['accuracy_mean'] = mean(accuracy)
    eval_model_dict['accuracy_std'] = stdev(accuracy)

    model_info['metrics_cv'] = eval_model_dict

    model.fit([X_train, X_angle_train], y_train, epochs=1000,
              validation_data=([X_valid, X_angle_valid], y_valid),
              batch_size=bs,
              callbacks=callbacks)

    model_info['optimizer'] = model.get_config()

    eval_model = model.evaluate([X_valid, X_angle_valid], y_valid,
                                verbose=1, batch_size=bs)

    eval_model_dict['loss'] = eval_model[0]
    eval_model_dict['accuracy'] = eval_model[1]

    model_info['metrics_val'] = eval_model_dict

    preds_val = model.predict([X_valid, X_angle_valid], verbose=1, batch_size=bs)
    preds_test = model.predict([X_test, X_angle_test], verbose=1, batch_size=bs)

    preds_val_df = pd.DataFrame({'id': val_id,
                                 'is_iceberg': preds_val.reshape((preds_val.shape[0]))})
    preds_test_df = pd.DataFrame({'id': test_id,
                                  'is_iceberg': preds_test.reshape((preds_test.shape[0]))})

    preds_val_df.to_csv(MODEL_PATH + model_name + "_val_prob.csv", index=False)
    preds_test_df.to_csv(MODEL_PATH + model_name + "_test_prob.csv", index=False)

    with open(MODEL_PATH + model_name + "-config.json", 'w') as f:
        json.dump(model_info, f)


np.random.seed(966)

DATA_PATH = "../../data/processed/"
MODEL_PATH = "../../reports/models/"

train = pd.read_json(DATA_PATH + 'train.json')
val = pd.read_json(DATA_PATH + 'val.json')
test = pd.read_json(DATA_PATH + 'test.json')

print("Data loaded")

model = get_model()
model.summary()

X_train, X_valid, X_test, X_angle_train, X_angle_valid, X_angle_test, y_train, y_valid = prepare_data(train, val, test)

train_model(X_train, X_valid, X_test,
            X_angle_train, X_angle_valid, X_angle_test,
            y_train, y_valid,
            train['id'], val['id'], test['id'], nb_fold=10)


