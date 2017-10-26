import numpy as np
import pandas as pd

from src.models.model_def import get_callbacks, get_model
from src.features.prepare_data import prepare_data

#TODO port loading + processing data to function

np.random.seed(966)

DATA_PATH = "../../data/raw/"
MODEL_PATH = "../reports/" #Still not used for now
FILE_PATH = ".model_weights.hdf5"


train = pd.read_json(DATA_PATH + 'train.json')
test = pd.read_json(DATA_PATH + 'test.json')

train.inc_angle = train.inc_angle.replace('na', 0) #TODO Explore impute value
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

print("Data loaded")


model = get_model()
model.summary()

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = prepare_data(train, test)

callbacks = get_callbacks(filepath=FILE_PATH, patience=5)

model.fit([X_train, X_angle_train], y_train, epochs=1000,
          validation_data=([X_valid, X_angle_valid], y_valid),
          batch_size=32,
          callbacks=callbacks)

model.load_weights(filepath=FILE_PATH)

print(model.evaluate([X_train, X_angle_train], y_train, verbose=1, batch_size=32))
print(model.evaluate([X_valid, X_angle_valid], y_valid, verbose=1, batch_size=32))

prediction = model.predict([X_test, X_angle_test], verbose=1, batch_size=32)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})

submission.head(5)

submission.to_csv("submission.csv", index=False)
