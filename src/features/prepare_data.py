import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(train, test):
    train.inc_angle = train.inc_angle.replace('na', 0) #TODO Explore impute value
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

    print("Data loaded")

    # Train data
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , (x_band1+x_band1/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_train = np.array(train.inc_angle)
    y_train = np.array(train["is_iceberg"])

    # Test data
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , (x_band1+x_band1/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_test = np.array(test.inc_angle)

    X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train, X_angle_train, y_train,
                                                                                        random_state=123, train_size=0.75)

    return X_train, X_valid, X_test, X_angle_train, X_angle_valid, X_angle_test, y_train, y_valid