import numpy as np
from sklearn.model_selection import train_test_split


def pd_to_np(df, shape="dl"):
    '''Function specific to project, has hard coded value inside. No plan to change.
    Is used to prepare data from JSON to something model can consume.

    dl: Prepare to output has image. Use average of band1 and band2 to create band3
    df: Prepare to output for extraction stats from images. Remove bands from df and return bands as np array
    '''
    if shape == "dl":
        band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
        band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
        X = np.concatenate([band1[:, :, :, np.newaxis],
                            band2[:, :, :, np.newaxis],
                            ((band1 + band2) / 2)[:, :, :, np.newaxis]], axis=-1)
        X_angle = np.array(df.inc_angle)

        return X, X_angle

    if shape == "df":
        band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
        band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])

        df = df.drop(['band_1', 'band_2'], axis=1)

        bands = np.stack((band1, band2, 0.8*(band1 + band2)), axis=-1)
        del band1, band2

        return df, bands

def prepare_data(train, val, test):
    train.inc_angle = train.inc_angle.replace('na', 0) #TODO Explore impute value
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

    val.inc_angle = val.inc_angle.replace('na', 0)  # TODO Explore impute value
    val.inc_angle = val.inc_angle.astype(float).fillna(0.0)

    print("Data loaded")

    X_train, X_angle_train = pd_to_np(train)
    y_train = np.array(train["is_iceberg"])

    X_valid, X_angle_valid = pd_to_np(val)
    y_valid = np.array(val["is_iceberg"])

    X_test, X_angle_test = pd_to_np(test)

    return X_train, X_valid, X_test, X_angle_train, X_angle_valid, X_angle_test, y_train, y_valid
