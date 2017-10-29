from src.features import extract_stats, prepare_data
import pandas as pd

DATA_PATH = "../../data/raw/"

train = pd.read_json(DATA_PATH + 'train.json')

df, bands = prepare_data.pd_to_np(train, shape="df")

train_X = extract_stats.process(df=df, bands=bands)
train_y = train['is_iceberg'].values

print(train_X.shape)