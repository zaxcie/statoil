import json
import random

DATA_PATH = "../../data/raw/"
DATA_OUT_PATH = "../../data/processed/"

with open(DATA_PATH + 'train.json') as f:
    train = json.load(f)

random.seed(966)

sample_id = random.sample(range(len(train)), int(round(len(train) * 0.8, 0)))

train_out = []
val_out = []

for i in range(len(train)):
    if i in sample_id:
        train_out.append(train[i])
    else:
        val_out.append(train[i])

with open(DATA_OUT_PATH + "val.json", 'w') as f:
    json.dump(val_out, f)

with open(DATA_OUT_PATH + "train.json", 'w') as f:
    json.dump(train_out, f)
