import utils
import tensorflow as tf
import os
import cv2
import numpy as np
import random
import argparse
import uuid
import json
import datetime

# todays date
today = datetime.date.today()

# for reproducibility
np.random.seed(42)

# ** command line arguments

# accept command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-root_dir", "--root_dir", help = "Directory where images", type=str, required=True)
parser.add_argument("-epochs", "--epochs", help = "Number of epochs", type=int, default=25)
parser.add_argument("-batch_size", "--batch", help = "Batch size", type=int, default=32)
parser.add_argument("-learning_rate", "--learning_rate", help = "Learning rate", default=0.01, type=float)
parser.add_argument("-optimizer", "--optimizer", help = "Optimizer", default="sgd", type=str)
parser.add_argument("-steps_per_epoch", "--steps_per_epoch", help = "Steps per epoch", default=50, type=int)
parser.add_argument("-validation_steps", "--validation_steps", help = "Validation per epoch", default=25, type=int)
parser.add_argument("-notes", "--notes", help="Model training notes", default=None,type=str)
args = parser.parse_args()

# making output dirs
id = uuid.uuid1().hex
os.mkdir(f"results/experiment_id_{id}")
os.mkdir(f"results/experiment_id_{id}/model")
os.mkdir(f"results/experiment_id_{id}/eval")
os.mkdir(f"results/experiment_id_{id}/weights")

# parse arguments
root_dir = args.root_dir
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
optimizer = args.optimizer

# get optimizer
optimizer = utils.get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
steps_per_epoch = args.steps_per_epoch
validation_steps = args.validation_steps
notes = str(args.notes)

# get train and val files
train_files = os.listdir(f"{root_dir}/train")
val_files = os.listdir(f"{root_dir}/val")


# shuffeling data
for _ in range(100):
    random.shuffle(train_files)
    random.shuffle(val_files)

# encoding data
labels = "blues classical country disco hiphop metal pop reggae rock".split()
label_map = {elem:i for i, elem in enumerate(labels)}

# build train dataset
X_train, y_train = [], []

for i, train_file in enumerate(train_files):
    image = cv2.imread(f"{root_dir}/train/{train_file}")
    label = train_file.split("_")[-1].split(".")[0]

    # remove jazz
    if label=="jazz":
        continue
    else:
        X_train.append(image)
        y_train.append(label_map[label])

# to np array
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1, 1)

# build val dataset
X_val, y_val = [], []
for i, val_file in enumerate(val_files):
    image = cv2.imread(f"{root_dir}/val/{val_file}")
    label = val_file.split("_")[-1].split(".")[0]

    # remove jazz
    if label=="jazz":
        continue
    else:
        X_val.append(image)
        y_val.append(label_map[label])

# to np array
X_val, y_val = np.array(X_val), np.array(y_val).reshape(-1, 1)

# train data
train_data = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .batch(batch_size=batch_size)
            .shuffle(batch_size*8)
            .prefetch(tf.data.AUTOTUNE))

# val data
val_data = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(batch_size=batch_size)
            .shuffle(batch_size*8)
            .prefetch(tf.data.AUTOTUNE))

# create model
model = utils.build_model()

# save model network summary
utils.save_model_summary(model=model, filename=f"results/experiment_id_{id}/network.txt")

# compiling model
# metrics to monitor
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=metrics)

print('Setting up callbacks ...')
# setting up callbacks
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(f"results/experiment_id_{id}/weights", "weights" + "_epoch_{epoch}"),
    monitor="loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    ), 
    tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=15,
    verbose=1,
    mode="auto")]

# train start time
start = datetime.datetime.now()
# training
history = model.fit(
    train_data.repeat(),
    validation_data=val_data.repeat(),
    epochs=epochs,
    callbacks=callbacks_list,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
    )
# train end time 
end = datetime.datetime.now()

print("Saving model...")
# save model
model.save(filepath=f"results/experiment_id_{id}/model")

print('Saving model histroy...')

# saving loss results
with open(f"results/experiment_id_{id}/eval/loss.json", 'w', encoding='utf-8') as outfile:
    json.dump(history.history, outfile)

print('Saving model info...')
# saving used hyperparameters
model_info = {}
model_info["start training date"] = str(today)
model_info['epochs'] = epochs
model_info['batch_size'] = batch_size
model_info['train_time'] = (end-start).seconds # in seconds
model_info['train_time_per_epoch'] = round(model_info['train_time']/model_info['epochs'], 3)
model_info['train_time_per_step'] = round(model_info['train_time_per_epoch']/steps_per_epoch, 3)
model_info['learning_rate'] = learning_rate
model_info['optimizer'] = args.optimizer.lower()
model_info['notes'] = str(notes).replace("_", " ")

# saving loss results
with open(f"experiment_id_{id}/model_info.json", 'w', encoding='utf-8') as outfile:
    json.dump(model_info, outfile)
print('Model training is complete.')