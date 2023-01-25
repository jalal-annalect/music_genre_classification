import os
from utils import chunk_audio
import argparse
import tqdm
import json
from collections import Counter

# ** command line arguments

# accept command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-root_dir", "--root_dir", help = "Directory to read data from", required=True, type=str)
parser.add_argument("-out_dir", "--out_dir", help = "Directory to output split data", required=True, type=str)
parser.add_argument("-window", "--window", help = "Window for segmenting audio", type=int, default=3)
args = parser.parse_args()

# parcing cml arguments
root_dir = args.root_dir
out_dir = args.out_dir
window = args.window

# read from dir
train_root_dir = f"{root_dir}/train"
val_root_dir = f"{root_dir}/val"
test_root_dir = f"{root_dir}/test"


# create output dirs
if os.path.isdir(out_dir):
    pass
else:
    os.mkdir(f"{out_dir}")
    
# train and val split
os.mkdir(f"{out_dir}/train")
os.mkdir(f"{out_dir}/val")
os.mkdir(f"{out_dir}/test")

# output dirs
out_train_root_dir = f"{out_dir}/train"
out_val_root_dir = f"{out_dir}/val"
out_test_root_dir = f"{out_dir}/test"

# make sure to only take audio files
train_files = [elem for elem in os.listdir(train_root_dir) if elem.endswith(".wav")]
val_files = [elem for elem in os.listdir(val_root_dir) if elem.endswith(".wav")]
test_files = [elem for elem in os.listdir(test_root_dir) if elem.endswith(".wav")]

try:
    assert not bool(set(train_files).intersection(val_files).intersection(test_files))
except:
    raise Exception("found overlap in train val test files.")

# splitting training data
pbar = tqdm.tqdm(total=len(train_files))
for elem in train_files:
    audio_path = os.path.join(train_root_dir, elem)
    chunk_audio(filename=audio_path, output_dir=out_train_root_dir, window=window)
    pbar.update(1)
pbar.close()

# splitting valing data
pbar = tqdm.tqdm(total=len(val_files))
for elem in val_files:
    audio_path = os.path.join(val_root_dir, elem)
    chunk_audio(filename=audio_path, output_dir=out_val_root_dir, window=window)
    pbar.update(1)
pbar.close()

# splitting test data
pbar = tqdm.tqdm(total=len(test_files))
for elem in test_files:
    audio_path = os.path.join(test_root_dir, elem)
    chunk_audio(filename=audio_path, output_dir=out_test_root_dir, window=window)
    pbar.update(1)
pbar.close()

# getting data stats
data_info = {}
train_files = os.listdir(out_train_root_dir)
val_files = os.listdir(out_val_root_dir)
test_files = os.listdir(out_test_root_dir)

# labels
train_files_labels = dict(Counter([elem.split("_")[-1].split(".")[0] for elem in train_files]))
val_files_labels = dict(Counter([elem.split("_")[-1].split(".")[0] for elem in val_files]))
test_files_labels = dict(Counter([elem.split("_")[-1].split(".")[0] for elem in test_files]))

all_labels = list(set([elem.split("_")[-1].split(".")[0] for elem in train_files+val_files+test_files]))

for elem in all_labels:
    data_info[elem] = {"train":train_files_labels[elem], "val":val_files_labels[elem], "test":test_files_labels[elem]}

total_train = len(train_files)
total_val = len(val_files)
total_test = len(test_files)
total = total_train+total_val+total_test

data_info["total train"] = total_train
data_info["total val"] = total_val
data_info["total test"] = total_test
data_info["total"] = total
data_info["window"] = window

# save data info
with open(f"{out_dir}/data_info.json", mode="w") as outfile:
    json.dump(data_info, outfile)

