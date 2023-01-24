import os
import argparse
import random
import shutil
import integv
import tqdm
import json
import warnings
warnings.filterwarnings("ignore")


# ** command line arguments

# accept command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-root_dir", "--root_dir", help = "Directory to read data from", required=True, type=str)
parser.add_argument("-out_dir", "--out_dir", help = "Directory to output split data", required=True, type=str)
parser.add_argument("-split", "--split", help = "Training split", type=float, default=0.8)
args = parser.parse_args()

# parcing cml arguments
root_dir = args.root_dir
out_dir = args.out_dir
split = args.split

print("Checking for corrupted files...")
# ** remove corrupted files
for elem in os.listdir(root_dir):
    dir = f"{root_dir}/{elem}"
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            corrupt = not integv.verify(open(f"{dir}/{file}", "rb"), file_type="wav")
            if not corrupt:
                pass
            else:
                print(f"Found corrupt file, will remove {dir}/{file}.")
                os.remove(f"{dir}/{file}")

# create output dirs
if os.path.isdir(out_dir):
    pass
else:
    os.mkdir(f"{out_dir}")
    
# train and val split
os.mkdir(f"{out_dir}/train")
os.mkdir(f"{out_dir}/val")

print("Splitting data...")

# ** creating train and val data
data_info = {}
for elem in os.listdir(root_dir):

    dir = f"{root_dir}/{elem}"
    if os.path.isdir(dir):
        files = os.listdir(dir)
        print(f"found: {len(files)} files in category {elem}")

        # shuffle files
        random.shuffle(files)
        random.shuffle(files)

        # splitting
        n = round(len(files)*split)
        train_files = files[:n]
        val_files = files[n:]

        # add train and val numbers per category
        data_info[elem] = {"train":len(train_files), "val":len(val_files)}

        # copying train files
        for train_file in train_files:
            if os.path.isfile(f"{dir}/{train_file}"):
                shutil.copy(f"{dir}/{train_file}", f"{out_dir}/train/{train_file}")
            else:
                raise Exception(f"""{dir}/{train_file} not found""")
                
        # copying val files
        for val_file in val_files:
            if os.path.isfile(f"{dir}/{val_file}"):
                shutil.copy(f"{dir}/{val_file}", f"{out_dir}/val/{val_file}")
            else:
                raise Exception(f"""{dir}/{val_file} not found""")
    else:
        pass
    
# adding total data set stats
total_train = sum([data_info[key]["train"] for key in data_info])
total_val = sum([data_info[key]["val"] for key in data_info])
total = total_train+total_val
data_info["total train"] = total_train
data_info["total val"] = total_val
data_info["total"] = total
data_info["train split"] = split

print("Saving data stats...")
# save data info
with open(f"{out_dir}/data_info.json", mode="w") as outfile:
    json.dump(data_info, outfile)
print("Done!")