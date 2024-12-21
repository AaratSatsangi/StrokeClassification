import numpy as np
import copy
import os
import shutil

PATH_DATASET = "./Data/Compiled/PNG/"
PATH_DATASET_SPLIT = "./Data/Compiled/Split/"
GENERATOR = np.random.default_rng(seed=26)
SPLIT = [0.8, 0.2]   
    
def get_nums():
    classes = os.listdir(PATH_DATASET)
    total_num = {}
    total = 0
    for _class in classes:
        total_num[_class] = len(os.listdir(PATH_DATASET + _class))
        total += total_num[_class]

    percent_num = copy.deepcopy(total_num)
    for key in percent_num.keys():
        percent_num[key] = np.round(percent_num[key] / total, 3)
    
    return total_num, percent_num

def train_test_split(pick_percent, total_class_num):
    indices = np.arange(total_class_num)
    GENERATOR.shuffle(indices)
    end = int(np.round(total_class_num*pick_percent))
    train_indices = indices[:end]
    test_indices = indices[end:]

    return train_indices, test_indices

def save_set(_class, indices, dst_folder):
    src = PATH_DATASET + _class
    dst = PATH_DATASET_SPLIT + dst_folder + _class
    
    if(not os.path.exists(dst)):
        os.makedirs(dst)
    
    step = 0
    files = sorted(os.listdir(src))
    for idx in indices:
        src_file = os.path.join(src, files[idx])
        dest_file = os.path.join(dst, files[idx])
        shutil.copy(src_file, dest_file)
        if step == 0: 
            print("Copying: |", end="\r") 
            step +=1
        elif step == 1: 
            print("Copying: /", end="\r")
            step +=1
        else:
            print("Copying: \\", end="\r")
            step = 0


if __name__ == "__main__":
    
    total_num, percent_num = get_nums()
    print(total_num)
    print(percent_num)
    print("\n" + "="*50)
    exit()
    train_indices = {}
    test_indices = {}
    total_train = 0
    total_test = 0
    
    for _class in total_num.keys():
        train_indices[_class], test_indices[_class] = train_test_split(pick_percent=SPLIT[0], total_class_num = total_num[_class])
        total_train += len(train_indices[_class])
        total_test += len(test_indices[_class])

    for _class in train_indices.keys():
        print("CLASS: " + _class)
        print("\tTotal Number - Train:" + str(len(train_indices[_class])) + " | Percent:" + str(round(len(train_indices[_class])/total_train, 3)))
        print("\tTotal Number - Test:" + str(len(test_indices[_class])) + " | Percent:" + str(round(len(test_indices[_class])/total_test, 3)))
    print("="*50 + "\n")

    for _class in train_indices.keys():
        save_set(_class, train_indices[_class], "Train/")
        save_set(_class, test_indices[_class], dst_folder="Test/")

    
    