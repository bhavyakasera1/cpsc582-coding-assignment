import matplotlib.pyplot as plt
import shutil
import os
import random

def make_unified_dataset(dataset_path):
    """
    make unified dataset of patches arranged by class
    """
    for patient in os.listdir(dataset_path):
        if patient == ".DS_Store":
            continue
        for image in os.listdir(f'{dataset_path}/{patient}/1'):
            if image == ".DS_Store":
                continue
            shutil.copy(f"{dataset_path}/{patient}/1/{image}", f"unified_dataset/1/{image}")
        for image in os.listdir(f'{dataset_path}/{patient}/0'):
            if image == ".DS_Store":
                continue
            shutil.copy(f"{dataset_path}/{patient}/0/{image}", f"unified_dataset/0/{image}")

def make_train_val_sets(dataset_path):
    train_len_0 = 12000
    train_len_1 = 12000
    val_len_0 = 3000
    val_len_1 = 3000
    
    images_0 = os.listdir(f'{dataset_path}/0')
    images_1 = os.listdir(f'{dataset_path}/1')
    random.shuffle(images_0)
    random.shuffle(images_1)

    train_images_0, val_images_0 = images_0[:train_len_0], images_0[train_len_0:train_len_0+val_len_0]
    train_images_1, val_images_1 = images_1[:train_len_1], images_1[train_len_1:train_len_1+val_len_1]

    for image in train_images_0:
        shutil.copy(f"{dataset_path}/0/{image}", f"hist_data/train/0/{image}")
    for image in train_images_1:
        shutil.copy(f"{dataset_path}/1/{image}", f"hist_data/train/1/{image}")
    for image in val_images_0:
        shutil.copy(f"{dataset_path}/0/{image}", f"hist_data/val/0/{image}")
    for image in val_images_1:
        shutil.copy(f"{dataset_path}/1/{image}", f"hist_data/val/1/{image}")

if __name__ == "__main__":
    # make_unified_dataset("breast_histopathology")
    make_train_val_sets("unified_dataset")
    # pass
    
