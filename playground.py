import os
import cv2

if __name__=="__main__":
    dataset = "unified_dataset"
    for image in os.listdir(f'{dataset}/1'):
        if image == ".DS_Store":
            continue
        imfile = cv2.imread(f"{dataset}/1/{image}")
        if imfile.shape != (50, 50, 3):
            print("removing image: ", f"{dataset}/1/{image}")
            os.remove(f"{dataset}/1/{image}")
    for image in os.listdir(f'{dataset}/0'):
        if image == ".DS_Store":
            continue
        imfile = cv2.imread(f"{dataset}/0/{image}")
        if imfile.shape != (50, 50, 3):
            print("removing image: ", f"{dataset}/0/{image}")
            os.remove(f"{dataset}/0/{image}")