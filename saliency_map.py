from model import *
import torch
from dataset import *
from utils import plot_saliency_map
import numpy as np
import random

def get_saliency(model, data):
    model.eval()
    data.requires_grad_()
    scores = model(data)
    score_max_index = scores.argmax()
    prediction = score_max_index.item()
    score_max = scores[0,score_max_index]
    score_max.backward()
    saliency, _ = torch.max(data.grad.data.abs(),dim=1)
    return saliency, prediction

if __name__=="__main__":
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((50, 50)),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((50, 50)),
    ])

    model = ClassificationModelVGG19(num_classes=2)
    # TODO: enter path to model checkpoint
    checkpoint = torch.load("classfication_model_vgg19_lr_0.0001_epochs_20.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    train_dataset, val_dataset = get_datasets(f"hist_data", train_transforms, val_transforms)
    val_dataset = torch.utils.data.Subset(val_dataset, random.sample(range(len(val_dataset)), 25))
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, 1)

    for data, target in val_loader:
        saliency, prediction = get_saliency(model, data)
        data = data.detach().squeeze(0)
        img = np.transpose(data, (1, 2, 0))
        saliency = np.transpose(saliency, (1, 2, 0))
        plot_saliency_map(saliency, img, target, prediction)

