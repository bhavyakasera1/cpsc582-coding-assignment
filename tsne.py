from model import *
from dataset import *
from utils import *
from torchvision import transforms
import torch
import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights

if __name__=="__main__":
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = 128
    learning_rate = 1e-4
    epochs = 15
    output_channels = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: add correct path to dataset
    train_dataset, val_dataset = get_datasets(f"hist_data", train_transforms, val_transforms)
    val_dataset = torch.utils.data.Subset(val_dataset, random.sample(range(len(val_dataset)), 1500))
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size)

    model = ClassificationModelVGG19()
    # TODO: enter path to model checkpoint
    checkpoint = torch.load("classfication_model_vgg19_lr_0.0001_epochs_20.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    labels = []
    features = None

    intermediate = {}

    def get_features(name):
        def hook(model, input, output):
            intermediate[name] = output.detach()

        return hook
    
    model.vgg.classifier[3].register_forward_hook(get_features("feats"))

    for data, target in val_loader:
        output = model(data)
        feats = torch.flatten(intermediate["feats"], 1)
        if features is None:
            features = feats.detach().numpy()
        else:
            features = np.concatenate((features, feats), axis=0)
        labels.extend(target.tolist())
    
    print("features = ", features.shape)
    
    tsne = TSNE(n_components=3).fit_transform(features)

    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
    
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range
    
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tz = tsne[:, 2]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    tz = scale_to_01_range(tz)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colours = ['#1f77b4', '#ff7f0e']
    for i in range(2):
        label = labels[i]
        indices = [i for i, l in enumerate(labels) if l == label]
        tx_pick = np.take(tx, indices)
        ty_pick = np.take(ty, indices)
        tz_pick = np.take(tz, indices)
        ax.scatter(tx_pick, ty_pick, tz_pick, c=colours[i], label=label)
    ax.legend(loc='best')
    
    plt.show()

