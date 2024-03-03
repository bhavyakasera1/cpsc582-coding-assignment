from dataset import *
from utils import *
from torchvision import transforms
import torch
from model import ClassificationModelVGG16, ClassificationModelVGG19, ClassificationModelResNet18
import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def train(model, loss_fn, train_loader, val_loader, optimizer, epochs=5):
    """Train the model.
    Args:
        model: the model
        loss_fn: the loss function
        train_loader: the training data loader
        val_loader: the testing data loader
        optimizer: the optimizer
        epochs: the number of epochs to train
    Returns:
        train_losses: the training losses
        test_losses: the testing losses
    """
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    loop = tqdm.tqdm(range(1, epochs + 1))

    for epoch in loop:

        # train the model for one epoch
        train_loss, train_accuracy = train_epoch(model, loss_fn, train_loader, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # test the model for one epoch
        test_loss, test_accuracy = test_epoch(model, loss_fn, val_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(train_loss=train_loss, test_loss=test_loss, train_accuracy=train_accuracy, test_accuracy=test_accuracy)
    return train_losses, test_losses, train_accuracies, test_accuracies


def train_epoch(model, loss_fn, train_loader, optimizer):
    """Train the model for one epoch.
    Args:
        model: the model
        loss_fn: the loss function
        train_loader: the training data loader
        optimizer: the optimizer
    Returns:
        train_loss: the loss of the epoch
    """
    model.train()  # set model to training mode
    train_loss = 0
    train_accuracy = 0
    num_correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        bs = data.shape[0]

        preds = model.forward(data)
        loss = loss_fn(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print("predicted class = ", pred_class, "real class = ", target)
        pred_class = torch.argmax(preds, axis=1).detach().numpy()
        num_correct += (pred_class==target.detach().numpy()).sum()
        total += bs

        train_loss += loss.detach().numpy() * bs

    train_loss = train_loss/total
    train_accuracy = num_correct/total

    return train_loss, train_accuracy

def test_epoch(model, loss_fn, val_loader):
    """Test the model for one epoch.
    Args:
        model: the model
        loss_fn: the loss function
        val_loader: the testing data loader
    Returns:
        test_loss: the loss of the epoch
    """
    model.eval()  # set model to evaluation mode
    test_loss = 0
    test_accuracy = 0
    num_correct = 0
    total = 0

    with torch.no_grad():  # disable gradient calculation
        for data, target in val_loader:

            bs = data.shape[0]
            preds = model.forward(data)
            loss = loss_fn(preds, target)

            test_loss += loss.detach().numpy() * bs

            pred_class = torch.argmax(preds, axis=1).detach().numpy()
            num_correct += (pred_class==target.detach().numpy()).sum()
            total += bs

    test_loss = test_loss/total
    test_accuracy = num_correct/total

    return test_loss, test_accuracy

if __name__ == "__main__":
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

    # TODO: in place of "hist_data" enter correct path to data
    train_dataset, val_dataset = get_datasets(f"hist_data", train_transforms, val_transforms)
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size)

    model = ClassificationModelVGG16(num_classes=output_channels)
    model_name = f"classfication_model_vgg16_lr_{learning_rate}_epochs_{epochs}_adam"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.vgg.classifier.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_losses, test_losses, train_accuracies, test_accuracies = train(model, loss_fn, train_loader, val_loader, optimizer, epochs=epochs)
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': test_losses,
            }, f"{model_name}.pth")

    plt.subplot(2, 1, 1)
    plot_metrics(train_losses, test_losses, xlabel="Epoch", ylabel="Loss", title="Loss", save_path=f"{model_name}_losses.png")
    plt.subplot(3, 1, 3)
    plot_metrics(train_accuracies, test_accuracies, xlabel="Epoch", ylabel="Accuracy", title="Accuracy", save_path=f"{model_name}_accuracies.png")

