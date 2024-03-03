from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

def get_datasets(dataset_path, train_transforms=None, val_transforms=None):
    """
    make dataset for training and validation
    """
    print("[INFO] loading the training and validation dataset...")
    train_dataset = ImageFolder(root=f"{dataset_path}/train",
            transform=train_transforms)
    val_dataset = ImageFolder(root=f"{dataset_path}/val", 
            transform=val_transforms)
    print("[INFO] training dataset contains {} samples...".format(
            len(train_dataset)))
    print("[INFO] validation dataset contains {} samples...".format(
            len(val_dataset)))
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size):
    """
    make dataloaders
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader