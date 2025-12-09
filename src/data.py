from torchvision import datasets, transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from model import get_normalization


class HFDogsDataset(Dataset):
    def __init__(self, hf_dataset_split, transform=None):
        self.data = hf_dataset_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["pixel_values"]             # PIL Image already
        label = item["label"]             # integer label

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size=224, model_type="resnet"):

    mean, std = get_normalization(model_type)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return train_tf, val_tf




def create_dataloaders(dataset, batch_size=32, img_size=224, model_type="resnet"):

    train_tf, val_tf = get_transforms(img_size, model_type)

    train_ds = HFDogsDataset(dataset["train"], transform=train_tf)
    val_ds   = HFDogsDataset(dataset["test"],  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader}




