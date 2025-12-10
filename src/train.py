import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
import os


from datasets import load_dataset
from .data import create_dataloaders
from .model import build_model, DogBreedClassifier


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main(
    dataset: Dataset,
    model: DogBreedClassifier,
    save_path: str,
    model_type: str = "resnet",
    batch_size: int = 64,
    img_size: int = 224,
    num_epochs: int = 5,
    lr: float = 1e-4,
    freeze_base: bool = False,
    history_path: str = None,
    resume: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    num_classes = dataset["train"].features["label"].num_classes

    dataloaders = create_dataloaders(
        dataset,
        batch_size=batch_size,
        img_size=img_size,
        model_type=model_type,  # if your data.py needs it for normalization
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start_epoch = 1
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    if resume and os.path.exists(save_path):
        print(f"Resuming from checkpoint: {save_path}")
        ckpt = torch.load(save_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if "history" in ckpt:
            history = ckpt["history"]
        if "epoch" in ckpt:
            start_epoch = history["epoch"][-1] + 1

        print(f"Starting from epoch {start_epoch}")

    # ----- Training Loop -----
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"| train_loss={train_loss:.4f}, train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if len(history["epoch"]) > 0:
            total_epoch = epoch + history["epoch"][-1]
        else:
            total_epoch = epoch

        # ---- update history ----
        history["epoch"].append(total_epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)


        torch.save({
            "epoch": total_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        }, save_path)
        print("Model saved!")

    if history_path is not None:
        # make sure directory exists
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f)
        print("Training history saved to:", history_path)

    return history

if __name__ == "__main__":
    main()



