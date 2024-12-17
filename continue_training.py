import torch
from transformers import ViTForImageClassification
from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor
import torch

NUM_EPOCHS = 3

# Initialize the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=120)

# Load the saved state_dict
model.load_state_dict(torch.load('model.pth'))
dataset = load_dataset("amaye15/stanford-dogs")

# Define the transform
transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# Preprocess the dataset
def preprocess(example):
    example['pixel_values'] = torch.stack([transform(img) for img in example['pixel_values']])
    example['labels'] = torch.tensor(example['label'])
    return example

from torch.utils.data import DataLoader

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess, batched=True)

from torch.utils.data import DataLoader

# Split the dataset
train_dataset = dataset['train']
test_dataset = dataset['test']

#collate fn
def collate_fn(batch):
    pixel_values = torch.tensor([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler

# Define the loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define the learning rate scheduler
num_training_steps = len(train_loader) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


# Move to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = NUM_EPOCHS

from sklearn.metrics import accuracy_score
import time

training_start_time = time.time()

all_losses = []

for epoch in range(num_epochs):
    model.train()
    i = 0
    epoch_start_time = time.time()
    for batch in train_loader:
        i += 1
        batch_start_time = time.time()
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

        outputs = model(pixel_values=inputs, labels=labels)
        
        # Check if outputs are None
        if outputs.loss is None:
            raise ValueError("Model outputs loss is None")
        
        
        loss = outputs.loss
        logits = outputs.logits
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        print(f"Batch {i + 1}/{len(train_loader)} processed in {batch_duration:.4f} seconds")
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    all_losses.append(loss.item())
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.4f} seconds")

model_save_path = "/Users/wilsonide/Desktop/Development/PyTorch/Dog_Project/model.pth"
torch.save(model.state_dict(), model_save_path)

training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"Training completed in {training_duration:.4f} seconds")

import csv

# Save losses to a CSV file
with open('cross_entropy_losses.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss"])
    writer.writerows([[loss] for loss in all_losses])


model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy}")
