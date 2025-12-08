import torch
from transformers import ViTForImageClassification
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <image>")
    exit(-1)
model = ViTForImageClassification.from_pretrained("WillyIde545/dog_classifier")
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=120)
# model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(sys.argv[1])
pixel_values = transform(image).unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
pixel_values = pixel_values.to(device)

# Make the prediction
with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()

from datasets import load_dataset

dataset = load_dataset("amaye15/stanford-dogs")

labels = dataset['train'].features['label'].names
label_mapping = {i: label for i, label in enumerate(labels)}

print(f"Predicted breed: {label_mapping[predicted_class_idx]}")