
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
print(torch.__version__)
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import pipeline
import torch.nn.functional as F
from PIL import Image
import deeplake
import hub

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

#Dataset
#Train
ds_paint = deeplake.load("hub://activeloop/domainnet-paint-train")
ds_clip = deeplake.load("hub://activeloop/domainnet-clip-train")
ds_real = deeplake.load("hub://activeloop/domainnet-real-train")
ds_quick = deeplake.load("hub://activeloop/domainnet-quick-train")
ds_sketch = deeplake.load("hub://activeloop/domainnet-sketch-train")
ds_info = deeplake.load("hub://activeloop/domainnet-info-train")

#Test 
ds_paint_test = deeplake.load("hub://activeloop/domainnet-paint-test")
ds_clip_test = deeplake.load("hub://activeloop/domainnet-clip-test")
ds_real_test = deeplake.load("hub://activeloop/domainnet-real-test")
ds_quick_test = deeplake.load("hub://activeloop/domainnet-quick-test")
ds_sketch_test = deeplake.load("hub://activeloop/domainnet-sketch-test")
ds_info_test = deeplake.load("hub://activeloop/domainnet-info-test")

#Adding domain column
def add_domain_column(tensor_ds, start, size, domain_value, from_end=False):
    # Calculate start and end indices
    if from_end:
        # When counting from the end of the dataset
        end = len(tensor_ds["images"]) - start
        start = end - size
    else:
        # When counting from the beginning of the dataset
        end = start + size

    tensor_ds_images = tensor_ds["images"][start:end]
    tensor_ds_labels = tensor_ds["labels"][start:end]

    numpy_images_list = tensor_ds_images.numpy(aslist=True)
    numpy_labels_list = tensor_ds_labels.numpy(aslist=True)

    domain_list = [domain_value] * len(numpy_images_list)

    # Combine labels, images, and domains into a single list of tuples
    combined_data = list(zip(numpy_labels_list, numpy_images_list, domain_list))
    df_new_data = pd.DataFrame(combined_data, columns=['Label', 'Image', 'Domain'])

    return df_new_data

new_data_paint = add_domain_column(ds_paint, 0, 2000, 0) #selecting 2000 images from the paint domain as assign the value 0
new_data_clip = add_domain_column(ds_clip, 2000, 2000, 1) #selecting 2000 images (after the first 2000 images) from the clipart domain as assign the value 1
new_data_real = add_domain_column(ds_real, 4000, 2000, 2) #selecting 2000 images (after the first 4000 images) from the real domain as assign the value 2
new_data_quick = add_domain_column(ds_quick, 6000, 2000, 3) #selecting 2000 images (after the first 6000 images) from the quickdraw domain as assign the value 3
new_data_sketch = add_domain_column(ds_sketch, 8000, 2000, 4) #selecting 2000 images (after the first 8000 images) from the sketch domain as assign the value 4
new_data_info = add_domain_column(ds_info, 10000, 2000, 5) #selecting 2000 images (after the first 10000 images) from the infograph domain as assign the value 5

new_data_paint_test = add_domain_column(ds_paint_test, 500, 500, 0, from_end=True) #selecting the last 200 images from the paint domain as assign the value 0
new_data_clip_test = add_domain_column(ds_clip_test, 1000, 500, 1, from_end=True)
new_data_real_test = add_domain_column(ds_real_test, 1500, 500, 2, from_end=True)
new_data_quick_test = add_domain_column(ds_quick_test, 2000, 500, 3, from_end=True)
new_data_sketch_test = add_domain_column(ds_sketch_test, 2500, 500, 4, from_end=True)
new_data_info_test = add_domain_column(ds_info_test, 3000, 500, 5, from_end=True)

all_data = pd.concat([new_data_paint, new_data_clip, new_data_real, new_data_quick, new_data_sketch, new_data_info], axis=0)
all_data.reset_index(drop=True, inplace=True)

all_data_test = pd.concat([new_data_paint_test, new_data_clip_test, new_data_real_test, new_data_quick_test, new_data_sketch_test, new_data_info_test], axis=0)
all_data_test.reset_index(drop=True, inplace=True)

#print(all_data.tail())

#resize the images 
def resize_image(image_array, target_size):
    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_array)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(resized_image)
#because the original size is different 
target_size = (128, 128)

all_data['Image'] = all_data['Image'].apply(lambda img: resize_image(img, target_size))
all_data_test['Image'] = all_data_test['Image'].apply(lambda img: resize_image(img, target_size))
all_data['Label'] = all_data['Label'].apply(lambda x: x[0] if len(x) > 0 else None)
all_data_test['Label'] = all_data_test['Label'].apply(lambda x: x[0] if len(x) > 0 else None)

#building a model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Updated input size for the linear layer
        self.fc_input_size = 32 * 32 * 32  # For 128x128 input image

        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Initialize the model
model = SimpleCNN(num_classes=7)

class CustomDataset(Dataset):
    def __init__(self, all_dataset):
        self.all_dataset = all_dataset

    def __len__(self):
        return len(self.all_dataset)

    def __getitem__(self, idx):
        # Extracting the image and domain
        image = self.all_dataset.iloc[idx]['Image']
        domain = self.all_dataset.iloc[idx]['Domain']

        image = np.array(image)

        # Converting to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) # CHW format
        domain = torch.tensor(domain, dtype=torch.int64)

        return image, domain
all_dataset = CustomDataset(all_data)
subset_dataset = Subset(all_dataset, range(4150))
train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)

all_dataset_test = CustomDataset(all_data_test)
subset_dataset_test = Subset(all_dataset_test, range(1200))
test_loader = DataLoader(subset_dataset_test, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, domains = data
        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, domains)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += domains.size(0)
        correct += (predicted == domains).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%')

print('Finished Training')
# Plot training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Over Epochs')
plt.legend()

plt.show()
torch.save(model.state_dict(), 'simple_cnn_model.pth')

# test 
test_losses = []
test_accuracies = []

model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, domains = data
        outputs = model(inputs.float())
        loss = criterion(outputs, domains)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += domains.size(0)
        correct += (predicted == domains).sum().item()

test_losses.append(test_loss / len(test_loader))
test_accuracy = 100 * correct / total  # Calculate test accuracy
test_accuracies.append(test_accuracy)

print(f'Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%')
