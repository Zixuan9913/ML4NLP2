
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
#print(f"Using {device} device")

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

 # Define a dictionary mapping from numerical labels to lists of string labels
domain_mapping = {
    0: [
        'an artistic painting, showcasing creative artwork',
        'a canvas expressing artistic skills',
        'painting displaying artistic talent',
        'creative art piece on canvas',
        'artistic expression through painting'
    ],
    1: [
        'clipart images, typically used for illustrations',
        'simple and colorful clipart',
        'illustrative clipart design',
        'digital clipart for presentations',
        'graphic clipart for creative use'
    ],
    2: [
        'real-world photographs, depicting true-to-life scenes',
        'high-resolution photo capturing reality',
        'lifelike photography with vivid details',
        'realistic photograph showcasing natural beauty',
        'true-to-life image capturing the essence of reality'
    ],
    3: [
        'quickdraw images, representing simple and fast sketches',
        'rapid drawn art in a hurry',
        'quick and simple line drawing',
        'speedy drawing, capturing the basics',
        'fast-drawn image, simplicity at its core'
    ],
    4: [
        'sketches, detailed hand-drawn images',
        'handcrafted sketch with attention to detail',
        'artistic sketch showing intricate designs',
        'detailed sketched diagram made by hand',
        'hand-drawn sketch with artistic flair'
    ],
    5: [
        'informational graphics, such as charts or graphs',
        'educational infographic presenting data',
        'graphical representation of information',
        'data-rich infographic for easy comprehension',
        'An infograph that is a visually appealing chart displaying key facts'
    ]
}

def replace_domains(domains):
# Flatten the array to one dimension if it's two-dimensional
    if len(domains.shape) == 2 and domains.shape[1] == 1:
        domains = domains.flatten()

    # Replace each numeric label with a randomly chosen string label from the corresponding list
    string_domains = np.array([random.choice(domain_mapping[domain]) for domain in domains])
    return string_domains

all_string_data = all_data

all_string_data['Domain'] = all_string_data['Domain'].apply(lambda x: random.choice(domain_mapping[x]))

class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        # Reusing your existing CNN layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc_input_size = 32 * 32 * 32

    def forward(self, x):
        # Forward pass through your CNN
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))

        # Instead of passing through the final classification layer, return the feature vector
        # Flatten the output for the feature vector
        features = torch.flatten(x, 1)
        return features

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1) #for mean pooling

class ZeroShotModel(nn.Module):
    def __init__(self, feature_extractor, embedding_size):
        super(ZeroShotModel, self).__init__()
        self.feature_extractor = feature_extractor
        # The CNN feature vector and BERT embeddings are of different sizes,
        # use a linear layer to project them to a common size. the former one is the image feature size and the second one is text embedding size
        self.projection = nn.Linear(self.feature_extractor.fc_input_size, embedding_size)

    def forward(self, image, text_embedding):
        image_features = self.feature_extractor(image)
        projected_image_features = self.projection(image_features)
        # The goal is to make projected_image_features similar to text_embedding
        return projected_image_features, text_embedding #in the same space (size)

class CustomDataset(Dataset):
    def __init__(self, all_dataset):
        self.all_dataset = all_dataset

    def __len__(self):
        return len(self.all_dataset)

    def __getitem__(self, idx):
        # Extracting the image and domain description
        image = self.all_dataset.iloc[idx]['Image']
        domain_description = self.all_dataset.iloc[idx]['Domain']

        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) # CHW format

        # Note that domain is now a string, not an integer
        return image, domain_description

all_string_dataset = CustomDataset(all_string_data)
subset_string_dataset = Subset(all_string_dataset, range(4150))
train_data_loader = DataLoader(subset_string_dataset, batch_size=64, shuffle=True)

num_epochs = 30

# Initialize your models
feature_extractor = FeatureExtractorCNN()
zero_shot_model = ZeroShotModel(feature_extractor, embedding_size=768)  # 768 for BERT base

# Example loss function
loss_fn = nn.MSELoss()

# Example training loop
optimizer = torch.optim.Adam(zero_shot_model.parameters(), lr=0.001)

# List to store loss values
batch_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, (image, domain_description) in enumerate(train_data_loader, 1):
        # Convert domain description to embedding
        domain_embedding = get_text_embedding(domain_description)
        projected_image_features, domain_embedding = zero_shot_model(image, domain_embedding)
        loss = loss_fn(projected_image_features, domain_embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append the loss
        batch_losses.append(loss.item())

        # Print batch loss
        print(f"Batch {batch_idx}/{len(train_data_loader)} Loss: {loss.item():.4f}")

# Plotting the batch loss
plt.figure(figsize=(10, 5))
plt.plot(batch_losses, label='Batch Loss')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Batch Loss During Training')
plt.legend()
plt.savefig('Batch_Loss_During_Training_2ndmodel.png')