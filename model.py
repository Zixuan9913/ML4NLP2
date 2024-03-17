!pip install -q hub
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

print(all_data.tail())