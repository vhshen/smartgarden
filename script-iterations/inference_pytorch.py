# Add the path to torchvision - change as needed
import sys
sys.path.insert(0, '/home/vivian/dev/torchvision')

# Choose an image to pass through the model
test_image = 'images/dog.jpg'

# Imports
import torch, json
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image

# Import matplotlib and configure it for pretty inline plots
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

print("[INFO] imports finished...")

# Prepare the labels
with open("imagenet-simple-labels.json") as f:
    labels = json.load(f)

print("[INFO] labels loaded...")

# First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# Load the image
image = Image.open(test_image)
plt.imshow(image), plt.xticks([]), plt.yticks([])

print("[INFO] data transformed...")

# Now apply the transformation, expand the batch dimension, and send the image to the GPU
image = data_transform(image).unsqueeze(0).cuda()

print("[INFO] transformation applied...")

# Download the model if it's not there already. It will take a bit on the first run, after that it's fast
model = models.resnet50(pretrained=True)
print("[INFO] models downloaded...")
# Send the model to the GPU 
model.cuda()
# Set layers such as dropout and batchnorm in evaluation mode
model.eval();
print("[INFO] model send to GPU and evaluated...")

# Get the 1000-dimensional model output
out = model(image)
# Find the predicted class
print("Predicted class is: {}".format(labels[out.argmax()]))
