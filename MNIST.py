import numpy as np
import os
from data_source import data_sources

def load_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        if magic != 2051:
            raise ValueError("Invalid IDX3 file!")

        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape into (num_images, rows, cols)
        images = data.reshape(num_images, rows, cols)

        return images

# Usage
# images = load_idx3_ubyte("./dataset/train-images.idx3-ubyte")

data_dir = "dataset/"
mnist_dataset = {}
# Images
for key in ("training_images", "test_images"):
    with open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)

# Labels
for key in ("training_labels", "test_labels"):
    with open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=8
        )

x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)

print(
    "The shape of training images: {} and training labels: {}".format(
        x_train.shape, y_train.shape
    )
)
print(
    "The shape of test images: {} and test labels: {}".format(
        x_test.shape, y_test.shape
    )
)

import matplotlib.pyplot as plt

# Take the 60,000th image (indexed at 59,999) from the training set,
# reshape from (784, ) to (28, 28) to have a valid shape for displaying purposes.
mnist_image = x_train[59999, :].reshape(28, 28)

fig, ax = plt.subplots()
# Set the color mapping to grayscale to have a black background.
ax.imshow(mnist_image, cmap="gray")

# Display 5 random images from the training set.
num_examples = 5
seed = 147197952744
rng = np.random.default_rng(seed)

fig, axes = plt.subplots(1, num_examples)
for sample, ax in zip(rng.choice(x_train, size=num_examples, replace=False), axes):
    ax.imshow(sample.reshape(28, 28), cmap="gray")