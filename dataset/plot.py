import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def custom_sort(filename):
    numeric_part = "".join(filter(str.isdigit, filename))
    return int(numeric_part)


num_rows = 4
num_columns = 3
plot = 1

images_folder = 'images_train'
annotations_folder = 'anno_train'

images = sorted(os.listdir(images_folder), key=custom_sort)

plt.figure(figsize=(15, 5))

position = 1
for idx, image_file in enumerate(images):
    image_path = os.path.join(images_folder, image_file)
    img = Image.open(image_path)

    annotation_file = os.path.splitext(image_file)[0] + '.txt'
    annotation_path = os.path.join(annotations_folder, annotation_file)

    with open(annotation_path, 'r') as f:
        text = f.read()

    row = idx // num_columns  # Determine row
    col = idx % num_columns  # Determine column
    ax = plt.subplot(num_rows, num_columns, position)
    position += 1
    ax.imshow(img, cmap='gray')
    ax.set_title(text)
    ax.axis('off')  # To not display axis
    if (idx + 1) % 12 == 0:
        plt.tight_layout()
        output_path = f'image_label_check/output_plot_{plot}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        plot += 1
        position = 1
        plt.figure(figsize=(15, 5))

print('\u2705 Completed')
