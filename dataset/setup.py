import os
import shutil

image_dir = 'training_data/images'
annotation_dir = 'training_data/annotations'

output_image_dir = 'images_train'
output_annotation_dir = 'anno_train'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

total_images = 2700
current_image_count = 0
current_annotation_count = 0


def custom_sort(filename):
    numeric_part = "".join(filter(str.isdigit, filename))
    return int(numeric_part)


def numerical_sort(filename):
    if 'DS_Store' not in filename:
        num = int(filename.split('.')[0])
        return num
    else:
        pass


for subfolder in sorted(os.listdir(image_dir), key=custom_sort):
    subfolder_path = os.path.join(image_dir, subfolder)
    if os.path.isdir(subfolder_path):
        ds_store_path = os.path.join(subfolder_path, '.DS_Store')
        if os.path.exists(ds_store_path):
            os.remove(ds_store_path)
        for filename in sorted(os.listdir(subfolder_path), key=numerical_sort):
            if current_image_count >= total_images:
                break

            src_image_path = os.path.join(subfolder_path, filename)

            new_image_name = f"{current_image_count}.jpg"
            dest_image_path = os.path.join(output_image_dir, new_image_name)

            shutil.copy(src_image_path, dest_image_path)
            src_annotation_path = os.path.join(
                annotation_dir, f"{subfolder}.txt")
            with open(src_annotation_path, 'r', encoding='utf-8') as annotation_file:
                lines = annotation_file.readlines()
                annotation_line = lines[current_annotation_count % 27].strip()
            new_annotation_name = f"{current_image_count}.txt"
            dest_annotation_path = os.path.join(
                output_annotation_dir, new_annotation_name)

            with open(dest_annotation_path, 'w', encoding='utf-8') as new_annotation_file:
                path, label = annotation_line.split('\t')
                new_annotation_file.write(label)

            current_image_count += 1
            current_annotation_count += 1

if current_image_count < total_images:
    print(f"Not enough {total_images} images and annotations.")

print(f"\u2705 Complete")
