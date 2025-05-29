import os

train_img_dir = r"data/images/train"
val_img_dir = r"data/images/val"
train_label_dir = r"data/labels/train"
val_label_dir = r"data/labels/val"

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)


def rename_images_and_create_labels(img_directory, label_directory, start_index):

    index = start_index
    for filename in sorted(os.listdir(img_directory)):
        if os.path.isfile(os.path.join(img_directory, filename)):
            new_filename = f"screwdriver_{index}.png"
            new_filepath = os.path.join(img_directory, new_filename)

            if not os.path.exists(new_filepath):
                os.rename(os.path.join(img_directory, filename), new_filepath)
                print(f"Renamed {filename} to {new_filename}")
            else:
                print(f"Skipping renaming for {filename} as {new_filename} already exists")

            label_filename = f"screwdriver_{index}.txt"
            label_path = os.path.join(label_directory, label_filename)
            if not os.path.exists(label_path):
                open(label_path, 'w').close()  # Create an empty .txt file
                print(f"Created label file: {label_filename}")
            else:
                print(f"Label file {label_filename} already exists, skipping creation")

            index += 1
    return index


next_index = rename_images_and_create_labels(train_img_dir, train_label_dir, start_index=1)

rename_images_and_create_labels(val_img_dir, val_label_dir, start_index=next_index)