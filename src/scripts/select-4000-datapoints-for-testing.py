import os
import random
import shutil


def create_folder_structure(base_path, subfolders):
    for subfolder in subfolders:
        path = os.path.join(base_path, subfolder)
        os.makedirs(path, exist_ok=True)


def move_files(file_list, source_folder, dest_folder, subfolder):
    for file_name in file_list:
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(dest_folder, subfolder, file_name)
        shutil.move(source, destination)


def split_test_data(test_image_folder, test_labels_folder, small_test_folder, remaining_test_folder,
                    small_test_size=4000):
    # Create the necessary folder structure
    create_folder_structure(small_test_folder, ['images', 'labels'])
    create_folder_structure(remaining_test_folder, ['images', 'labels'])

    # List all image files and their base names
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}  # Add other image extensions as needed
    all_images = [f for f in os.listdir(test_image_folder) if os.path.splitext(f)[1].lower() in image_extensions]
    image_base_names = {os.path.splitext(f)[0] for f in all_images}

    # List all label files and match them with image base names
    label_extension = '.txt'  # Adjust if your labels have a different extension
    all_labels = [f for f in os.listdir(test_labels_folder) if os.path.splitext(f)[1].lower() == label_extension]
    label_base_names = {os.path.splitext(f)[0] for f in all_labels}

    # Find matching pairs
    matching_base_names = image_base_names.intersection(label_base_names)
    matched_images = [f for f in all_images if os.path.splitext(f)[0] in matching_base_names]
    matched_labels = [f for f in all_labels if os.path.splitext(f)[0] in matching_base_names]

    # Ensure the image and label file lists are the same length
    assert len(matched_images) == len(matched_labels), "Mismatch between number of images and labels"

    # Shuffle and select 4000 files for the smaller test set
    random.seed(42)  # For reproducibility
    combined = list(zip(matched_images, matched_labels))
    random.shuffle(combined)
    small_test_set = combined[:small_test_size]
    remaining_test_set = combined[small_test_size:]

    # Move smaller test set files
    small_test_images, small_test_labels = zip(*small_test_set)
    move_files(small_test_images, test_image_folder, small_test_folder, 'images')
    move_files(small_test_labels, test_labels_folder, small_test_folder, 'labels')

    # Move remaining test set files
    remaining_test_images, remaining_test_labels = zip(*remaining_test_set)
    move_files(remaining_test_images, test_image_folder, remaining_test_folder, 'images')
    move_files(remaining_test_labels, test_labels_folder, remaining_test_folder, 'labels')

    print(
        f"Moved {len(small_test_images)} files to the smaller test set and {len(remaining_test_images)} files to the remaining test set.")


if __name__ == "__main__":
    test_image_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Test Set\images'
    test_labels_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Test Set\labels'
    small_test_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Test Set\\test'
    remaining_test_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Test Set\\remaining'

    split_test_data(test_image_folder, test_labels_folder, small_test_folder, remaining_test_folder)
