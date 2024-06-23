import os
import random
from PIL import Image


def add_localization_noise_to_labels(label_folder, image_folder, noise_percentage, modified_log_file):
    # Create a modified log file
    with open(modified_log_file, 'w') as log:
        # List all label files in the label folder
        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

        # Determine the number of files to modify based on noise percentage
        num_files_to_modify = int(len(label_files) * (noise_percentage / 100))

        # Randomly select files to modify
        files_to_modify = random.sample(label_files, num_files_to_modify)

        # Iterate over the selected files
        for file_name in files_to_modify:
            # Read the label file
            label_file_path = os.path.join(label_folder, file_name)
            with open(label_file_path, 'r') as label_file:
                yolo_label = label_file.readlines()

            # Get the corresponding image size
            image_name = os.path.splitext(file_name)[0] + ".jpg"  # Assuming image files have .jpg extension
            image_path = os.path.join(image_folder, image_name)
            image_width, image_height = get_image_size(image_path)

            # Add localization noise to the label
            noisy_label = add_localization_noise(yolo_label, image_width, image_height)

            # Write the modified label back to the file
            with open(label_file_path, 'w') as modified_label_file:
                modified_label_file.writelines(noisy_label)

            # Write the modified file name to the log
            log.write(file_name + '\n')


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def add_localization_noise(yolo_label, image_width, image_height):
    noisy_label = []
    for line in yolo_label:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        # Calculate absolute coordinates
        x_min = (x_center - box_width / 2) * image_width
        y_min = (y_center - box_height / 2) * image_height
        x_max = (x_center + box_width / 2) * image_width
        y_max = (y_center + box_height / 2) * image_height

        # Calculate noise ranges (randomly between 10% to 50% of bounding box size)
        noise_range_min = 0.1  # Minimum noise range
        noise_range_max = 0.5  # Maximum noise range

        # Randomly select noise range between 10% to 50%
        noise_range = random.uniform(noise_range_min, noise_range_max)

        x_noise = int(noise_range * (x_max - x_min))
        y_noise = int(noise_range * (y_max - y_min))

        # Add noise
        x_min += random.randint(-x_noise, x_noise)
        y_min += random.randint(-y_noise, y_noise)
        x_max += random.randint(-x_noise, x_noise)
        y_max += random.randint(-y_noise, y_noise)

        # Ensure the bounding box remains within the image boundaries
        x_min = max(0, min(x_min, image_width))
        y_min = max(0, min(y_min, image_height))
        x_max = max(0, min(x_max, image_width))
        y_max = max(0, min(y_max, image_height))

        # Convert back to YOLO format
        x_center = (x_min + x_max) / (2 * image_width)
        y_center = (y_min + y_max) / (2 * image_height)
        box_width = (x_max - x_min) / image_width
        box_height = (y_max - y_min) / image_height

        noisy_label.append(f"{int(class_id)} {x_center} {y_center} {box_width} {box_height}\n")

    return noisy_label


# Example usage
label_folder = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Localization Noise\\Noise-10\labels"
image_folder = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Localization Noise\\Noise-10\images"
noise_percentage = 10  # Example noise percentage
modified_log_file = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Localization Noise\\Noise-10\modified_files_log.txt"

add_localization_noise_to_labels(label_folder, image_folder, noise_percentage, modified_log_file)
