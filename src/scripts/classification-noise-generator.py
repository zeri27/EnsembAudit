import os
import random


def add_noise_to_labels(folder_path, num_classes, amount_of_noise):
    # List all label files in the folder
    label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Calculate the number of files to add noise to
    num_files_to_modify = int(len(label_files) * (amount_of_noise / 100))

    # Randomly select files to modify
    files_to_modify = random.sample(label_files, num_files_to_modify)

    # File to store the names of modified files
    modified_files_log = os.path.join(folder_path, 'modified_files_log.txt')

    with open(modified_files_log, 'w') as log_file:
        for file_name in files_to_modify:
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                original_class = int(parts[0])
                new_class = original_class
                # Ensure the new class is different from the original class
                while new_class == original_class:
                    new_class = random.randint(0, num_classes - 1)
                parts[0] = str(new_class)
                new_lines.append(' '.join(parts))

            with open(file_path, 'w') as file:
                for new_line in new_lines:
                    file.write(f"{new_line}\n")

            # Write the modified file name to the log
            log_file.write(f"{file_name}\n")

folder_path = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Classification Noise\\Noise-50\labels'
num_classes = 20  # Number of classes
amount_of_noise = 50  # Add noise to x% of the dataset

add_noise_to_labels(folder_path, num_classes, amount_of_noise)
