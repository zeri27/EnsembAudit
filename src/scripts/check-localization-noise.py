import os

def compare_labels(original_folder, modified_folder, modified_files_log):
    with open(modified_files_log, 'r') as log_file:
        modified_files = log_file.read().splitlines()

    for file_name in modified_files:
        original_label_path = os.path.join(original_folder, file_name)
        modified_label_path = os.path.join(modified_folder, file_name)

        with open(original_label_path, 'r') as original_label_file:
            original_label = original_label_file.readlines()

        with open(modified_label_path, 'r') as modified_label_file:
            modified_label = modified_label_file.readlines()

        if original_label != modified_label:
            print(f"File: {file_name}")
            print("WAS:")
            for line in original_label:
                print(line.strip())
            print("CHANGED TO:")
            for line in modified_label:
                print(line.strip())
            print("\n")

# Example usage
original_folder = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL\labels"
modified_folder = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Localization Noise\\Noise-10\labels"
modified_files_log = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Localization Noise\\Noise-10\modified_files_log.txt"

compare_labels(original_folder, modified_folder, modified_files_log)
