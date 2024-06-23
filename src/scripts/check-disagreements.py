import os


def read_filenames_from_file(file_path):
    """Read filenames from a text file and return them as a set."""
    with open(file_path, 'r') as file:
        filenames = {line.strip() for line in file.readlines()}
    return filenames


def list_filenames_in_folder(folder_path):
    """List filenames in a folder and return them as a set."""
    filenames = {name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))}
    return filenames


def compare_filenames(modified_labels_file, new_labels_folder):
    # Step 1: Read filenames from the modified labels file
    modified_filenames = read_filenames_from_file(modified_labels_file)

    # Step 2: List filenames in the new labels folder
    new_filenames = list_filenames_in_folder(new_labels_folder)

    # Step 3: Compare the sets of filenames
    matching_files = modified_filenames & new_filenames
    non_matching_files_in_modified = modified_filenames - new_filenames
    non_matching_files_in_new = new_filenames - modified_filenames

    # Print the counts and results
    print(f"Number of files in modified labels: {len(modified_filenames)}")
    print(f"Number of files in new labels: {len(new_filenames)}")
    print(f"Number of matching files: {len(matching_files)}")
    print(f"Number of non-matching files in modified labels: {len(non_matching_files_in_modified)}")
    print(f"Number of non-matching files in new labels: {len(non_matching_files_in_new)}")

    if matching_files:
        print("Matching files:")
        for file in matching_files:
            print(file)
    else:
        print("No matching files found.")

modified_labels_file = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Classification Noise\\Noise-25\modified_files_log.txt'
new_labels_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\\test\\threshold\cls-25\\new label'

compare_filenames(modified_labels_file, new_labels_folder)
