import os


def get_txt_files(directory):
    """
    Get a set of all .txt files in a directory.

    Parameters:
    - directory: The path to the directory.

    Returns:
    - A set of .txt file names.
    """
    return {f for f in os.listdir(directory) if f.endswith('.txt')}


def compare_directories(dir1, dir2):
    """
    Compare .txt files in two directories and ensure they contain the same file names.

    Parameters:
    - dir1: Path to the first directory.
    - dir2: Path to the second directory.

    Prints the results of the comparison.
    """
    files1 = get_txt_files(dir1)
    files2 = get_txt_files(dir2)

    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    if not only_in_dir1 and not only_in_dir2:
        print("Both directories contain the same .txt files.")
    else:
        if only_in_dir1:
            print("Files only in directory 1:")
            for f in only_in_dir1:
                print(f)
        if only_in_dir2:
            print("Files only in directory 2:")
            for f in only_in_dir2:
                print(f)


folder1 = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\VOC\\2024-05-04_5-Fold_Cross-val\split_4\\split1\labels'
folder2 = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\VOC_classification_10\\2024-06-01_5-Fold_Cross\split_4\\split1\labels'
compare_directories(folder1, folder2)
