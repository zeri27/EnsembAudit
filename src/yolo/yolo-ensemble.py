from src.yolo.ensembling.strategies.strategy import EnsembleStrategy
from src.yolo.entities.yolo_ensembling_framework import YOLOEnsemble

# Model Weights

model_paths = ['C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\\Noise\Classification Noise\\10\split1\weights\\best.pt',
               'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\\Noise\Classification Noise\\10\split2\weights\\best.pt',
               'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\\Noise\Classification Noise\\10\split3\weights\\best.pt',
               'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\\Noise\Classification Noise\\10\split4\weights\\best.pt',
               'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\\Noise\Classification Noise\\10\split5\weights\\best.pt']

#img_path = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL\\2024-06-08_5-Fold_Cross-val\split_5\\val\images'
#label_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL\\2024-06-08_5-Fold_Cross-val\split_5\\val\labels'

# Main Test Set

img_path = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Test Set\main\images'
label_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Test Set\main\labels'

# Output Directories

output_dir = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\\test\\validation\\new annotation'
actual_dir = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\\test\\validation\old annotation'

new_labels_dir = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\\test\\validation\\new label"
old_labels_dir = "C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\\test\\validation\old label"

yoloEnsemble = YOLOEnsemble(model_paths, img_path)

predictions = yoloEnsemble.predict_images()
normalized = yoloEnsemble.normalize_predictions(predictions)
ensembled = yoloEnsemble.ensemble_predictions(EnsembleStrategy.FUSION, normalized)

yoloEnsemble.clear_output_folder(output_dir)
yoloEnsemble.draw_annotations(img_path, ensembled, output_dir)

yoloEnsemble.clear_output_folder(actual_dir)
#old = yoloEnsemble.drawOriginalLabels(label_folder, ensembled, actual_dir, old_labels_dir)
yoloEnsemble.validate(label_folder, ensembled, True, actual_dir)

yoloEnsemble.clear_output_folder(new_labels_dir)
#yoloEnsemble.save_yolo_labels(img_path, ensembled, old, new_labels_dir, old_labels_dir)
