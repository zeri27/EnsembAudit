TU Delft Bachelor Thesis Project: EnsembAudit Repository
Author: Zeryab Alam

Paper: https://repository.tudelft.nl/record/uuid:1181a5fc-1d6e-4c3c-99b7-13499b9ec044

To replicate experiments from the paper:

Step 1: Download the VOC dataset via Ultralytics (https://docs.ultralytics.com/datasets/detect/voc/)
Step 2: Follow the Experiment Setup: Dataset Preparation
  - Use the 4000-datapoints script to create the test set from voc2007-testset
  - Combine all other datapoints into single folder via images and labels (as per YOLO requirements)
  - Use K-Folding script to generate the same splits as per the experiments
  - Noise can also be added using the scripts, however, this is random, so there will be slight variance in the results obtained
  - Follow the protocols in the paper for training the YOLOv8 models
  - Use the relevant scripts for testing
