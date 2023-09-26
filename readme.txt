Step-1: create training, validation dataset using "data_Otsu_train_val.py".

Step-2: perform multi-scale deep learning with "main_augmented_100.py" for scale 100x100x12, "main_augmented_200.py" for scale 200x200x24, and "main_augmented_400" for scale 400x400x48.

Step-3: perform random forest classification with deep learning features and hand-crafted features with "transfer_learning.py". Note that transfer learning is a bad name as this algorithm is not transfer learning, but rather random forest.

Step-4: analyze classification result with "test_mouse_analysis.py".
