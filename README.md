# PRODIGY_ML_03
# PRODIGY_ML_03
SVM Classifier for Cats vs. Dogs
Overview
This project involves building a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The classifier is trained using a set of labeled images and evaluated on a separate test set to measure its performance.

Project Structure
svm_cats_vs_dogs_classifier.py: The main script for training and evaluating the SVM model.
submission.csv: Output file containing predictions for test images.
train/: Directory containing training images, labeled as cat. or dog..
test1/: Directory containing test images.
Requirements
Python 3.x
OpenCV (cv2)
NumPy
Pandas
scikit-learn
Script Overview
Data Preparation
Loading Images:

Images are loaded from the train/ directory.
The code preprocesses images by resizing them to 64x64 pixels and converting them to grayscale.
A maximum of 250 images per class (cat and dog) are loaded.
Feature Extraction:

Images are normalized and flattened into 1D arrays.
Principal Component Analysis (PCA) is applied to reduce dimensionality.
Training:

An SVM model with a linear kernel is trained on the processed training images.
Evaluation:

The model is evaluated using a validation set extracted from the training data.
Performance metrics such as accuracy, precision, recall, and F1-score are reported.
Testing
Loading Test Images:

Test images are loaded from the test1/ directory.
They are preprocessed similarly to the training images.
Prediction:

The trained SVM model predicts labels for test images.
Predictions are saved in submission.csv with image IDs and predicted labels.
Usage
To run the script, execute the following command in your terminal:
python svm_cats_vs_dogs_classifier.py
Results
After running the script, the model’s performance metrics on the validation set will be printed to the terminal. The predictions for the test set will be saved in submission.csv.

Notes
Ensure that the directory paths (train/ and test1/) are correctly set in the script.
Adjust the number of PCA components if necessary to optimize performance.
