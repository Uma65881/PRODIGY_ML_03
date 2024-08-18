import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

# Paths to the dataset folders
train_dir = "C:/Users/HP/Desktop/PRODIGY_ML_03/train/"
test_dir = "C:/Users/HP/Desktop/PRODIGY_ML_03/test1/"

print("Loading training images...")


# Function to load and preprocess images with a limit
def load_images(data_dir, max_cats=4000, max_dogs=4000):
    images = []
    labels = []
    num_cats = 0
    num_dogs = 0

    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            if "cat" in file and num_cats < max_cats:
                label = "cat"
                num_cats += 1
            elif "dog" in file and num_dogs < max_dogs:
                label = "dog"
                num_dogs += 1
            else:
                continue

            img_path = os.path.join(data_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img)
            labels.append(label)

            # Stop loading once we have enough images
            if num_cats >= max_cats and num_dogs >= max_dogs:
                break

    print(f"Loaded {num_cats} cat images and {num_dogs} dog images.")
    return images, labels


# Load images
images, labels = load_images(train_dir, max_cats=4000, max_dogs=4000)
print(f"Total images loaded: {len(images)}.")

# Check if both classes are present
if len(set(labels)) < 2:
    raise ValueError(
        "The dataset must contain at least two classes. Ensure both cats and dogs are present in the training data."
    )

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0

# Flatten images for SVM input
images_flattened = images.reshape(len(images), -1)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    images_flattened, labels_encoded, test_size=0.2, random_state=42
)

print("Applying PCA...")
# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("PCA applied.")

print("Training SVM model...")
# Create and train the SVM model
svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_train_pca, y_train)
print("Model training complete.")

# Make predictions on validation set
y_pred = svm_model.predict(X_test_pca)

# Evaluate the model
print(f"Accuracy on validation set: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Loading test images...")


# Function to load test images from 'test1' folder
def load_test_images(test_dir):
    images = []
    image_ids = []
    for file in os.listdir(test_dir):
        if file.endswith(".jpg"):
            image_id = file.split(".")[0]
            img_path = os.path.join(test_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img)
            image_ids.append(image_id)
    return images, image_ids


# Load test images
test_images, test_image_ids = load_test_images(test_dir)
print(f"Loaded {len(test_images)} test images.")

# Convert to numpy array
test_images = np.array(test_images) / 255.0
test_images_flattened = test_images.reshape(len(test_images), -1)

print("Applying PCA to test images...")
# Apply PCA to test images
test_images_pca = pca.transform(test_images_flattened)

# Predict on test images
test_predictions = svm_model.predict(test_images_pca)

# Map predictions to class labels
test_predictions_labels = le.inverse_transform(test_predictions)

print("Creating submission file...")
# Create submission DataFrame
submission_df = pd.DataFrame({"id": test_image_ids, "label": test_predictions_labels})
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: 'submission.csv'")
