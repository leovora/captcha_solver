import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from sklearn.model_selection import train_test_split
import cv2
from keras.callbacks import EarlyStopping
import visualkeras
from PIL import ImageFont

# Set seed for reproducibility
SEED = 1111
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset preparation
DATASET_PATH = "captcha_dataset/Google_Recaptcha_V2_Images_Dataset/images"
#DATASET_PATH = "captcha_immagini/immagini/images"
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]

#Data directory setup
def load_image_paths_and_labels(dataset_path):
    data = [
        {"img_path": os.path.join(root, file), "label": os.path.basename(root)}
        for root, _, files in os.walk(dataset_path)
        if os.path.basename(root) != "Other"
        for file in files
        if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS
    ]
    return pd.DataFrame(data)

# Data splitting
dataset = load_image_paths_and_labels(DATASET_PATH).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
train_df, test_df = train_test_split(dataset, test_size=0.15, random_state=SEED)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

class_names = dataset["label"].unique()

# Utility functions for label encoding/decoding
def encode_label(label):
    return np.argmax(label == class_names)

def decode_label(label_idx):
    return class_names[label_idx]

# Load images and labels
def load_images_and_labels(df, img_size=(120, 120)):
    images, labels = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row['img_path'], cv2.IMREAD_COLOR)
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(encode_label(row['label']))
    return np.array(images), np.array(labels)

X_train, y_train = load_images_and_labels(train_df)
X_test, y_test = load_images_and_labels(test_df)

# Build the model
def build_model(input_shape=(120, 120, 3), num_classes=len(class_names)):
    model = keras.Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=100, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Display predictions
def display_predictions(model, X_test, y_test, num_samples=16):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    sample_images = X_test[indices]
    sample_labels = y_test[indices]

    for i in range(num_samples):
        img = sample_images[i]
        plt.figure(figsize=(5, 5)) 
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
        
        # Make prediction
        pred = model.predict(img[np.newaxis])  
        pred_label = decode_label(np.argmax(pred))
        true_label = decode_label(sample_labels[i])

        # Title with correct or incorrect label
        if pred_label == true_label:
            plt.title(f"Pred: {pred_label} (correct)")
            plt.gca().add_patch(plt.Rectangle((0, 0), 120, 120, fill=False, edgecolor='green', linewidth=7))
        else:
            plt.title(f"Pred: {pred_label} - True: {true_label}")
            plt.gca().add_patch(plt.Rectangle((0, 0), 120, 120, fill=False, edgecolor='red', linewidth=7))
        
        # Show the plot
        plt.show()

# Example usage
display_predictions(model, X_test, y_test)

# Save the trained model
model.save('trained_model.h5')

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()

plot_training_history(history)