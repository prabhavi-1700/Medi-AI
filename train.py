import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==============================
# 1. SETTINGS
# ==============================

DATASET_DIR = "split_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

MODEL_PATH = "skin_disease_mobilenetv2.keras"

# Minimum confidence required to call an image
# one of the four skin diseases.
#
# If disease confidence is below this value,
# the prediction will be Healthy Skin.
DISEASE_CONFIDENCE_THRESHOLD = 0.70


# ==============================
# 2. LOAD DATASET
# ==============================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ==============================
# 3. CHECK CLASS NAMES
# ==============================

class_names = train_dataset.class_names

print("\nClasses:")
print(class_names)


# ==============================
# 4. DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


# ==============================
# 5. PREPROCESSING
# ==============================

def preprocess_dataset(images, labels):

    images = preprocess_input(images)

    return images, labels


train_dataset = train_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==============================
# 6. LOAD MOBILENETV2
# ==============================

print("\n===================================")
print("LOADING MOBILENETV2")
print("===================================")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False


# ==============================
# 7. BUILD MODEL
# ==============================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    5,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ==============================
# 8. COMPILE MODEL
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# 9. MODEL SUMMARY
# ==============================

print("\n===================================")
print("MODEL SUMMARY")
print("===================================")

model.summary()


# ==============================
# 10. TRAIN MODEL
# ==============================

print("\n===================================")
print("STARTING TRAINING")
print("===================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ==============================
# 11. ORIGINAL TEST ACCURACY
# ==============================

print("\n===================================")
print("TEST EVALUATION")
print("===================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# ==============================
# 12. GENERATE PREDICTIONS
# ==============================

print("\n===================================")
print("GENERATING PREDICTIONS")
print("===================================")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    for prediction, label in zip(
        predictions,
        labels.numpy()
    ):

        # --------------------------------
        # Find class with highest probability
        # --------------------------------

        predicted_index = np.argmax(prediction)

        confidence = prediction[predicted_index]

        predicted_class = class_names[predicted_index]

        # --------------------------------
        # HEALTHY SKIN FALLBACK
        # --------------------------------
        #
        # Healthy Skin is one of the 5
        # trained classes.
        #
        # For the four disease classes:
        # if confidence < 70%,
        # use Healthy Skin as fallback.
        # --------------------------------

        if predicted_class != "Healthy Skin":

            if confidence < DISEASE_CONFIDENCE_THRESHOLD:

                predicted_index = class_names.index(
                    "Healthy Skin"
                )

        # --------------------------------
        # TRUE CLASS
        # --------------------------------

        true_index = np.argmax(label)

        y_true.append(true_index)
        y_pred.append(predicted_index)


# Convert lists to NumPy arrays

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==============================
# 13. CLASSIFICATION REPORT
# ==============================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ==============================
# 14. CONFUSION MATRIX
# ==============================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ==============================
# 15. DISPLAY CONFUSION MATRIX
# ==============================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "MobileNetV2 - Skin Disease Classification"
)

plt.tight_layout()

plt.show()


# ==============================
# 16. SAVE MODEL
# ==============================

print("\n===================================")
print("SAVING MODEL")
print("===================================")

model.save(MODEL_PATH)

print(
    f"Model saved successfully: {MODEL_PATH}"
)


# ==============================
# 17. FINAL SUMMARY
# ==============================

print("\n===================================")
print("TRAINING AND EVALUATION COMPLETE")
print("===================================")

print(
    f"Original Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Disease Confidence Threshold: "
    f"{DISEASE_CONFIDENCE_THRESHOLD:.2f}"
)

print("\nModel file:")
print(MODEL_PATH)


# ==============================
# 18. TRAINING ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.show()


# ==============================
# 19. TRAINING LOSS GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.show()