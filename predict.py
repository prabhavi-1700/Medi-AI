import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "skin_disease_mobilenetv2.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Healthy Skin",
    "Melanoma",
    "vitiligo"
]


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# IMAGE QUALITY CHECK
# ==========================================

def check_image_quality(image):

    if image is None:
        return False, "No image received."

    # Check image size
    height, width = image.shape[:2]

    if height < 200 or width < 200:
        return False, "Image is too small."

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 50:
        return False, "Image is too blurry. Please capture a clearer image."

    # Brightness check
    brightness = np.mean(gray)

    if brightness < 40:
        return False, "Image is too dark. Please improve the lighting."

    if brightness > 230:
        return False, "Image is too bright. Please avoid strong lighting."

    return True, "Image quality is good."


# ==========================================
# SKIN REGION / ROI
# ==========================================

def get_skin_roi(image):

    height, width = image.shape[:2]

    # For the first version, use the
    # central region of the camera image.

    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    y1 = int(height * 0.20)
    y2 = int(height * 0.80)

    roi = image[y1:y2, x1:x2]

    return roi


# ==========================================
# PREPROCESS IMAGE
# ==========================================

def preprocess_image(image):

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    # OpenCV uses BGR
    # MobileNetV2 expects RGB

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    )

    image = preprocess_input(
        image
    )

    # Add batch dimension

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================================
# PREDICT
# ==========================================

def predict_skin_disease(image):

    # --------------------------------------
    # 1. QUALITY CHECK
    # --------------------------------------

    quality_ok, quality_message = check_image_quality(image)

    if not quality_ok:

        return {
            "success": False,
            "message": quality_message
        }


    # --------------------------------------
    # 2. GET SKIN REGION
    # --------------------------------------

    roi = get_skin_roi(image)


    # --------------------------------------
    # 3. PREPROCESS
    # --------------------------------------

    processed_image = preprocess_image(
        roi
    )


    # --------------------------------------
    # 4. MODEL PREDICTION
    # --------------------------------------

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]


    # --------------------------------------
    # 5. FIND CLASS
    # --------------------------------------

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )


    # --------------------------------------
    # 6. ALL PROBABILITIES
    # --------------------------------------

    all_probabilities = {}

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        all_probabilities[class_name] = round(
            float(probability) * 100,
            2
        )


    # --------------------------------------
    # 7. RESULT
    # --------------------------------------

    return {
        "success": True,
        "predicted_class": predicted_class,
        "confidence": round(
            confidence * 100,
            2
        ),
        "probabilities": all_probabilities
    }


# ==========================================
# CAMERA TEST
# ==========================================

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open camera.")

        exit()


    print("\nCamera started.")
    print("Press SPACE to capture.")
    print("Press Q to quit.\n")


    while True:

        ret, frame = camera.read()

        if not ret:

            print("ERROR: Could not read camera.")

            break


        # ----------------------------------
        # Display camera
        # ----------------------------------

        cv2.imshow(
            "Skin Disease Camera",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------
        # SPACE = CAPTURE
        # ----------------------------------

        if key == 32:

            print("\nChecking image quality...")

            result = predict_skin_disease(
                frame
            )


            # ------------------------------
            # Display result
            # ------------------------------

            print("\n===================================")

            if result["success"]:

                print(
                    "Predicted Class :",
                    result["predicted_class"]
                )

                print(
                    "Confidence      :",
                    result["confidence"],
                    "%"
                )

                print("\nClass probabilities:")

                for class_name, probability in result[
                    "probabilities"
                ].items():

                    print(
                        f"{class_name:15s}: "
                        f"{probability:.2f}%"
                    )

            else:

                print(
                    "Image rejected:"
                )

                print(
                    result["message"]
                )

            print(
                "===================================\n"
            )


        # ----------------------------------
        # Q = QUIT
        # ----------------------------------

        elif key == ord("q"):

            break


    camera.release()

    cv2.destroyAllWindows()