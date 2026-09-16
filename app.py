import os
import io
import base64
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image

# ==========================================
# APP & MODEL CONFIGURATION
# ==========================================

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_disease_mobilenetv2.keras")
SPLIT_DATASET_DIR = os.path.join(BASE_DIR, "split_dataset")
TEST_DATASET_DIR = os.path.join(SPLIT_DATASET_DIR, "test")

IMG_SIZE = (224, 224)
DISEASE_CONFIDENCE_THRESHOLD = 0.70  # 70% threshold as defined in train.py

CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Healthy Skin",
    "Melanoma",
    "vitiligo"
]

# Medical knowledge database
DISEASE_METADATA = {
    "Acne": {
        "title": "Acne Vulgaris",
        "category": "Inflammatory Dermatosis",
        "severity": "Mild to Moderate",
        "severity_badge": "warning",
        "urgency": "Routine Dermatological Care",
        "description": "Acne is a prevalent skin condition that occurs when hair follicles become clogged with oil (sebum) and dead skin cells, causing whiteheads, blackheads, papules, or cysts.",
        "symptoms": [
            "Whiteheads (closed plugged pores)",
            "Blackheads (open plugged pores)",
            "Tender red bumps (papules) or pimples with pus (pustules)",
            "Painful cysts or nodules beneath the skin surface"
        ],
        "common_causes": [
            "Excess sebum production",
            "Hair follicles clogged by dead skin cells and oil",
            "Cutibacterium acnes bacterial proliferation",
            "Hormonal fluctuations (androgens, stress, cycle)"
        ],
        "recommendations": [
            "Wash affected areas twice daily with a gentle, non-comedogenic cleanser.",
            "Use topical treatments such as salicylic acid, benzoyl peroxide, or retinoids.",
            "Avoid picking, squeezing, or popping lesions to prevent scarring and infection.",
            "Consult a dermatologist if over-the-counter products do not improve conditions within 6-8 weeks."
        ],
        "when_to_see_doctor": "Severe cystic nodules, unresponsive breakouts, or signs of scarring require prescription topical/oral therapy."
    },
    "Eczema": {
        "title": "Eczema (Atopic Dermatitis)",
        "category": "Chronic Inflammatory Barrier Disorder",
        "severity": "Moderate",
        "severity_badge": "warning",
        "urgency": "Dermatological Management",
        "description": "Eczema is a chronic skin disorder characterized by a weakened skin barrier, intense pruritus (itching), dry patches, and recurring inflammatory flare-ups.",
        "symptoms": [
            "Dry, cracked, or scaly patches",
            "Intense itching, often worsening at night",
            "Red to brownish-gray patches (often in skin creases)",
            "Small raised bumps that may weep fluid and crust when scratched"
        ],
        "common_causes": [
            "Genetic barrier dysfunction (filaggrin deficiency)",
            "Environmental allergens (dust, pollen, pet dander)",
            "Harsh soaps, detergents, fragrance chemicals",
            "Temperature extremes, dry air, emotional stress"
        ],
        "recommendations": [
            "Moisturize skin at least twice daily with thick, fragrance-free ointments or ceramide creams.",
            "Take short, lukewarm showers and pat skin dry gently instead of rubbing.",
            "Identify and avoid known environmental contact triggers and synthetic fabrics.",
            "Use OTC hydrocortisone sparingly for acute itch flare-ups as advised."
        ],
        "when_to_see_doctor": "Persistent pain, signs of bacterial infection (honey-colored crusting, fever), or sleep disruption."
    },
    "Healthy Skin": {
        "title": "Healthy / Normal Skin",
        "category": "Normal Skin Physiology",
        "severity": "Normal / Benign",
        "severity_badge": "success",
        "urgency": "Preventive Maintenance",
        "description": "The scanned image demonstrates regular pigmentation, uniform skin texture, and no prominent macroscopic dermatological abnormalities or suspicious lesions.",
        "symptoms": [
            "Smooth, consistent texture",
            "Uniform pigmentation and coloration",
            "No active inflammation, erythema, or irregular lesions",
            "Normal hydration and intact epidermal barrier"
        ],
        "common_causes": [
            "Balanced skin barrier and healthy microbiome",
            "Consistent hydration and sun protection",
            "Nutritious lifestyle and gentle skincare"
        ],
        "recommendations": [
            "Apply broad-spectrum sunscreen (SPF 30 or higher) every day, rain or shine.",
            "Cleanse gently and moisturize to preserve the protective stratum corneum barrier.",
            "Perform regular self-examinations to monitor for any new or evolving spots.",
            "Schedule an annual total-body skin check with a certified dermatologist."
        ],
        "when_to_see_doctor": "Continue regular annual preventative skin exams, or sooner if an existing spot changes shape, color, or size."
    },
    "Melanoma": {
        "title": "Melanoma (Malignant Skin Neoplasm)",
        "category": "Malignant Skin Cancer",
        "severity": "High / Critical",
        "severity_badge": "danger",
        "urgency": "Urgent Medical Evaluation Required",
        "description": "Melanoma is the most serious type of skin cancer, originating in the pigment-producing melanocytes. Early detection and immediate excision are critical for successful treatment.",
        "symptoms": [
            "ABCDE Signs: Asymmetry (uneven halves)",
            "Border irregularity (scalloped, notched, or blurred edges)",
            "Color variation (shades of brown, black, blue, pink, or white)",
            "Diameter larger than 6mm (pencil eraser size)",
            "Evolving: Rapid changes in size, shape, surface elevation, itching, or bleeding"
        ],
        "common_causes": [
            "Ultraviolet (UV) radiation from sunlight or tanning beds",
            "Multiple atypical or dysplastic nevi (moles)",
            "Fair skin, history of severe blistering sunburns",
            "Genetic family history or CDKN2A mutations"
        ],
        "recommendations": [
            "Schedule an immediate in-person evaluation with a board-certified dermatologist or oncologist.",
            "Do NOT attempt home remedies, squeezing, or topical acid treatments on suspicious lesions.",
            "Record when the lesion first appeared and note any bleeding, rapid changes, or pain.",
            "Avoid sun exposure and tanning beds."
        ],
        "when_to_see_doctor": "URGENT: Consult a medical dermatologist immediately for professional dermoscopy and potential biopsy."
    },
    "vitiligo": {
        "title": "Vitiligo",
        "category": "Autoimmune Pigmentary Disorder",
        "severity": "Moderate",
        "severity_badge": "info",
        "urgency": "Dermatological Consultation",
        "description": "Vitiligo is an autoimmune condition in which the immune system targets and destroys melanocytes, resulting in macules and patches of depigmentation (chalky white skin patches).",
        "symptoms": [
            "Depigmented patches of milky white skin",
            "Premature whitening or greying of hair on the scalp, eyebrows, or eyelashes",
            "Loss of color in mucosal tissues (inside of mouth or nose)",
            "Patches commonly symmetric on hands, face, genital areas, or body folds"
        ],
        "common_causes": [
            "Autoimmune destruction of pigment-producing melanocyte cells",
            "Genetic predisposition with polygenic susceptibility",
            "Oxidative stress, neurochemical factors, or sunburn triggers"
        ],
        "recommendations": [
            "Apply high-SPF (50+) broad-spectrum sunscreen to white patches to prevent sunburn and contrast.",
            "Consult a dermatologist regarding phototherapy (narrowband UVB) or topical immunomodulators.",
            "Cosmetic cover-up products (tanning creams, micropigmentation) can assist with aesthetic management.",
            "Supportive group counseling may help address emotional and social well-being."
        ],
        "when_to_see_doctor": "Consult a dermatologist early in the progression to explore repigmentation therapies and assess for associated thyroid autoantibodies."
    }
}

# ==========================================
# MODEL LOADER
# ==========================================

print("Loading Skin Disease MobileNetV2 Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("MobileNetV2 Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None


# ==========================================
# IMAGE QUALITY ANALYSIS
# ==========================================

def check_image_quality(image):
    """
    Evaluates image resolution, blur (Laplacian variance),
    and illumination (mean grayscale intensity).
    Matches and enhances predict.py checks.
    """
    if image is None or image.size == 0:
        return {
            "passed": False,
            "message": "No valid image data was received.",
            "metrics": {"blur_score": 0, "brightness": 0, "resolution": "0x0"}
        }

    height, width = image.shape[:2]
    resolution_str = f"{width}x{height}"

    if height < 150 or width < 150:
        return {
            "passed": False,
            "message": f"Image resolution is too low ({resolution_str}). Minimum recommended is 200x200.",
            "metrics": {"blur_score": 0, "brightness": 0, "resolution": resolution_str}
        }

    # Convert to grayscale for blur and brightness analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur score via Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    warnings = []
    is_acceptable = True

    if blur_score < 40:
        warnings.append("Image is blurry. A sharper image will improve accuracy.")
        if blur_score < 20:
            is_acceptable = False

    if brightness < 35:
        warnings.append("Image appears underexposed (too dark).")
        if brightness < 20:
            is_acceptable = False
    elif brightness > 235:
        warnings.append("Image appears overexposed (harsh glare/flash).")
        if brightness > 250:
            is_acceptable = False

    return {
        "passed": is_acceptable,
        "warnings": warnings,
        "message": "Image quality is optimal for analysis." if not warnings else " ".join(warnings),
        "metrics": {
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "resolution": resolution_str
        }
    }


# ==========================================
# ROI AND PREPROCESSING
# ==========================================

def extract_roi(image, use_roi=True):
    """
    Extracts central 60% of image if use_roi is True,
    matching get_skin_roi in predict.py.
    """
    if not use_roi:
        return image

    height, width = image.shape[:2]
    x1 = int(width * 0.20)
    x2 = int(width * 0.80)
    y1 = int(height * 0.20)
    y2 = int(height * 0.80)

    # Ensure valid dimensions
    if x2 > x1 and y2 > y1:
        return image[y1:y2, x1:x2]
    return image


def preprocess_cv_image(image):
    """
    Prepares OpenCV image for MobileNetV2 input:
    Resize to (224, 224), convert BGR to RGB, preprocess_input.
    """
    resized = cv2.resize(image, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    float_img = rgb.astype(np.float32)
    preprocessed = preprocess_input(float_img)
    batch_img = np.expand_dims(preprocessed, axis=0)
    return batch_img


def encode_image_base64(image):
    """Encodes OpenCV image to base64 JPEG data URL for frontend preview."""
    success, buffer = cv2.imencode(".jpg", image)
    if success:
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    return None


# ==========================================
# CORE INFERENCE LOGIC
# ==========================================

def run_prediction(image, use_roi=True):
    """
    Full pipeline: Quality Check -> ROI Extraction ->
    MobileNetV2 Inference -> Threshold Logic -> Medical Data Binding.
    """
    if model is None:
        return {
            "success": False,
            "error": "TensorFlow model is not loaded."
        }

    # 1. Quality evaluation
    quality = check_image_quality(image)
    if not quality["passed"]:
        return {
            "success": False,
            "error": quality["message"],
            "quality": quality
        }

    # 2. ROI Extraction
    roi = extract_roi(image, use_roi=use_roi)

    # 3. Preprocess
    processed = preprocess_cv_image(roi)

    # 4. Inference
    preds = model.predict(processed, verbose=0)[0]

    # 5. Extract probabilities
    predicted_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[predicted_index])

    all_probabilities = {}
    for name, prob in zip(CLASS_NAMES, preds):
        all_probabilities[name] = round(float(prob) * 100, 2)

    # Sort probabilities descending
    sorted_probs = sorted(
        [{"class": name, "probability": prob, "display_name": DISEASE_METADATA.get(name, {}).get("title", name)}
         for name, prob in all_probabilities.items()],
        key=lambda x: x["probability"],
        reverse=True
    )

    # 6. Fallback & Confidence Threshold Check (70% rule)
    threshold_met = True
    threshold_warning = None

    if predicted_class != "Healthy Skin":
        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            threshold_met = False
            threshold_warning = (
                f"Confidence for detected condition ({round(confidence * 100, 1)}%) "
                f"is below the clinical benchmark threshold of {int(DISEASE_CONFIDENCE_THRESHOLD * 100)}%. "
                f"As configured in training, low-confidence screenings should be reviewed carefully by a specialist."
            )

    # 7. Bind medical details
    metadata = DISEASE_METADATA.get(predicted_class, {})

    # Generate preview thumbnail of ROI
    roi_preview = encode_image_base64(roi)

    return {
        "success": True,
        "predicted_class": predicted_class,
        "display_name": metadata.get("title", predicted_class),
        "confidence": round(confidence * 100, 2),
        "threshold_met": threshold_met,
        "threshold_warning": threshold_warning,
        "probabilities": all_probabilities,
        "sorted_probabilities": sorted_probs,
        "quality": quality,
        "roi_preview": roi_preview,
        "medical_info": metadata
    }


# ==========================================
# API ROUTES
# ==========================================

@app.route("/")
def index():
    """Serves the main web application."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts either multipart file upload ('image') or
    JSON payload with base64 string ('image_base64').
    """
    try:
        use_roi = request.form.get("use_roi", "true").lower() in ["true", "1", "yes"]

        # Case A: Multipart file upload
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected."}), 400

            file_bytes = np.frombuffer(file.read(), np.uint8)
            cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Case B: JSON base64 webcam payload
        elif request.is_json and "image_base64" in request.json:
            b64_data = request.json["image_base64"]
            use_roi = request.json.get("use_roi", True)

            # Strip header if present (e.g. data:image/jpeg;base64,)
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]

            image_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        else:
            return jsonify({"success": False, "error": "No image provided via file or JSON."}), 400

        if cv_img is None:
            return jsonify({"success": False, "error": "Failed to decode image data."}), 400

        result = run_prediction(cv_img, use_roi=use_roi)
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server processing error: {str(e)}"}), 500


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """
    Returns a list of sample test images per class for 1-click testing.
    """
    samples = []
    if os.path.exists(TEST_DATASET_DIR):
        for class_name in CLASS_NAMES:
            class_folder = os.path.join(TEST_DATASET_DIR, class_name)
            if os.path.exists(class_folder):
                valid_files = [
                    f for f in os.listdir(class_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                ]
                # Pick up to 2 samples per class
                selected = valid_files[:2]
                for file_name in selected:
                    samples.append({
                        "class_name": class_name,
                        "filename": file_name,
                        "display_name": DISEASE_METADATA.get(class_name, {}).get("title", class_name),
                        "url": f"/api/sample-image/{class_name}/{file_name}"
                    })
    return jsonify({"success": True, "samples": samples})


@app.route("/api/sample-image/<class_name>/<filename>", methods=["GET"])
def api_sample_image(class_name, filename):
    """Serves a test sample image."""
    folder = os.path.join(TEST_DATASET_DIR, class_name)
    return send_from_directory(folder, filename)


@app.route("/api/predict-sample", methods=["POST"])
def api_predict_sample():
    """Directly predicts a sample image by class and filename."""
    data = request.json or {}
    class_name = data.get("class_name")
    filename = data.get("filename")
    use_roi = data.get("use_roi", True)

    if not class_name or not filename:
        return jsonify({"success": False, "error": "Class name and filename are required."}), 400

    file_path = os.path.join(TEST_DATASET_DIR, class_name, filename)
    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": "Sample image not found on server."}), 404

    cv_img = cv2.imread(file_path)
    if cv_img is None:
        return jsonify({"success": False, "error": "Could not read sample image."}), 400

    result = run_prediction(cv_img, use_roi=use_roi)
    return jsonify(result)


@app.route("/api/disease-info", methods=["GET"])
def api_disease_info():
    """Returns the comprehensive clinical database for all classes."""
    return jsonify({"success": True, "diseases": DISEASE_METADATA})


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Returns technical architecture and model metadata."""
    return jsonify({
        "success": True,
        "architecture": "MobileNetV2 (Transfer Learning with Pretrained ImageNet Weights)",
        "input_size": "224 x 224 x 3",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "confidence_threshold": f"{int(DISEASE_CONFIDENCE_THRESHOLD * 100)}%",
        "framework": "TensorFlow / Keras",
        "model_file": "skin_disease_mobilenetv2.keras",
        "roi_support": "Central 60% dynamic bounding crop + full image"
    })


# ==========================================
# SERVER ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Skin Disease Detection AI Web Application")
    print("Server running on: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
