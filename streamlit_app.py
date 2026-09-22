import cv2
import numpy as np
import streamlit as st

import app as backend


st.set_page_config(
    page_title="Medi-AI Skin Screening",
    page_icon=":microscope:",
    layout="wide",
)


def display_prediction(result):
    if not result.get("success"):
        st.error(result.get("error", "Prediction failed."))
        return

    medical_info = result.get("medical_info", {})
    confidence = result["confidence"]

    if result.get("threshold_warning"):
        st.warning(result["threshold_warning"])

    summary, details = st.columns([1, 2])
    with summary:
        st.subheader(result["display_name"])
        st.metric("Confidence", f"{confidence:.2f}%")
        st.write(medical_info.get("category", ""))
        st.write(medical_info.get("urgency", ""))

    with details:
        st.subheader("Prediction probabilities")
        for item in result.get("sorted_probabilities", []):
            st.write(f"{item['display_name']}: {item['probability']:.2f}%")
            st.progress(min(item["probability"] / 100, 1.0))

    quality = result.get("quality", {})
    metrics = quality.get("metrics", {})
    st.subheader("Image quality")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Resolution", metrics.get("resolution", "Unknown"))
    metric_columns[1].metric("Blur score", metrics.get("blur_score", "Unknown"))
    metric_columns[2].metric("Brightness", metrics.get("brightness", "Unknown"))

    if medical_info:
        st.subheader("Information")
        st.write(medical_info.get("description", ""))

        info_columns = st.columns(2)
        with info_columns[0]:
            st.markdown("**Common symptoms**")
            for item in medical_info.get("symptoms", []):
                st.write(f"- {item}")
        with info_columns[1]:
            st.markdown("**Recommendations**")
            for item in medical_info.get("recommendations", []):
                st.write(f"- {item}")

        st.info(medical_info.get("when_to_see_doctor", "Consult a qualified dermatologist."))


st.title("Medi-AI Skin Screening")
st.caption("Educational screening only. This tool does not replace a professional dermatology examination.")

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png", "webp"],
)
use_roi = st.checkbox("Analyze the central skin region", value=True)

if uploaded_file is not None:
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("The uploaded file is not a valid image.")
    else:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)

        if st.button("Analyze image", type="primary"):
            with st.spinner("Analyzing image..."):
                display_prediction(backend.run_prediction(image, use_roi=use_roi))
