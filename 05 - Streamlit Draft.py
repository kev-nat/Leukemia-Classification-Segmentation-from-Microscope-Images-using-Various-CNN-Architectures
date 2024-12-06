# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class LeukemiaAnalyzer:
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.class_names = ['Benign', 'Early', 'Pre', 'Pro']
        
    def load_models(self):
        """Load both classification and segmentation models"""
        try:
            self.classification_model = tf.keras.models.load_model('path_to_classification_model.h5')
            self.segmentation_model = tf.keras.models.load_model('path_to_segmentation_model.h5')
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)

    def classify_image(self, preprocessed_image):
        """Perform classification prediction"""
        predictions = self.classification_model.predict(preprocessed_image)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        return self.class_names[class_idx], confidence

    def segment_image(self, preprocessed_image):
        """Perform segmentation prediction"""
        mask = self.segmentation_model.predict(preprocessed_image)
        return mask[0]  # Remove batch dimension

def main():
    st.title("Leukemia Analysis Tool")
    st.write("Upload an image for leukemia classification and segmentation")

    # Initialize analyzer
    analyzer = LeukemiaAnalyzer()
    if not analyzer.load_models():
        st.error("Failed to load models. Please check model paths.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Create two columns for results
        col1, col2 = st.columns(2)
        
        # Preprocess image
        preprocessed_image = analyzer.preprocess_image(image)
        
        with col1:
            st.subheader("Classification Results")
            if st.button("Classify"):
                with st.spinner("Classifying..."):
                    predicted_class, confidence = analyzer.classify_image(preprocessed_image)
                    st.success(f"Predicted Class: {predicted_class}")
                    st.info(f"Confidence: {confidence:.2%}")
        
        with col2:
            st.subheader("Segmentation Results")
            if st.button("Segment"):
                with st.spinner("Segmenting..."):
                    mask = analyzer.segment_image(preprocessed_image)
                    
                    # Convert mask to visible format
                    mask_image = (mask * 255).astype(np.uint8)
                    st.image(mask_image, caption="Segmentation Mask", use_column_width=True)
                    
                    # Optional: Overlay mask on original image
                    overlay = cv2.addWeighted(
                        np.array(image.resize((224, 224))), 
                        0.7, 
                        cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB), 
                        0.3, 
                        0
                    )
                    st.image(overlay, caption="Overlay", use_column_width=True)

if __name__ == "__main__":
    main()