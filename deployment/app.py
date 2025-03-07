from io import BytesIO

import requests
import streamlit as st
from PIL import Image


# Set page config to wide mode
st.set_page_config(layout="wide")

def format_predictions(predictions_str):
    """Format predictions string into readable JSON"""
    try:
        # Safely evaluate the string representation of predictions
        predictions = eval(predictions_str)

        if not predictions:
            return "[]"  # Return empty list if no predictions
        
        formatted_json = "[\n"

        for bbox, class_name, confidence, text in predictions:
            formatted_json += '{\n';
            formatted_json += f"    \"bbox\": {bbox},\n"
            formatted_json += f"    \"class\": \"{class_name}\",\n"
            formatted_json += f"    \"confidence\": {confidence:.2f},\n"
            formatted_json += f"    \"text\": \"{text}\"\n"
            formatted_json += "  },\n"
        formatted_json = formatted_json.rstrip(",\n") + "\n]"

        return formatted_json

    except Exception as e:
        return f"Error formatting predictions: {str(e)}"

def process_image_url(url, api_url="http://localhost:8000"):
    """Process image from URL using the OCR API"""
    try:
        # Send image URL to prediction API
        response = requests.get(f"{api_url}/ocr", params={"image_url": url})
        response.raise_for_status()

        # Get predictions from headers
        predictions = response.headers.get("X-Predictions", "[]")

        # Display the processed image
        image = Image.open(BytesIO(response.content))
        return image, predictions

    except requests.RequestException as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None
    
def process_uploaded_file(file, api_url="http://localhost:8000"):
    """Process uploaded image file using the OCR API"""
    try:
        # Verify it's an image file
        try:
            Image.open(file)

            file.seek(0)
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            return None, None
        
        # Prepare the file for upload
        files = {"file": ("image.png", file, "image/png")}

        response = requests.post(f"{api_url}/ocr/upload", files=files)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Server Error: {error_detail}")
            return None, None
        
        response.raise_for_status()

        # Get predictions from headers
        predictions = response.headers.get("X-Predictions", "[]")

        # Display the processed image
        image = Image.open(BytesIO(response.content))
        return image, predictions
    
    except requests.RequestException as e:
        st.error(f"Error processing image: {str(e)}")
        if hasattr(e.response, "json"):
            try:
                error_detail = e.response.json().get("detail", "")
                st.error(f"Server details: {error_detail}")
            except:
                pass
        return None, None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None
    
def main():
    st.title("OCR App with Streamlit")

    # API endpoint configuration
    api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

    tab1, tab2 = st.tabs(["Process URL", "Process Image File"])

    with tab1:
        st.header("Process Image from URL")
        image_url = st.text_input("Enter Image URL")

        # Create two columns for images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")

            if image_url:
                try:
                    response = requests.get(image_url)
                    original_image = Image.open(BytesIO(response.content))
                    st.image(original_image, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")

        with col2:
            st.subheader("Processed Image")
             # Empty placeholder for processed image
            st.empty()

        if image_url and st.button("Process URL", key="process_url"):
            processed_image, predictions = process_image_url(image_url, api_url)

            if processed_image:
                with col2:
                    st.image(processed_image, use_container_width=True)
                
                st.subheader("Predictions")
                st.code(format_predictions(predictions), language="text")

    with tab2:
        st.header("Upload Image")

        # Create two columns for upload and process button
        upload_col, button_col = st.columns([4, 1])
        
        with upload_col:
            uploaded_file = st.file_uploader(
                "Choose an image file", type=["jpg", "jpeg", "png"]
            )
        with button_col:
            # Add some vertical space to align with uploader
            st.write("")
            st.write("")
            process_button = st.button("Process Image", key="process_upload")

        if uploaded_file is not None:
            # Create two columns for images
            col1, col2 = st.columns(2)

            # Original image
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_container_width=True)

            # Processed image placeholder
            with col2:
                st.subheader("Processed Image")
                # Empty placeholder for processed image
                st.empty()

            # Process when button is clicked
            if process_button:
                with st.spinner("Processing image..."):
                    image, predictions = process_uploaded_file(uploaded_file, api_url)
                    if image:
                        # Update the processed image in col2
                        with col2:
                            st.image(image, use_container_width=True)

                        # Predictions below
                        st.subheader("Detected Text")
                        st.code(format_predictions(predictions), language="text")

if __name__ == "__main__":
    main()