import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging
from datetime import datetime

# ----------------------------------------------------
# Constants and Configuration
# ----------------------------------------------------
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "webp"]
MAX_IMAGE_SIZE = 800
DEFAULT_QUALITY = 85

# ----------------------------------------------------
# Configure logging
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Class to handle all image processing operations."""
    
    @staticmethod
    def load_image(image_file) -> Optional[Image.Image]:
        """Safely load an image file."""
        try:
            return Image.open(image_file)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            st.error("Error loading image. Please try another file.")
            return None

    @staticmethod
    def resize_image(img: Image.Image, max_width: int = MAX_IMAGE_SIZE) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        w, h = img.size
        if w > max_width:
            ratio = max_width / float(w)
            new_height = int(ratio * h)
            return img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        return img

    @staticmethod
    def apply_adjustments(image: np.ndarray, adjustments: dict) -> np.ndarray:
        """Apply multiple image adjustments with error handling."""
        try:
            img = Image.fromarray(image)
            
            # Basic adjustments using PIL
            basic_adjustments = {
                'brightness': 'Brightness',
                'contrast': 'Contrast',
                'sharpness': 'Sharpness',
                'saturation': 'Color'
            }

            for adj, enhancer_name in basic_adjustments.items():
                if adj in adjustments:
                    enhancer = getattr(ImageEnhance, enhancer_name)(img)
                    img = enhancer.enhance(adjustments[adj])

            img = np.array(img.convert("RGB"))

            # Advanced adjustments using OpenCV
            if 'exposure' in adjustments:
                img = cv2.convertScaleAbs(img, alpha=adjustments['exposure'], beta=0)

            if 'white_balance' in adjustments:
                wb = adjustments['white_balance']
                wb_matrix = np.array([1 + wb/100, 1, 1 - wb/100])
                img = (img * wb_matrix).clip(0, 255).astype(np.uint8)

            if 'vibrance' in adjustments:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hsv[..., 1] = hsv[..., 1] * adjustments['vibrance']
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            if 'hue' in adjustments:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hsv[..., 0] = (hsv[..., 0].astype(int) + adjustments['hue']) % 180
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            if 'temperature' in adjustments:
                temp = adjustments['temperature']
                temp_matrix = np.array([1 + temp/100, 1, 1 - temp/100])
                img = (img * temp_matrix).clip(0, 255).astype(np.uint8)

            if 'noise_reduction' in adjustments:
                img = cv2.fastNlMeansDenoisingColored(img, None, adjustments['noise_reduction'], 10, 7, 21)

            if 'hdr_blending' in adjustments:
                alpha = adjustments['hdr_blending']
                img = cv2.addWeighted(img, alpha, cv2.GaussianBlur(img, (5, 5), 0), 1 - alpha, 0)

            if 'clarity' in adjustments:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                clarity_mask = cv2.GaussianBlur(gray, (0, 0), 3)
                img = cv2.addWeighted(img, 1 + adjustments['clarity'], 
                                    cv2.cvtColor(clarity_mask, cv2.COLOR_GRAY2RGB), 
                                    -adjustments['clarity'], 0)

            return img.clip(0, 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"Error applying adjustments: {e}")
            st.error("Error applying image adjustments")
            return image

    @staticmethod
    def apply_effect(image: np.ndarray, effect: str, params: dict, adjustments_history: Dict[str, dict]) -> np.ndarray:
        """Apply selected effect to the image while maintaining adjustment history."""
        try:
            # Store current adjustments in history if not exists
            if effect not in adjustments_history:
                adjustments_history[effect] = params.copy()
            else:
                # Update existing parameters with new ones
                adjustments_history[effect].update(params)

            # Apply the effect using stored parameters
            effect_params = adjustments_history[effect]
            
            if effect == "Pencil Sketch":
                return ImageProcessor._create_pencil_sketch(image, effect_params.get('blur_intensity', 21))
            elif effect == "Grayscale":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif effect == "Vintage":
                return ImageProcessor._create_vintage_effect(image)
            elif effect == "Oil Painting":
                return ImageProcessor._create_oil_painting_effect(image)
            else:
                return image

        except Exception as e:
            logger.error(f"Error applying effect {effect}: {e}")
            st.error(f"Error applying {effect} effect")
            return image

    @staticmethod
    def _create_pencil_sketch(image: np.ndarray, blur_intensity: int) -> np.ndarray:
        """Create enhanced pencil sketch effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (blur_intensity, blur_intensity), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(sketch, -1, kernel)

    @staticmethod
    def _create_vintage_effect(image: np.ndarray) -> np.ndarray:
        """Create a vintage photo effect."""
        img_float = image.astype(float) / 255.0
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia = cv2.transform(img_float, sepia_matrix)
        sepia = np.clip(sepia, 0, 1)
        sepia = cv2.GaussianBlur(sepia, (3, 3), 0)
        return (sepia * 255).astype(np.uint8)

    @staticmethod
    def _create_oil_painting_effect(image: np.ndarray) -> np.ndarray:
        """Create oil painting effect."""
        return cv2.xphoto.oilPainting(image, 7, 1)

class StreamlitUI:
    """Class to handle all Streamlit UI elements."""
    
    @staticmethod
    def setup_page():
        """Configure page settings and header."""
        st.set_page_config(
            page_title="Advanced Image Transformer",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("\U0001F3A8 Advanced Image Transformation Studio")
        st.markdown("""
        Transform your images with professional-grade adjustments and artistic effects:
        - \U0001F58C\U0000FE0F Multiple artistic effects including Pencil Sketch, Oil Painting, and Vintage
        - \U0001F3A7 Professional adjustment controls
        - ⚡ Real-time preview
        - \U0001F4BE High-quality export options
        """)

    @staticmethod
    def create_sidebar_controls() -> Tuple[str, dict, dict]:
        """Create and return all sidebar controls with enhanced adjustments."""
        st.sidebar.title("Controls")
        
        effect = st.sidebar.selectbox(
            "Choose Effect",
            ["None", "Pencil Sketch", "Grayscale", "Vintage", "Oil Painting"]
        )
        
        effect_params = {}
        if effect == "Pencil Sketch":
            effect_params['blur_intensity'] = st.sidebar.slider(
                "Sketch Detail", 1, 51, 21, step=2
            )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Image Adjustments")
        
        # Enhanced adjustments
        adjustments = {
            'brightness': st.sidebar.slider("Brightness", 0.5, 3.0, 1.0, step=0.1),
            'contrast': st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, step=0.1),
            'exposure': st.sidebar.slider("Exposure", 0.5, 2.0, 1.0, step=0.1),
            'white_balance': st.sidebar.slider("White Balance", -100, 100, 0, step=5),
            'saturation': st.sidebar.slider("Saturation", 0.0, 3.0, 1.0, step=0.1),
            'vibrance': st.sidebar.slider("Vibrance", 0.5, 2.0, 1.0, step=0.1),
            'hue': st.sidebar.slider("Hue", -50, 50, 0, step=1),
            'temperature': st.sidebar.slider("Temperature", -100, 100, 0, step=5),
            'sharpness': st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0, step=0.1),
            'clarity': st.sidebar.slider("Clarity", 0.0, 1.0, 0.0, step=0.1),
            'noise_reduction': st.sidebar.slider("Noise Reduction", 0, 100, 0, step=5),
            'hdr_blending': st.sidebar.slider("HDR Blending", 0.0, 1.0, 0.5, step=0.1)
        }
        
        return effect, adjustments, effect_params

    @staticmethod
    def create_export_options(result_image: Image.Image):
        """Create export options section."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Options")
        
        quality = st.sidebar.slider("Export Quality", 1, 100, DEFAULT_QUALITY)
        format_option = st.sidebar.selectbox("Export Format", ["PNG", "JPEG", "WebP"])
        
        if st.sidebar.button("Export Image"):
            try:
                buf = BytesIO()
                result_image.save(
                    buf,
                    format=format_option,
                    quality=quality if format_option != "PNG" else None
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name=f"processed_image_{timestamp}.{format_option.lower()}",
                    mime=f"image/{format_option.lower()}"
                )
                st.success("Image processed successfully!")
            except Exception as e:
                logger.error(f"Error exporting image: {e}")
                st.error("Error exporting image. Please try again.")

def main():
    """Main application function with enhanced adjustment history."""
    processor = ImageProcessor()
    ui = StreamlitUI()
    
    # Setup page
    ui.setup_page()
    
    # Create sidebar controls
    effect, adjustments, effect_params = ui.create_sidebar_controls()

    # Initialize session state for adjustment history if not exists
    if 'adjustments_history' not in st.session_state:
        st.session_state.adjustments_history = {}
    
    # Image upload
    image_file = st.file_uploader(
        "Upload an image",
        type=ALLOWED_EXTENSIONS,
        help="Supported formats: " + ", ".join(ALLOWED_EXTENSIONS)
    )
    
    if image_file is not None:
        # Load and process image
        input_image = processor.load_image(image_file)
        if input_image:
            input_image = processor.resize_image(input_image)
            np_image = np.array(input_image)
            
            # Apply effect if selected
            if effect != "None":
                processed = processor.apply_effect(
                    np_image, 
                    effect, 
                    effect_params, 
                    st.session_state.adjustments_history
                )
            else:
                processed = np_image.copy()
            
            # Apply adjustments
            result = processor.apply_adjustments(processed, adjustments)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(input_image, use_container_width=True)
            with col2:
                st.subheader("Processed Image")
                st.image(result, use_container_width=True)
            
            # Create export options
            ui.create_export_options(Image.fromarray(result))

if __name__ == "__main__":
    main()