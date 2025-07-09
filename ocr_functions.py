#!/usr/bin/env python3
"""
OCR Functions - Standalone OCR model implementations

This module provides function-based implementations of various OCR models
for easy use in chat/interactive environments.

Usage:
    from ocr_functions import *
    
    # Initialize models
    initialize_ocr_models()
    
    # Run single OCR
    text = tesseract_ocr("image.jpg")
    
    # Run all OCR models
    results = run_all_ocr("image.jpg")
"""

import os
from typing import Dict, List
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try importing OCR libraries
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import kraken
    from kraken import pageseg, rpred
except ImportError:
    kraken = None

try:
    from google.cloud import vision
except ImportError:
    vision = None

# Global variables to store initialized models
easyocr_reader = None
paddleocr_reader = None
google_client = None

# Check model availability
models_available = {
    'tesseract': pytesseract is not None,
    'easyocr': easyocr is not None,
    'paddleocr': PaddleOCR is not None,
    'kraken': kraken is not None,
    'google_vision': vision is not None
}

def initialize_ocr_models():
    """Initialize all available OCR models"""
    global easyocr_reader, paddleocr_reader, google_client
    
    print("Initializing OCR models...")
    
    # Initialize EasyOCR
    try:
        if models_available['easyocr']:
            easyocr_reader = easyocr.Reader(['en', 'fr'])
            print("✓ EasyOCR initialized")
    except Exception as e:
        print(f"✗ EasyOCR initialization failed: {e}")
        models_available['easyocr'] = False
    
    # Initialize PaddleOCR
    try:
        if models_available['paddleocr']:
            paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
            print("✓ PaddleOCR initialized")
    except Exception as e:
        print(f"✗ PaddleOCR initialization failed: {e}")
        models_available['paddleocr'] = False
    
    # Initialize Google Cloud Vision
    try:
        if models_available['google_vision']:
            google_client = vision.ImageAnnotatorClient()
            print("✓ Google Cloud Vision initialized")
    except Exception as e:
        print(f"✗ Google Cloud Vision initialization failed: {e}")
        models_available['google_vision'] = False
    
    available_models = [k for k, v in models_available.items() if v]
    print(f"\nAvailable models: {available_models}")
    return available_models

def tesseract_ocr(image_path: str) -> str:
    """Run Tesseract OCR on an image"""
    try:
        if not models_available['tesseract']:
            return "Tesseract not available"
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Tesseract error on {image_path}: {e}")
        return ""

def easyocr_ocr(image_path: str) -> str:
    """Run EasyOCR on an image"""
    global easyocr_reader
    try:
        if not easyocr_reader:
            return "EasyOCR not initialized. Run initialize_ocr_models() first."
        results = easyocr_reader.readtext(image_path)
        text = ' '.join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        print(f"EasyOCR error on {image_path}: {e}")
        return ""

def paddleocr_ocr(image_path: str) -> str:
    """Run PaddleOCR on an image"""
    global paddleocr_reader
    try:
        if not paddleocr_reader:
            return "PaddleOCR not initialized. Run initialize_ocr_models() first."
        results = paddleocr_reader.ocr(image_path, cls=True)
        text_parts = []
        for line in results:
            if line:
                for word_info in line:
                    if len(word_info) > 1:
                        text_parts.append(word_info[1][0])
        return ' '.join(text_parts).strip()
    except Exception as e:
        print(f"PaddleOCR error on {image_path}: {e}")
        return ""

def kraken_ocr(image_path: str) -> str:
    """Run Kraken OCR on an image"""
    try:
        if not kraken:
            return "Kraken not available"
        # This is a simplified implementation
        # Kraken requires more complex setup with models
        return "Kraken requires model setup - see documentation"
    except Exception as e:
        print(f"Kraken error on {image_path}: {e}")
        return ""

def google_vision_ocr(image_path: str) -> str:
    """Run Google Cloud Vision OCR on an image"""
    global google_client
    try:
        if not google_client:
            return "Google Cloud Vision not initialized. Run initialize_ocr_models() first."
        
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        response = google_client.text_detection(image=image)
        texts = response.text_annotations
        
        if texts:
            return texts[0].description.strip()
        return ""
    except Exception as e:
        print(f"Google Vision error on {image_path}: {e}")
        return ""

def run_single_ocr(image_path: str, model_name: str) -> str:
    """Run a specific OCR model on an image"""
    ocr_functions = {
        'tesseract': tesseract_ocr,
        'easyocr': easyocr_ocr,
        'paddleocr': paddleocr_ocr,
        'kraken': kraken_ocr,
        'google_vision': google_vision_ocr
    }
    
    if model_name not in ocr_functions:
        return f"Unknown model: {model_name}"
    
    if not models_available.get(model_name, False):
        return f"Model {model_name} not available"
    
    return ocr_functions[model_name](image_path)

def run_all_ocr(image_path: str) -> Dict[str, str]:
    """Run all available OCR models on an image"""
    results = {}
    
    ocr_functions = {
        'tesseract': tesseract_ocr,
        'easyocr': easyocr_ocr,
        'paddleocr': paddleocr_ocr,
        'kraken': kraken_ocr,
        'google_vision': google_vision_ocr
    }
    
    for model_name, ocr_function in ocr_functions.items():
        if models_available.get(model_name, False):
            print(f"Running {model_name}...")
            results[model_name] = ocr_function(image_path)
        else:
            results[model_name] = f"{model_name} not available"
    
    return results

def get_available_models() -> List[str]:
    """Get list of available OCR models"""
    return [k for k, v in models_available.items() if v]

def get_pricing_info() -> Dict[str, str]:
    """Get pricing information for all OCR models"""
    return {
        'tesseract': 'Free',
        'easyocr': 'Free',
        'paddleocr': 'Free',
        'kraken': 'Free',
        'google_vision': '$1.50 per 1,000 images (first 1,000 free monthly)'
    }

def print_model_status():
    """Print status of all OCR models"""
    print("=== OCR Model Status ===")
    pricing = get_pricing_info()
    
    for model, available in models_available.items():
        status = "✓ Available" if available else "✗ Not available"
        price = pricing.get(model, "Unknown")
        print(f"{model:15} | {status:15} | {price}")

# Example usage
if __name__ == "__main__":
    print("OCR Functions Module")
    print_model_status()
    
    print("\nTo use:")
    print("1. initialize_ocr_models()")
    print("2. text = tesseract_ocr('image.jpg')")
    print("3. results = run_all_ocr('image.jpg')")
