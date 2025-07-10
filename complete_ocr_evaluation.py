#!/usr/bin/env python3

import os
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import warnings
from difflib import SequenceMatcher
import re

warnings.filterwarnings('ignore')

# Configuration
FRENCH_DATASET_PATH = "French_OCR_dataset"
ENGLISH_DATASET_PATH = "English_OCR_dataset"
SAMPLES_PER_DATASET = 10
TOTAL_SAMPLES = SAMPLES_PER_DATASET * 2

# Global variables for models
easyocr_reader = None
google_client = None
doctr_model = None
surya_recognition_predictor = None
surya_detection_predictor = None

# Model availability flags
models_available = {
    'tesseract': False,
    'easyocr': False,
    'google_vision': False,
    'doctr': False,
    'surya': False
}

def check_and_import_libraries():
    global models_available
    
    # Tesseract
    try:
        import pytesseract
        models_available['tesseract'] = True
        print("Tesseract available")
    except ImportError:
        print("Tesseract not available")

    # EasyOCR
    try:
        import easyocr
        models_available['easyocr'] = True
        print("EasyOCR available")
    except ImportError:
        print("EasyOCR not available")

    # Google Vision
    try:
        from google.cloud import vision
        models_available['google_vision'] = True
        print("Google Vision available")
    except ImportError:
        print("Google Vision not available")

    # DocTR
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        models_available['doctr'] = True
        print("DocTR available")
    except ImportError:
        print("DocTR not available")

    # Surya
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        models_available['surya'] = True
        print("Surya available")
    except ImportError:
        print("Surya not available")

def initialize_ocr_models():
    global easyocr_reader, google_client, doctr_model
    global surya_recognition_predictor, surya_detection_predictor
    
    print("\nInitializing OCR models...")
    
    # EasyOCR
    if models_available['easyocr']:
        try:
            import easyocr
            easyocr_reader = easyocr.Reader(['en', 'fr'])
            print("EasyOCR initialized")
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            models_available['easyocr'] = False

    # Google Vision
    if models_available['google_vision']:
        try:
            from google.cloud import vision
            google_client = vision.ImageAnnotatorClient()
            print("Google Vision initialized")
        except Exception as e:
            print(f"Google Vision initialization failed: {e}")
            models_available['google_vision'] = False

    # DocTR
    if models_available['doctr']:
        try:
            from doctr.models import ocr_predictor
            doctr_model = ocr_predictor(pretrained=True)
            print("DocTR initialized")
        except Exception as e:
            print(f"DocTR initialization failed: {e}")
            models_available['doctr'] = False

    # Surya
    if models_available['surya']:
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            surya_recognition_predictor = RecognitionPredictor()
            surya_detection_predictor = DetectionPredictor()
            print("Surya initialized")
        except Exception as e:
            print(f"Surya initialization failed: {e}")
            models_available['surya'] = False

def get_french_samples(dataset_path, num_samples):
    samples = []
    
    if not os.path.exists(dataset_path):
        print(f"French dataset path not found: {dataset_path}")
        return samples
    
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    valid_pairs = []
    
    for img_file in image_files:
        base_name = img_file.replace('_default.jpg', '')
        xml_file = None
        
        possible_xml = [
            f"{base_name}_default.xml",
            f"{base_name.replace('_', 'g')}_default.xml"
        ]
        
        for xml_name in possible_xml:
            if os.path.exists(os.path.join(dataset_path, xml_name)):
                xml_file = xml_name
                break
        
        if xml_file:
            valid_pairs.append((img_file, xml_file))
    
    selected_pairs = random.sample(valid_pairs, min(num_samples, len(valid_pairs)))
    
    for img_file, xml_file in selected_pairs:
        samples.append({
            'dataset': 'French',
            'image_path': os.path.join(dataset_path, img_file),
            'annotation_path': os.path.join(dataset_path, xml_file),
            'image_name': img_file,
            'annotation_name': xml_file
        })
    
    return samples

def get_english_samples(dataset_path, num_samples):
    samples = []
    
    if not os.path.exists(dataset_path):
        print(f"English dataset path not found: {dataset_path}")
        return samples
    
    images_path = os.path.join(dataset_path, 'images')
    annotations_path = os.path.join(dataset_path, 'annotations')
    
    if not os.path.exists(images_path) or not os.path.exists(annotations_path):
        print(f"English dataset structure not found in: {dataset_path}")
        return samples
    
    image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in selected_files:
        base_name = img_file.replace('.png', '')
        json_file = f"{base_name}.json"
        
        if os.path.exists(os.path.join(annotations_path, json_file)):
            samples.append({
                'dataset': 'English',
                'image_path': os.path.join(images_path, img_file),
                'annotation_path': os.path.join(annotations_path, json_file),
                'image_name': img_file,
                'annotation_name': json_file
            })
    
    return samples

def extract_text_from_french_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        texts = []
        for string_elem in root.findall('.//{http://www.loc.gov/standards/alto/ns-v4#}String'):
            content = string_elem.get('CONTENT')
            if content:
                texts.append(content)
        
        return ' '.join(texts)
    except Exception as e:
        print(f"Error extracting text from {xml_path}: {e}")
        return ""

def extract_text_from_english_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        for form_item in data.get('form', []):
            text = form_item.get('text', '').strip()
            if text:
                texts.append(text)
        
        return ' '.join(texts)
    except Exception as e:
        print(f"Error extracting text from {json_path}: {e}")
        return ""

def extract_boxes_from_french_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        for string_elem in root.findall('.//{http://www.loc.gov/standards/alto/ns-v4#}String'):
            content = string_elem.get('CONTENT')
            hpos = string_elem.get('HPOS')
            vpos = string_elem.get('VPOS')
            width = string_elem.get('WIDTH')
            height = string_elem.get('HEIGHT')
            
            if all([content, hpos, vpos, width, height]):
                x1 = float(hpos)
                y1 = float(vpos)
                x2 = x1 + float(width)
                y2 = y1 + float(height)
                boxes.append([x1, y1, x2, y2, content])
        
        return boxes
    except Exception as e:
        print(f"Error extracting boxes from {xml_path}: {e}")
        return []

def extract_boxes_from_english_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        boxes = []
        for form_item in data.get('form', []):
            text = form_item.get('text', '').strip()
            box = form_item.get('box', [])
            
            if text and len(box) == 4:
                boxes.append([box[0], box[1], box[2], box[3], text])
        
        return boxes
    except Exception as e:
        print(f"Error extracting boxes from {json_path}: {e}")
        return []

# OCR Functions
def tesseract_ocr(image_path):
    try:
        import pytesseract
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Tesseract error on {image_path}: {e}")
        return ""

def easyocr_ocr(image_path):
    global easyocr_reader
    try:
        if not easyocr_reader:
            print("EasyOCR not initialized.")
            return ""
        results = easyocr_reader.readtext(image_path)
        text = ' '.join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        print(f"EasyOCR error on {image_path}: {e}")
        return ""

def surya_ocr(image_path):
    global surya_recognition_predictor, surya_detection_predictor
    try:
        if not surya_recognition_predictor or not surya_detection_predictor:
            print("Surya not initialized")
            return ""

        image = Image.open(image_path)
        predictions = surya_recognition_predictor([image], det_predictor=surya_detection_predictor)

        text_parts = []
        for page in predictions:
            for line in page.text_lines:
                text_parts.append(line.text)

        return ' '.join(text_parts)
    except Exception as e:
        print(f"Surya error on {image_path}: {e}")
        return ""

def doctr_ocr(image_path):
    """Run DocTR OCR on image"""
    global doctr_model
    try:
        if not doctr_model:
            print("DocTR not initialized")
            return ""

        from doctr.io import DocumentFile
        doc = DocumentFile.from_images(image_path)
        result = doctr_model(doc)

        text_parts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text_parts.append(word.value)

        return ' '.join(text_parts)
    except Exception as e:
        print(f"DocTR error on {image_path}: {e}")
        return ""

def google_vision_ocr(image_path):
    """Run Google Vision OCR on image"""
    global google_client
    try:
        if not google_client:
            print("Google Vision not initialized")
            return ""

        from google.cloud import vision
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = google_client.text_detection(image=image)

        if response.text_annotations:
            return response.text_annotations[0].description.strip()
        return ""
    except Exception as e:
        print(f"Google Vision error on {image_path}: {e}")
        return ""

# Evaluation Functions
def clean_text(text):
    """Clean text for comparison"""
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    return SequenceMatcher(None, clean1, clean2).ratio()

def calculate_word_accuracy(predicted, ground_truth):
    """Calculate word-level precision, recall, and F1"""
    pred_words = set(clean_text(predicted).split())
    gt_words = set(clean_text(ground_truth).split())

    if not gt_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if not pred_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    intersection = pred_words.intersection(gt_words)
    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(gt_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_ocr_result(predicted, ground_truth):
    """Evaluate OCR result with multiple metrics"""
    similarity = calculate_similarity(predicted, ground_truth)
    word_metrics = calculate_word_accuracy(predicted, ground_truth)

    return {
        'similarity': similarity,
        'precision': word_metrics['precision'],
        'recall': word_metrics['recall'],
        'f1': word_metrics['f1'],
        'predicted_length': len(predicted),
        'ground_truth_length': len(ground_truth)
    }

def run_single_ocr_evaluation(samples, ocr_model_name):
    """Run OCR evaluation for a single model"""
    results = []

    ocr_functions = {
        'tesseract': tesseract_ocr,
        'easyocr': easyocr_ocr,
        'doctr': doctr_ocr,
        'surya': surya_ocr,
        'google_vision': google_vision_ocr
    }

    if ocr_model_name not in ocr_functions:
        print(f"Unknown OCR model: {ocr_model_name}")
        return results

    if not models_available.get(ocr_model_name, False):
        print(f"Model {ocr_model_name} not available")
        return results

    ocr_function = ocr_functions[ocr_model_name]

    print(f"\nRunning {ocr_model_name} evaluation...")
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: {sample['image_name']}")

        start_time = time.time()
        predicted_text = ocr_function(sample['image_path'])
        processing_time = time.time() - start_time

        metrics = evaluate_ocr_result(predicted_text, sample['ground_truth'])

        result = {
            'sample_id': i,
            'dataset': sample['dataset'],
            'image_name': sample['image_name'],
            'ground_truth': sample['ground_truth'],
            'ocr_result': {
                'model': ocr_model_name,
                'predicted_text': predicted_text,
                'processing_time': processing_time,
                'metrics': metrics
            }
        }

        results.append(result)

    return results

def calculate_average_metrics(results_dict):
    """Calculate average metrics for all models"""
    summary = []

    for model_name, samples in results_dict.items():
        if not samples:
            continue

        print(f"Processing {model_name}...")

        metrics_lists = {
            'similarity': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'processing_time': []
        }

        for sample in samples:
            metrics = sample['ocr_result']['metrics']
            processing_time = sample['ocr_result']['processing_time']

            metrics_lists['similarity'].append(metrics['similarity'])
            metrics_lists['precision'].append(metrics['precision'])
            metrics_lists['recall'].append(metrics['recall'])
            metrics_lists['f1'].append(metrics['f1'])
            metrics_lists['processing_time'].append(processing_time)

        # Calculate averages
        result = {'model': model_name, 'sample_count': len(samples)}

        for metric_name, values in metrics_lists.items():
            if values:
                result[f'avg_{metric_name}'] = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - result[f'avg_{metric_name}']) ** 2 for x in values) / len(values)
                    result[f'std_{metric_name}'] = variance ** 0.5
                else:
                    result[f'std_{metric_name}'] = 0.0

        summary.append(result)

    return pd.DataFrame(summary).round(4)

def display_results(df):
    """Display comprehensive results"""
    print("\n" + "="*80)
    print("OCR MODELS PERFORMANCE SUMMARY")
    print("="*80)

    # Sort by F1 score
    df_sorted = df.sort_values('avg_f1', ascending=False)

    # Display main metrics
    display_cols = ['model', 'sample_count', 'avg_similarity', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_processing_time']
    available_cols = [col for col in display_cols if col in df_sorted.columns]

    print("\nOVERALL PERFORMANCE (sorted by F1 score):")
    print("-" * 60)
    print(df_sorted[available_cols].to_string(index=False, float_format='%.4f'))


def save_results(all_results, output_file="ocr_evaluation_results.json"):
    """Save all results to JSON file"""
    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    clean_results = convert_numpy_types(all_results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

def main():
    """Main execution function"""
    print("="*80)
    print("COMPLETE OCR EVALUATION PIPELINE")
    print("="*80)

    # Check libraries
    print("\nChecking available libraries...")
    check_and_import_libraries()

    # Initialize models
    initialize_ocr_models()

    # Load datasets
    print(f"\nLoading datasets...")
    print(f"French dataset path: {FRENCH_DATASET_PATH}")
    print(f"English dataset path: {ENGLISH_DATASET_PATH}")
    print(f"Samples per dataset: {SAMPLES_PER_DATASET}")

    french_samples = get_french_samples(FRENCH_DATASET_PATH, SAMPLES_PER_DATASET)
    english_samples = get_english_samples(ENGLISH_DATASET_PATH, SAMPLES_PER_DATASET)

    all_samples = french_samples + english_samples

    print(f"\nDataset loaded:")
    print(f"  French samples: {len(french_samples)}")
    print(f"  English samples: {len(english_samples)}")
    print(f"  Total samples: {len(all_samples)}")

    if not all_samples:
        print("No samples found! Please check dataset paths.")
        return

    # Extract ground truth
    print("\nExtracting ground truth...")
    for sample in all_samples:
        if sample['dataset'] == 'French':
            sample['ground_truth'] = extract_text_from_french_xml(sample['annotation_path'])
            sample['ground_truth_boxes'] = extract_boxes_from_french_xml(sample['annotation_path'])
        else:
            sample['ground_truth'] = extract_text_from_english_json(sample['annotation_path'])
            sample['ground_truth_boxes'] = extract_boxes_from_english_json(sample['annotation_path'])

    # Run evaluations
    all_results = {}
    available_models = [model for model, available in models_available.items() if available]

    print(f"\nRunning evaluations for {len(available_models)} models...")

    for model_name in available_models:
        try:
            results = run_single_ocr_evaluation(all_samples, model_name)
            if results:
                all_results[model_name] = results
                print(f" {model_name} completed: {len(results)} samples processed")
            else:
                print(f"{model_name} failed: no results")
        except Exception as e:
            print(f"{model_name} failed with error: {e}")

    if not all_results:
        print(" No successful evaluations! Please check model configurations.")
        return

    # Calculate and display results
    print(f"\nCalculating average metrics...")
    df = calculate_average_metrics(all_results)

    # Display results
    display_results(df)


    print(f"\nEvaluation completed successfully!")
    print(f"   Models evaluated: {len(all_results)}")
    print(f"   Total samples: {len(all_samples)}")
    print(f"   Results saved to: ocr_evaluation_results.json")

if __name__ == "__main__":
    main()
