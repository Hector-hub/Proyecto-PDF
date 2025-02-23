import os
import json
import fitz
import pytesseract
import torch
import clip
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
# from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import torchvision.transforms as transforms
from openai import OpenAI
import tiktoken

# OpenAI configuration
client = OpenAI()

# Global configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_embedding(text, model="text-embedding-3-large", chunk_size=8000):
    """Generate embeddings for the given text using OpenAI's API.
    
    Args:
        text (str): The text to embed.
        model (str): The model to use for embedding.
        chunk_size (int): The size of text chunks to process.
    
    Returns:
        list: A list of embeddings for the text chunks.
    """
    text = text.replace("\n", " ")
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    embeddings = []

    for i in range(0, len(tokens), chunk_size):
        chunk = encoder.decode(tokens[i:i + chunk_size])
        embedding = client.embeddings.create(
            model=model,
            input=chunk,
            encoding_format="float"
        )
        embeddings.append(embedding.data[0].embedding)
        
    return embeddings

def check_list_type(embedding):
    """Ensure the embedding is a flat list.
    
    If the embedding is a list of lists, return the first list.
    
    Args:
        embedding (list): The embedding to check.
    
    Returns:
        list: A flat list embedding.
    """
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        return embedding[0]
    return embedding

# Load base models
# text_model = SentenceTransformer("all-mpnet-base-v2")  # Improved model (commented out, possibly for future use)
clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
yolo_model = YOLO("yolov8x.pt")

# Directory configuration
BASE_DIR = "/Users/hecrey/Desktop/PDF_NLP_Project/process_pdf"
INPUT_PDFS = "/Users/hecrey/Desktop/PDF_NLP_Project/input_pdfs"

# Specialized prompts for AIP
AERO_PROMPTS = [
    "Aeronautical navigation chart with flight paths",
    "Air traffic control radar display",
    "Airport layout diagram with runways",
    "Meteorological aviation weather map",
    "Instrument approach procedure chart",
    "Airspace classification diagram",
    "Radio navigation aid schematic",
    "Aircraft technical specification diagram",
    "Aviation coordinate grid system",
    "Flight procedure turn diagram"
]

# Configuration of specialized models
SPECIALIZED_MODELS = {
    "chart": {
        "processor": "google/vit-large-patch16-384",
        "model": "google/vit-large-patch16-384"
    },
    "radar": {
        "processor": "facebook/convnext-base-224",
        "model": "facebook/convnext-base-224-22k"
    },
    "airport": {
        "processor": "microsoft/swin-base-patch4-window7-224",
        "model": "microsoft/swin-base-patch4-window7-224-in22k"
    }
}

class HybridClassifier:
    """A hybrid image classifier that uses CLIP for initial classification and specialized models for detailed classification if necessary."""
    
    def __init__(self):
        """Initialize the classifier with empty model dictionaries."""
        self.specialized_models = {}
        self.loaded_models = set()
        
    def load_model(self, model_type):
        """Load a specialized model if it hasn't been loaded yet.
        
        Args:
            model_type (str): The type of model to load.
        
        Returns:
            dict: A dictionary containing the processor and model.
        """
        if model_type not in self.specialized_models:
            if model_type not in SPECIALIZED_MODELS:
                raise ValueError(f"Model type {model_type} not supported")
            
            processor = AutoImageProcessor.from_pretrained(SPECIALIZED_MODELS[model_type]["processor"])
            model = AutoModelForImageClassification.from_pretrained(SPECIALIZED_MODELS[model_type]["model"]).to(device)
            
            self.specialized_models[model_type] = {
                "processor": processor,
                "model": model
            }
            self.loaded_models.add(model_type)
            
        return self.specialized_models[model_type]

    def classify(self, img_path):
        """Classify an image using a hybrid approach.
        
        First uses CLIP for initial classification, then determines if specialized classification is needed.
        
        Args:
            img_path (str): Path to the image to classify.
        
        Returns:
            dict: Classification results, including base and possibly detailed classifications.
        """
        base_result = self._clip_classification(img_path)
        model_type = self._needs_specialized(base_result["class"])
        
        if model_type:
            detailed_result = self._specialized_classification(img_path, model_type)
            return {
                "base_classification": base_result,
                "detailed_classification": detailed_result
            }
            
        return {"base_classification": base_result}

    def _clip_classification(self, img_path):
        """Perform initial classification using the CLIP model.
        
        Args:
            img_path (str): Path to the image to classify.
        
        Returns:
            dict: Results including class, confidence, and all probabilities.
        """
        image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(clip.tokenize(AERO_PROMPTS).to(device))
            
        probs = (image_features @ text_features.T).softmax(dim=-1)
        top_idx = probs.argmax().item()
        
        return {
            "class": AERO_PROMPTS[top_idx],
            "confidence": round(probs[0][top_idx].item(), 4),
            "all_probabilities": {AERO_PROMPTS[i]: round(p.item(), 4) for i, p in enumerate(probs[0])}
        }

    def _needs_specialized(self, class_name):
        """Determine if specialized classification is needed based on the initial classification.
        
        Args:
            class_name (str): The class name from initial classification.
        
        Returns:
            str or None: The model type for specialized classification, or None if not needed.
        """
        class_name = class_name.lower()
        if 'chart' in class_name or 'map' in class_name:
            return 'chart'
        if 'radar' in class_name:
            return 'radar'
        if 'airport' in class_name or 'runway' in class_name:
            return 'airport'
        return None

    def _specialized_classification(self, img_path, model_type):
        """Perform specialized classification using a pre-trained model.
        
        Args:
            img_path (str): Path to the image to classify.
            model_type (str): The type of specialized model to use.
        
        Returns:
            dict: Detailed classification results.
        """
        model_info = self.load_model(model_type)
        image = Image.open(img_path).convert("RGB")
        
        inputs = model_info["processor"](
            images=image, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model_info["model"](**inputs)
            
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_idx = logits.argmax().item()
        
        return {
            "model_type": model_type,
            "class": model_info["model"].config.id2label[top_idx],
            "confidence": round(probabilities[top_idx].item(), 4),
            "model_used": SPECIALIZED_MODELS[model_type]["model"]
        }

hybrid_classifier = HybridClassifier()

def _format_detections(results):
    """Format detection results from YOLO model.
    
    Args:
        results: Detection results from YOLO.
    
    Returns:
        list: Formatted list of detected objects with confidence and bounding box.
    """
    formatted = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            formatted.append({
                "object": yolo_model.names[cls_id],
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })
    return formatted

def process_pdf(pdf_path):
    """Process a complete PDF and generate global embeddings.
    
    Args:
        pdf_path (str): Path to the PDF file to process.
    
    Returns:
        tuple: Paths to the main JSON and embedding JSON files.
    """
    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    
    # Set up output directories
    images_dir = os.path.join(BASE_DIR, "extracted_images")
    embeddings_dir = os.path.join(BASE_DIR, "embeddings")
    pdf_processed_dir = os.path.join(BASE_DIR, "pdf_processed")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(pdf_processed_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)  

    # Main data structure
    pdf_data = {
        "metadata": {
            "document_name": pdf_name,
            "total_pages": len(doc),
            "document_type": "AIP",
            "processing_stack": [
                "CLIP-ViT-L/14@336px",
                "YOLOv8x",
                "HybridClassifier",
                "all-mpnet-base-v2"
            ]
        },
        "content": []
    }

    full_text = []

    # Process each page
    for page_num, page in enumerate(doc):
        page_data = {
            "page_number": page_num + 1,
            "text": "",
            "text_embedding": [],
            "graphical_elements": []
        }

        # Extract and process text
        page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
        if page_text.strip():
            clean_text = ' '.join(page_text.replace('\ufffd', '').split())
            page_data["text"] = clean_text
            page_data["text_embedding"] = check_list_type(get_embedding(clean_text))
            full_text.append(clean_text)

        # Process images
        # print(page.get_images(full=True))  # Commented out debugging print
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                img_filename = f"{pdf_name}_p{page_num}_i{img_index}.png"
                img_path = os.path.join(images_dir, img_filename)

                if not os.path.exists(img_path):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    with open(img_path, "wb") as f:
                        f.write(base_image["image"])
                else:
                    # print(f"Imagen {img_filename} ya existe, omitiendo creación.")  # Commented out debugging print
                    pass
                    
                # OCR optimized for technical documents
                ocr_text = pytesseract.image_to_string(
                    Image.open(img_path),
                    config='--psm 6 -l eng+spa+fra --oem 3'
                )

                classification = hybrid_classifier.classify(img_path)
                
                # Object detection with filtering for specific classes
                detections = yolo_model(img_path, classes=[4, 8, 9])  # Filter: airplanes, ships, traffic lights
                # print("graphical_elements inits")  # Commented out debugging print
                # print(classification['base_classification'])  # Commented out debugging print
                ocr_text_clean_text = ' '.join(ocr_text.replace('\ufffd', '').split())
                full_text.append(
                    f"{ocr_text_clean_text} [Context: this image is classified as "
                    f"'{classification['base_classification']['class']}' with a "
                    f"{classification['base_classification']['confidence']*100}% match confidence.]"
                )
                page_data["graphical_elements"].append({
                    "file_reference": img_filename,
                    "ocr_content": ocr_text.strip(),
                    "classification": classification,
                    "detected_objects": _format_detections(detections)
                })
                # print("graphical_elements processs")  # Commented out debugging print
            except Exception as e:
                # TODO: Consider logging errors instead of printing for production
                print(f"Error en página {page_num}, imagen {img_index}: {str(e)}")

        pdf_data["content"].append(page_data)

    # Generate global document embedding
    full_text_str = "\n".join(full_text)
    get_embeddings = get_embedding(full_text_str)
    pdf_embedding = check_list_type(get_embeddings)
    
    # Save global embedding
    embedding_data = {
        "document_name": pdf_name,
        "embedding": pdf_embedding,
        "pages": len(doc),
        "content_length": len(full_text_str),
        "content_hash": hash(full_text_str)
    }
    
    embedding_path = os.path.join(embeddings_dir, f"{pdf_name}.json")
    with open(embedding_path, "w", encoding="utf-8") as f:
        json.dump(embedding_data, f, indent=2, ensure_ascii=False)

    # Save complete data
    json_path = os.path.join(pdf_processed_dir, f"{pdf_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pdf_data, f, indent=2, ensure_ascii=False)

    return json_path, embedding_path

def process_all_pdfs():
    """Process all PDF files in the input directory.
    
    Returns:
        None
    """
    pdf_files = [f for f in os.listdir(INPUT_PDFS) if f.lower().endswith(".pdf")]
    
    for pdf_file in pdf_files:
        input_path = os.path.join(INPUT_PDFS, pdf_file)
        try:
            main_json, embedding_json = process_pdf(input_path)
            print(f"✅ {pdf_file} procesado:")
            print(f"   - JSON principal: {os.path.basename(main_json)}")
            print(f"   - Embedding global: {os.path.basename(embedding_json)}")
        except Exception as e:
            # TODO: Consider logging errors instead of printing for production
            print(f"❌ Error procesando {pdf_file}: {str(e)}")

if __name__ == "__main__":
    print("=== Inicio de procesamiento ===")
    print(f"PDFs a procesar: {len(os.listdir(INPUT_PDFS))} archivos")
    process_all_pdfs()
    print("=== Proceso completado ===")