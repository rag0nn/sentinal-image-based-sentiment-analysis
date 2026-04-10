from typing import List, Tuple, Union

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import logging
from .structs import EMOTION_DICT_TR, EMOTION_DICT, NUM_EMOTIONS, ModelTypes

# Yazı parametreleri
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONST_SCALE_WRATIO = 350
FONT_THICKNESS_WRATIO = 350
TEXT_COLOR = (255,255,255) 
BG_COLOR = (0, 255, 0)     

class SentimentClassifier:
    
    def __init__(self, 
                 model_type:ModelTypes, 
                 model_path:str,
                 gray_prediction,
                 device=None):
        """
        Duygu sınıflandırması yapan model
        Args:
            model_type (ModelTypes) : One of the [ModelTypes] model types,
            model_path (str, Optional) : Model '.pth' path. 
            gray_predction (bool) : Prediction scale
            device (str, Optional): cpu, cude etc.
        """
        self.model_type = model_type
        self.model_path = model_path
        self.gray_prediction = gray_prediction
        # dönüşümler
        if gray_prediction:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        self.model, self.device = self._load_model(device) 
    
    def _load_model(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")
            
        if self.model_type == ModelTypes.Resnet101:
            model = models.resnet101(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_EMOTIONS)
        elif self.model_type == ModelTypes.Resnet50:
            model = models.resnet50(pretrained=False)  
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_EMOTIONS)
        elif self.model_type == ModelTypes.MobileSmall:
            model = models.mobilenet_v3_small(pretrained=False)
            # Son classifier katmanını değiştir
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, NUM_EMOTIONS)
        else:
            raise KeyError(f"Invalid input for model_type = {self.model_type}")
        
        # Direkt state_dict yükle
        state_dict = torch.load(self.model_path, map_location=device)
        
        # Eğer _orig_mod prefix varsa temizle
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                name = k[len("_orig_mod."):]
            else:
                name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model, device

    def predict(self, images: Union[List[np.ndarray],np.ndarray], verbose=False) -> List[Tuple]:
        """
        Predict sentiment for a single image or a list of images.

        Args:
            images (np.ndarray or list[np.ndarray]): Single image or a list of images.
            verbose (bool, optional): If True, prints prediction info. Default is False.

        Returns:
            list[Tuple[int, float, torch.Tensor]]: 
                Each tuple contains (predicted_class, confidence, probabilities) for each image.
        """
        # type check
        if isinstance(images,np.ndarray):
            images = [images]
        elif isinstance(images, tuple):
            images = list(images)
        elif not isinstance(images, list):
            raise TypeError("images must be np.ndarray or list/tuple of np.ndarray")
            
        for img in images:
            if not isinstance(img, np.ndarray):
                raise TypeError("All items in images must be np.ndarray")
            
        # batch & prediction    
        pil_images = [Image.fromarray(img) for img in images]
        
        tensors = [self.transform(img) for img in pil_images]
        input_tensor = torch.stack(tensors).to(self.device)  # (B, C, H, W)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)

            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = probabilities.max(dim=1).values

        results = []
        for i in range(len(images)):
            pred = predicted_classes[i].item()
            conf = confidences[i].item()
            probs = probabilities[i]

            if verbose:
                logging.info(f"[{i}] Prediction: {pred}, conf: {conf}")

            results.append((pred, conf, probs))

        return results
    
    def visualize(self, image: np.ndarray, predicted_class: int, confidence: float, lang: str = "tr") -> np.ndarray:
        """
        Görsele tahmini etiket ve olasılığı ekler.
        
        Args:
            image (np.ndarray): BGR formatında resim
            predicted_class (int): Tahmin edilen sınıf indeksi
            confidence (float): Tahmin olasılığı (0-1)
            lang (str): Label dili ("tr" veya "en")
        
        Returns:
            annotated_image (np.ndarray): Annotated image
        """
        annotated_image = image.copy()
        h,w,_ = image.shape
        # Label seçimi
        if lang == "tr":
            label_text = EMOTION_DICT_TR.get(predicted_class, "Unknown")
        else:
            label_text = EMOTION_DICT.get(predicted_class, "Unknown")
        
        # Label + confidence
        text = f"{predicted_class}:{label_text}: {confidence*100:.1f}%"
        
        # Text boyutunu al
        

        fontScale = w / FONST_SCALE_WRATIO
        thickness = int(w / FONT_THICKNESS_WRATIO)
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, fontScale, thickness)
        
        # Text kutusu koordinatları
        x, y = 10, text_height + 10  # sol üst köşe
        cv2.rectangle(
            annotated_image,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            BG_COLOR,
            thickness=-1  # dolu kutu
        )
        
        # Text'i çiz
        cv2.putText(
            annotated_image,
            text,
            (x, y),
            FONT,
            fontScale,
            TEXT_COLOR,
            thickness,
            lineType=cv2.LINE_AA
        )
        
        return annotated_image