import numpy as np
from .utils import timer, Colors
from .face_recognition.detect import FaceDetector
from .sentiment_model.detect import SentimentClassifier
from .sentiment_model.structs import ModelTypes, Models
from typing import Dict, List, Tuple
import logging
import gdown
import os
import cv2
import time


CHOSEN_MODEL = Models.MobileSmall
GRAYSCALEPREDICTION = True

class Prediction:
    
    def __init__(self,
        x:int,
        y:int,
        w:int,
        h:int,
        conf:float,
        pred_lbl:int,
        probabilities:List=None,
        stabilized_conf:float=None,
        stabilized_label:int=None,
        real_lbl:int=None):
        
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.pred_lbl = pred_lbl
        self.real_lbl = real_lbl
        self.probabilites = probabilities
        self.stabilized_conf = stabilized_conf
        self.stabilized_label = stabilized_label
        
        
    def __repr__(self):
        return f"({self.x},{self.y},{self.w},{self.h}) {self.conf} {self.pred_lbl} {self.real_lbl}"
  
  



class Stabilizer:
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.2, # Ema alpha kadar yeni tahmine 1-alpha kadar bir önceki tahmine ağırlık verir.
        change_threshold: float = 0.7, # YEni bir sınıfa geçmek için minimum güven eşiği, yani ema değeri bu değerden büyük oldudğudna yeni bir sınıfa geçilebilir
        stay_threshold: float = 0.4, # YEni gelen label bir öncekiyle aynı olmasına rağmen confidince çok küçükse bile tahmin değiştirmeden beklemye yarar.
        cooldown: float = 2.0, # Bir label değişiminden en az 2 saniye bekelmeden başka label'a geçmeyecek
    ):
        self.num_classes = num_classes
        self.alpha = alpha
        self.change_threshold = change_threshold
        self.stay_threshold = stay_threshold
        self.cooldown = cooldown

        self.ema = None
        self.last_label = None
        self.last_change_time = 0.0

    def update(self, probs):
        """
        probabilities: torch tensor veya numpy array (shape: [num_classes])
        """
        # --- EMA ---
        if self.ema is None:
            self.ema = probs
        else:
            self.ema = self.alpha * probs + (1 - self.alpha) * self.ema

        # --- argmax ---
        new_label = int(np.argmax(self.ema))
        new_conf = float(self.ema[new_label])

        now = time.time()

        # first init
        if self.last_label is None:
            self.last_label = new_label
            self.last_change_time = now
            return self.last_label, new_conf

        # hysteresis + cooldown
        if new_label != self.last_label:
            if (
                new_conf > self.change_threshold
                and (now - self.last_change_time) > self.cooldown
            ):
                self.last_label = new_label
                self.last_change_time = now
        else:
            # aynı label → stay threshold kontrolü
            if new_conf < self.stay_threshold:
                # confidence çok düştüyse bile hemen değiştirme
                pass

        return self.last_label, float(self.ema[self.last_label])
          
class Sentinal:
    
    def __init__(self, sentiment_model:Models = None,
                 device = None,
                 grayscale_prediction = True,
                 use_stabilizer = True,
                 ):        
        if sentiment_model is not None:
            global CHOSEN_MODEL
            global GRAYSCALEPREDICTION
            CHOSEN_MODEL = sentiment_model
            GRAYSCALEPREDICTION = grayscale_prediction
            
        model_type, remote_path, local_path = CHOSEN_MODEL.value
        self._check_models(local_path,remote_path)
        
        self.grayscale_prediction = GRAYSCALEPREDICTION
        self.sentiment_model = SentimentClassifier(
            model_type, 
            local_path,
            self.grayscale_prediction,
            device)
        self.face_detector = FaceDetector()
        self.use_stabilizer = use_stabilizer
        if self.use_stabilizer:
            self.stabilizer = Stabilizer(8)
        logging.info(f"Graysacle Prediction: {self.grayscale_prediction}")
    
    def _check_models(self, local_path, remote_path):
        if os.path.exists(local_path):
            logging.info(f"Model file found: {local_path}")
        else:
            logging.info(f"Model file did'nt find, Downloading...: {remote_path}")
            try:
                gdown.download(remote_path, local_path, quiet=False)
            except Exception as e:
                raise Exception(f"{e} Error occured when downloading model file")
            logging.info("Model file downloaded successfully")


    @timer
    def detect(self, image: np.ndarray) -> List[Prediction]:
        """
        Verilen görüntüdei yüzleri bulur ve duygularını tahmin eder.
        Args:
            image: input image
        Returns:
            predictions: conjugated tuples of predictions like (label, conf)
            annotations: annotations of process, first one is face annotation others sentiment annotations
        """
        predictions = []
        
        # face recognition
        results = self.face_detector.detect_face(image)
        results = self.face_detector.add_margin(image,results)
        face_images = self.face_detector.crop_faces(image, results)

        # sentiment analysis
        for face, detection in zip(face_images, results.detections):
            pred, conf, probabilities = self.sentiment_model.predict(face)
            
            stable_class = None
            stable_conf = None
            probs = None
            if self.use_stabilizer:
                # probalities (for stabilization) tensor → numpy
                if hasattr(probabilities, "detach"):
                    probs = probabilities.detach().cpu().numpy()
                else:
                    probs = probabilities
                probs = probs.ravel()
                
                stable_class, stable_conf = self.stabilizer.update(probs)
                
            bbox = detection.bounding_box
            
            predictions.append(
                Prediction(
                    bbox.origin_x,bbox.origin_y,bbox.width,bbox.height,
                    conf,pred, 
                    probs,stable_conf,stable_class)
            )
            
        logging.info(f"Founded {len(predictions)} faces")
            
        return predictions
    
    def visualize(self, image:np.ndarray, predictions:List[Prediction], label_dict:Dict):
        output = image.copy()
        H,W,_ = output.shape
        for pred in predictions:
            label = pred.pred_lbl
            conf = pred.conf
            if pred.stabilized_conf is not None and pred.stabilized_label is not None:
                label = pred.stabilized_label
                conf = pred.stabilized_conf
                
            # Overlay kopyası
            overlay = output.copy()

            # Rectangle çiz (fill)
            cv2.rectangle(
                overlay,
                (pred.x, pred.y),
                (pred.x + pred.w, pred.y + pred.h),
                Colors.BLUE_LIGHT.value,
                -1
            )

            # ROI alpha blend
            alpha = 0.3
            output[pred.y:pred.y + pred.h, pred.x:pred.x + pred.w] = cv2.addWeighted(
                overlay[pred.y:pred.y + pred.h, pred.x:pred.x + pred.w], alpha,
                image[pred.y:pred.y + pred.h, pred.x:pred.x + pred.w], 1 - alpha,
                0
            )
            cv2.rectangle(
                        output, 
                        (pred.x,pred.y),
                        (pred.x+pred.w,pred.y+pred.h),
                        Colors.RED_PRIMARY.value,
                        int(min(H,W)/100))
            
            #label info
            label_rect_gap = int(min(H,W)/30)
            lbl = f"{conf:.2f} {label} {label_dict.get(label,'Uknown')}"
            font_scale = min(W,H)/500
            font_thickness = int(min(W,H)/200)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(lbl, font, font_scale, font_thickness)
            
            # label ractangle
            cv2.rectangle(
                output,
                (pred.x,pred.y),
                (pred.x+text_width+label_rect_gap, pred.y+text_height+label_rect_gap),
                Colors.RED_PRIMARY.value,
                -1
            )
            
            # label
            cv2.putText(
                output,
                lbl,
                (pred.x+int(label_rect_gap/2),pred.y+text_height+int(label_rect_gap/2)),
                font,
                font_scale,
                Colors.SNOWWHITE.value,
                font_thickness
                )
            
        return output
            
    def close(self):
        self.face_detector.close()
        
        
        
        
        
