"""
Face Recognition ve sentiment modellerini
config.py'deki verilerin path'lerini kullanarak 
ve MODEL_PATH, TEST_TYPE seçeneklerini kullanarak test eder.
"""

from test_config import loader, TestTypes
from sentinal.sentiment_model.structs import EMOTION_DICT, EMOTION_DICT_TR
from sentinal.utils import timer
import cv2
import rerun as rr
import rerun.blueprint as rrb
from sentinal.detector import Sentinal, Models, Stabilizer
import logging
import numpy as np

TEST_TYPE = TestTypes.VIDEO
WINDOW_H = 700

# LOGGERS
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

def apply_with_rerun(test_type: TestTypes):
    detector = Sentinal(Models.MobileSmall)
    # SDK başlatılıyor ve Viewer spawn ediliyor
    rr.init("sentiment_detection", spawn=True)
    
    # 🔵 2D layout blueprint
    rr.send_blueprint(
        rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="image/annotated"),
                    rrb.TimeSeriesView(origin="metrics/classes"),
                    rrb.TimeSeriesView(origin="metrics/confidence"),
                ),
        )
    )

    frame_time = 0.0  # simülasyon zamanı
    dt = 1 / 60.0     # frame süresi

    for real_label, frame_np in loader(test_type):
        # Zamanı set et
        rr.set_time("sim_time", sequence=int(frame_time))

        # Detect
        predictions = detector.detect(frame_np)
        annotated = timer(detector.visualize)(frame_np.copy(),predictions,EMOTION_DICT_TR)
        
        # Görselleri logla
        rr.log("image/annotated", rr.Image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)))

        # Gerçek sınıf
        rr.log("metrics/classes/real", rr.Scalars(real_label))

        # Tahminler ve confidence log
        for i, pred in enumerate(predictions):
            print(f"Real: {real_label}, Predicted: {pred.pred_lbl}, Conf: {pred.conf:.2f}")
            rr.log(f"metrics/classes/predicted/{i}", rr.Scalars(pred.pred_lbl))
            rr.log(f"metrics/confidence/{i}", rr.Scalars(pred.conf))

        frame_time += dt


def apply_with_cv2(test_type):
    detector = Sentinal(Models.MobileSmall)
    frame_time = 0
    old_one = None
    stabilizer = Stabilizer(8)
    for real_label, frame_np in loader(test_type):
        # Multiple paralelled inference handle
        if old_one is None:
            old_one = frame_np
            continue
        
        # detect
        # results = detector.detect([frame_np])
        results = detector.detect([frame_np, old_one])
        anoos = []
        for i, result in enumerate(results):
            if i == 0: # stabilization
                stable_class, stable_conf = stabilizer.update(result[0])
            annotated = timer(detector.visualize)(frame_np.copy(),result,EMOTION_DICT_TR)

            # cv2.putText(annotated,f"{i}",(50,100),cv2.FONT_HERSHEY_DUPLEX,2.9,(0,255,0),2)
            anoos.append(annotated)
            
        # Tahminler ve confidence log
        for i, result in enumerate(results):
            for pred in result:
                print(f"[{i}] Real: {real_label}, Predicted: {pred.pred_lbl}, Conf: {pred.conf:.2f}") 
                
        # output show resize      
        h,w,_ = frame_np.shape
        ratio = w/h
        new_w = int(ratio * WINDOW_H)
        anoos = [cv2.resize(anno, (new_w,WINDOW_H)) for anno in anoos]
            
        annomerged = np.hstack(anoos)
        
        # göster
        cv2.imshow("Annotated", annomerged)            
        if test_type == TestTypes.IMAGESEQ:
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            else:
                pass
        else:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        frame_time += 1
        
        old_one = frame_np


if __name__ == "__main__":
    apply_with_cv2(TEST_TYPE)
    # apply_with_rerun(TEST_TYPE)