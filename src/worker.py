from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from ultralytics import YOLO
import cv2
import os
import time
from src.detection.yolo_detection import YoloDetection


class Worker(QThread):
    frameReady = pyqtSignal(np.ndarray)
    detecciones = pyqtSignal(list)
    finVideo = pyqtSignal()
    videoInfo = pyqtSignal(float)   # fps de la fuente, para datar los eventos

    def __init__(self, fuente=0):
        super().__init__()
        self._corriendo = False
        self._yolo_activo = False
        self.modelo = None
        self.fuente = fuente
        
        
    def run(self):
        cap = cv2.VideoCapture(self.fuente)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 1.0 / 30
        self.videoInfo.emit(fps if fps > 0 else 30.0)

        self._corriendo = True
        while self._corriendo:
            ret, frame = cap.read()

            if not ret:
                self._corriendo = False
                break

            if self._yolo_activo and self.modelo is not None:
                # track() en vez de __call__: asigna un id persistente a cada vehiculo.
                # persist=True mantiene el estado del tracker entre frames.
                results = self.modelo.track(frame, persist=True, tracker="bytetrack.yaml",
                                            iou=YoloDetection.NMS_IOU,
                                            agnostic_nms=YoloDetection.AGNOSTIC_NMS)
                self.detecciones.emit(YoloDetection.parseToDictionary(None,results))
                # En caso de que los jueces quieran ver los bboxes se descomentan estas dos lineasss
                #frame = results[0].plot()
                #print(self.modelo.device)

            self.frameReady.emit(frame)
            time.sleep(delay)

        cap.release()
        self.finVideo.emit()

    def stop(self):
        self._corriendo = False

    def activar_yolo(self):
        if self.modelo is None:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.modelo = YOLO(os.path.join(BASE_DIR,'..',"models","yolov8n.pt"))
        self._yolo_activo = True

    def desactivar_yolo(self):
        self._yolo_activo = False

    