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

        self._corriendo = True
        while self._corriendo:
            ret, frame = cap.read()

            if not ret:
                self._corriendo = False
                break

            if self._yolo_activo and self.modelo is not None:
                results = self.modelo(frame)
                self.detecciones.emit(YoloDetection.parseToDictionary(None,results))
                frame = results[0].plot()
                print(self.modelo.device)

            self.frameReady.emit(frame)
            time.sleep(delay)

        cap.release()

    def stop(self):
        self._corriendo = False
        self.wait()

    def activar_yolo(self):
        if self.modelo is None:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.modelo = YOLO(os.path.join(BASE_DIR,'..',"models","yolov8n.pt"))
        self._yolo_activo = True

    def desactivar_yolo(self):
        self._yolo_activo = False

    