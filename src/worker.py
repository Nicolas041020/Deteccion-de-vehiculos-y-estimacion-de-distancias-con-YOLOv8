from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from ultralytics import YOLO
import cv2
import os

class Worker(QThread):
    frameReady = pyqtSignal(np.ndarray)

    def __init__(self, fuente=0):
        super().__init__()
        self._corriendo = False
        self._yolo_activo = False
        self.modelo = None
        self.fuente = fuente

    def run(self):
        cap = cv2.VideoCapture(self.fuente)
        self._corriendo = True
        while self._corriendo:
            ret, frame = cap.read()

            if not ret:
                break

            if self._yolo_activo and self.modelo is not None:
                results = self.modelo(frame)
                frame = results[0].plot()

            self.frameReady.emit(frame)

        cap.release()

    def stop(self):
        self._corriendo = False
        self.wait()

    def activar_yolo(self):
        if self.modelo is None:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.modelo = YOLO(os.path.join(BASE_DIR,"models","weights","yolov8n.pt"))
        self._yolo_activo = True

    def desactivar_yolo(self):
        self._yolo_activo = False

    