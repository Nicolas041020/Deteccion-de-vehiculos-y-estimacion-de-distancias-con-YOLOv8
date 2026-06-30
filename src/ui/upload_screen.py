from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
import cv2
from src.detection.yolo_detection import YoloDetection
from src.worker import Worker
from src.distanceEstimation.Distance_Estimation import DistanceEstimation
import os
from src.detection.roi_filter import RoiFilter
class UploadScreen(QWidget):
    navegar = pyqtSignal(int)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE_DIR,'..','..','models','yolov8n.pt')
    yolo = YoloDetection(ruta)
     
    
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):

        #label video
        self.label_archivo = QLabel("Resultados")
        self.label_archivo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_archivo.setMinimumSize(640, 480)
        self.label_archivo.setScaledContents(True)

        self.btn_upload = QPushButton("Subir Video/Imagen")

        layout_botones = QHBoxLayout()
        layout_botones.addWidget(self.btn_upload)

        self.btn_volver = QPushButton("←")
        layout_volver = QHBoxLayout()
        layout_volver.addWidget(self.btn_volver)

        self.btn_volver.setMaximumWidth(30)
        layout_volver.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout_principal = QVBoxLayout()
        layout_principal.addLayout(layout_volver)
        layout_principal.addWidget(self.label_archivo)
        layout_principal.addLayout(layout_botones)

        
        self.btn_volver.clicked.connect(lambda: self.navegar.emit(0))
        self.btn_upload.clicked.connect(self.upload_videoImage)

        self.setLayout(layout_principal)

        self.roi = None
        

    def filtrar_archivo(self, filename):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return 'img'
            elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return 'vid'

    def process(self,filename):
            tipo = self.filtrar_archivo(filename)
            if tipo == 'img':
                imagen = cv2.imread(filename)
                raw_result = self.yolo.detect(imagen)
                self.imagen_actual = raw_result[0].plot()
                detecciones = self.yolo.parseToDictionary(None,raw_result)
                self.mostrar_resultados(self.imagen_actual)
                self.procesar_detecciones(detecciones)
                #for r in detection:
                #     print(r.boxes)
                #     print(r.boxes.cls)                #print('Distancias')
                #print(DistanceEstimation.runValidation(filename,detection[0].boxes))
                pass
            elif tipo == 'vid':
                if hasattr(self, 'worker') and self.worker.isRunning():
                    self.worker.stop()
                self.worker = Worker(fuente=filename)
                self.worker.frameReady.connect(self.mostrar_resultados)
                self.worker.detecciones.connect(self.procesar_detecciones)
                self.worker.activar_yolo()
                self.worker.start()
            pass    
    
    def upload_videoImage(self):
            filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Archivos de imagen (*.jpg *.jpeg *.png);;Archivos de video (*.mp4 *.avi *.mov)")
            if filename:
                self.process(filename)

    def mostrar_resultados(self, imagen):
        
        h,w,ch = imagen.shape
        self.roi = RoiFilter(w,h)
        print(f"shape: {imagen.shape}, roi: {self.roi}")
        cv2.polylines(imagen,[self.roi.contour],True,(0,255,0),2)
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_archivo.setPixmap(QPixmap.fromImage(qimg))

    def procesar_detecciones(self, detecciones):
        if self.roi is None:
             return
        filtradas = self.roi.filtro(detecciones)
        print(f"Detecciones totales: {len(detecciones)} | Filtradas ROI: {len(filtradas)}")
        for fil in filtradas:
             print(fil['clase_id'])

