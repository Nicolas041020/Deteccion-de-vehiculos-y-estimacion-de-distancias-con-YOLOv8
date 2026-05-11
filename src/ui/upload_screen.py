from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
import cv2
from src.detection.yolo_detection import YoloDetection
from src.worker import Worker
from src.distanceEstimation.Distance_Estimation import DistanceEstimation

class UploadScreen(QWidget):
    navegar = pyqtSignal(int)
    yolo = YoloDetection("models/weights/yolov8n.pt")
    
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

    def filtrar_archivo(self, filename):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return 'img'
            elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return 'vid'

    def process(self,filename):
            tipo = self.filtrar_archivo(filename)
            if tipo == 'img':
                imagen = cv2.imread(filename)
                detection = self.yolo.detect(imagen)
                self.imagen_actual = detection[0].plot()
                self.mostrar_resultados(self.imagen_actual)
                print('Distancias')
                print(DistanceEstimation.run(filename,detection[0].boxes))
                pass
            elif tipo == 'vid':
                if hasattr(self, 'worker') and self.worker.isRunning():
                    self.worker.stop()
                self.worker = Worker(fuente=filename)
                self.worker.frameReady.connect(self.mostrar_resultados)
                self.worker.activar_yolo()
                self.worker.start()
            pass
    
    def upload_videoImage(self):
            filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Archivos de imagen (*.jpg *.jpeg *.png);;Archivos de video (*.mp4 *.avi *.mov)")
            if filename:
                self.process(filename)

    def mostrar_resultados(self, imagen):
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_archivo.setPixmap(QPixmap.fromImage(qimg))

