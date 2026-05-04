from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from src.worker import Worker
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección vehícular con YOLOv8")
        self.setMinimumSize(800, 600)

        self.worker = Worker()
        self.worker.frameReady.connect(self.actualizar_frame)

        self._init_ui()

    def _init_ui(self):
        # Widget Central
        contenedor = QWidget()
        self.setCentralWidget(contenedor)

        #label video
        self.label_video = QLabel("Camara Apagada")
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_video.setMinimumSize(640, 480)
        self.label_video.setScaledContents(True)

        self.btn_camara = QPushButton("Activar Cámara")
        self.btn_detection = QPushButton("Activar Detección")
        self.btn_detection.setEnabled(False)

        layout_botones = QHBoxLayout()
        layout_botones.addWidget(self.btn_camara)
        layout_botones.addWidget(self.btn_detection)

        layout_principal = QVBoxLayout()
        layout_principal.addWidget(self.label_video)
        layout_principal.addLayout(layout_botones)

        contenedor.setLayout(layout_principal)

        # Conectar botones
        self.btn_camara.clicked.connect(self.toggle_camara)
        self.btn_detection.clicked.connect(self.toggle_deteccion)

    def toggle_camara(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.btn_camara.setText("Desactivar Cámara")
            self.btn_detection.setEnabled(True)
        else:
            self.worker.stop()
            self.btn_camara.setText("Activar Cámara")
            self.btn_detection.setEnabled(False)

    def toggle_deteccion(self):
        if not self.worker._yolo_activo:
            self.worker.activar_yolo()
            self.btn_detection.setText("Desactivar Detección")
        else:
            self.worker.desactivar_yolo()
            self.btn_detection.setText("Activar Detección")

    def actualizar_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
        event.accept()

