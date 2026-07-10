from PyQt6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout,
                              QHBoxLayout, QGroupBox, QSizePolicy)
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
    ruta = os.path.join(BASE_DIR, '..', '..', 'models', 'yolov8n.pt')
    yolo = YoloDetection(ruta)

    def __init__(self):
        super().__init__()
        self._distancia_ref = None
        self._obj_ref = None
        self._vecinos = []
        self._init_ui()

    def _init_ui(self):
        self.label_archivo = QLabel("Resultados")
        self.label_archivo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_archivo.setMinimumSize(640, 480)
        self.label_archivo.setScaledContents(True)

        # Info panel
        panel_info = QGroupBox("Distancias detectadas")
        panel_info.setMinimumWidth(230)
        panel_info.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout_panel = QVBoxLayout()
        self.lbl_ref = QLabel("Vehículo de referencia:\n—")
        self.lbl_v1 = QLabel("Vecino 1:\n—")
        self.lbl_v2 = QLabel("Vecino 2:\n—")
        for lbl in [self.lbl_ref, self.lbl_v1, self.lbl_v2]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            lbl.setWordWrap(True)
            layout_panel.addWidget(lbl)
        layout_panel.addStretch()
        panel_info.setLayout(layout_panel)

        layout_contenido = QHBoxLayout()
        layout_contenido.addWidget(self.label_archivo)
        layout_contenido.addWidget(panel_info)

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
        layout_principal.addLayout(layout_contenido)
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

    def process(self, filename):
        tipo = self.filtrar_archivo(filename)
        if tipo == 'img':
            imagen = cv2.imread(filename)
            raw_result = self.yolo.detect(imagen)
            imagen_plot = raw_result[0].plot()
            detecciones = self.yolo.parseToDictionary(raw_result)
            h, w = imagen_plot.shape[:2]
            self.roi = RoiFilter(w, h)
            self.procesar_detecciones(detecciones)
            self.mostrar_resultados(imagen_plot)
        elif tipo == 'vid':
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.stop()
            self.worker = Worker(fuente=filename)
            self.worker.frameReady.connect(self.mostrar_resultados)
            self.worker.detecciones.connect(self.procesar_detecciones)
            self.worker.activar_yolo()
            self.worker.start()

    def upload_videoImage(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo", "",
            "Archivos de imagen (*.jpg *.jpeg *.png);;Archivos de video (*.mp4 *.avi *.mov)")
        if filename:
            self.process(filename)

    def _dibujar_lineas(self, imagen):
        if self._obj_ref is None or not self._vecinos:
            return
        x1r, y1r, x2r, y2r = self._obj_ref['bbox']
        cx_ref = int((x1r + x2r) / 2)
        cy_ref = int((y1r + y2r) / 2)
        cv2.circle(imagen, (cx_ref, cy_ref), 7, (0, 255, 0), -1)

        # naranja y azul (BGR)
        colores = [(0, 140, 255), (255, 80, 0)]
        for idx, (d_AB, obj_v) in enumerate(self._vecinos):
            x1v, y1v, x2v, y2v = obj_v['bbox']
            cx_v = int((x1v + x2v) / 2)
            cy_v = int((y1v + y2v) / 2)
            color = colores[idx % len(colores)]
            cv2.line(imagen, (cx_ref, cy_ref), (cx_v, cy_v), color, 2)
            mx = (cx_ref + cx_v) // 2
            my = (cy_ref + cy_v) // 2
            cv2.putText(imagen, f"{d_AB:.2f} m", (mx + 5, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    def _actualizar_panel(self):
        if self._obj_ref is None or self._distancia_ref is None:
            self.lbl_ref.setText("Vehículo de referencia:\n—")
            self.lbl_v1.setText("Vecino 1:\n—")
            self.lbl_v2.setText("Vecino 2:\n—")
            return
        self.lbl_ref.setText(
            f"Vehículo de referencia:\ndistancia Z = {self._distancia_ref:.2f} m")
        etiquetas = [self.lbl_v1, self.lbl_v2]
        for i, lbl in enumerate(etiquetas):
            if i < len(self._vecinos):
                d_AB, _ = self._vecinos[i]
                lbl.setText(f"Vecino {i + 1}:\ndistancia = {d_AB:.2f} m")
            else:
                lbl.setText(f"Vecino {i + 1}:\n—")

    def mostrar_resultados(self, imagen):
        h, w, ch = imagen.shape
        self.roi = RoiFilter(w, h)
        cv2.polylines(imagen, [self.roi.contour], True, (0, 255, 0), 2)
        self._dibujar_lineas(imagen)
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_archivo.setPixmap(QPixmap.fromImage(qimg))

    def procesar_detecciones(self, detecciones):
        if self.roi is None:
            return
        filtradas = self.roi.filtro(detecciones)
        print(f"Detecciones totales: {len(detecciones)} | Filtradas ROI: {len(filtradas)}")
        try:
            z_ref, obj_ref, vecinos = DistanceEstimation.distanciasIntervehiculares(filtradas)
            self._distancia_ref = z_ref
            self._obj_ref = obj_ref
            self._vecinos = vecinos if vecinos else []
        except Exception as e:
            print(f"Error estimando distancias: {e}")
            self._distancia_ref = None
            self._obj_ref = None
            self._vecinos = []
        self._actualizar_panel()
        return filtradas
