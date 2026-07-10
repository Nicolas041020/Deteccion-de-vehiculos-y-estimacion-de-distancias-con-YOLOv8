from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from src.distanceEstimation.Distance_Estimation import DistanceEstimation
import os
from src.detection.yolo_detection import YoloDetection

RANGOS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]


class ValidationScreen(QWidget):
    navegar = pyqtSignal(int)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE_DIR, '..', '..', 'models', 'yolov8n.pt')
    yolo = YoloDetection(ruta)

    def __init__(self):
        super().__init__()
        self._init_ui()

    # ------------------------------------------------------------------
    # Helpers de fuente
    # ------------------------------------------------------------------
    @staticmethod
    def _font(bold=False, size=None):
        f = QFont()
        if bold:
            f.setBold(True)
        if size:
            f.setPointSize(size)
        return f

    # ------------------------------------------------------------------
    # Construcción de UI
    # ------------------------------------------------------------------
    def _init_ui(self):
        # botón volver
        self.btn_volver = QPushButton("←")
        self.btn_volver.setMaximumWidth(30)
        self.btn_volver.clicked.connect(lambda: self.navegar.emit(0))
        layout_volver = QHBoxLayout()
        layout_volver.addWidget(self.btn_volver)
        layout_volver.addStretch()

        # título
        titulo = QLabel("Resultados de validación")
        titulo.setFont(self._font(bold=True, size=14))
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sep0 = self._separador()

        # grilla métricas globales
        grid_global = self._build_global_grid()

        sep1 = self._separador()

        # título sección rangos
        lbl_rangos = QLabel("Métricas por rango de distancia")
        lbl_rangos.setFont(self._font(bold=True, size=11))
        lbl_rangos.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # grilla rangos
        grid_rangos = self._build_rangos_grid()

        # botón
        self.btn_iniciarval = QPushButton("Iniciar Validación")
        self.btn_iniciarval.clicked.connect(self.iniciarval)

        # contenido interno (scroll)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.addLayout(layout_volver)
        inner_layout.addSpacing(6)
        inner_layout.addWidget(titulo)
        inner_layout.addWidget(sep0)
        inner_layout.addSpacing(10)
        inner_layout.addLayout(grid_global)
        inner_layout.addSpacing(16)
        inner_layout.addWidget(sep1)
        inner_layout.addSpacing(8)
        inner_layout.addWidget(lbl_rangos)
        inner_layout.addSpacing(6)
        inner_layout.addLayout(grid_rangos)
        inner_layout.addSpacing(20)
        inner_layout.addWidget(self.btn_iniciarval, alignment=Qt.AlignmentFlag.AlignCenter)
        inner_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    # ------------------------------------------------------------------
    def _separador(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    # ------------------------------------------------------------------
    def _build_global_grid(self):
        grid = QGridLayout()
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(20)

        # encabezados de columna
        for col, texto in enumerate(["H_px YOLO", "H_px KITTI"], start=1):
            lbl = QLabel(texto)
            lbl.setFont(self._font(bold=True, size=10))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)

        sep = self._separador()
        grid.addWidget(sep, 1, 0, 1, 3)

        metricas = [
            ("Número de Imágenes", "label_img"),
            ("Número de Objetos",  "label_n"),
            ("MAE (m)",            "label_mae"),
            ("RMSE (m)",           "label_rmse"),
            ("Desv. estándar (m)", "label_std"),
            ("Error máximo (m)",   "label_max"),
        ]

        for i, (texto, attr) in enumerate(metricas):
            fila = i + 2
            nombre = QLabel(texto)
            nombre.setFont(self._font(bold=True))
            nombre.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            for sufijo, col in (("_yolo", 1), ("_kitti", 2)):
                val = QLabel("—")
                val.setAlignment(Qt.AlignmentFlag.AlignCenter)
                setattr(self, f"{attr}{sufijo}", val)
                grid.addWidget(val, fila, col)

            grid.addWidget(nombre, fila, 0)

        return grid

    # ------------------------------------------------------------------
    def _build_rangos_grid(self):
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        for c in range(1, 6):
            grid.setColumnStretch(c, 1)
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(12)

        # encabezados
        headers = ["Rango", "n", "MAE YOLO", "RMSE YOLO", "MAE KITTI", "RMSE KITTI"]
        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setFont(self._font(bold=True, size=9))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)

        sep = self._separador()
        grid.addWidget(sep, 1, 0, 1, 6)

        # filas de datos (una por rango, pre-creadas)
        self._rango_rows = []
        for i, (rmin, rmax) in enumerate(RANGOS):
            fila = i + 2
            lbl_rango = QLabel(f"{rmin}-{rmax} m")
            lbl_rango.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl_rango, fila, 0)

            celdas = {}
            for col, key in enumerate(["n", "mae_yolo", "rmse_yolo", "mae_kitti", "rmse_kitti"], start=1):
                val = QLabel("—")
                val.setAlignment(Qt.AlignmentFlag.AlignCenter)
                grid.addWidget(val, fila, col)
                celdas[key] = val

            self._rango_rows.append(celdas)

        return grid

    # ------------------------------------------------------------------
    # Lógica de validación
    # ------------------------------------------------------------------
    def iniciarval(self):
        self.btn_iniciarval.setText("Calculando...")
        self.btn_iniciarval.setEnabled(False)

        muestras, n, mae, rmse, std, max_err, maee, rmsee, stdd, maxe, rangos_data = \
            DistanceEstimation.validacionKITTI(self.yolo)

        # métricas globales — columna YOLO
        self.label_img_yolo.setText(str(muestras))
        self.label_n_yolo.setText(str(n))
        self.label_mae_yolo.setText(f"{mae:.4f}")
        self.label_rmse_yolo.setText(f"{rmse:.4f}")
        self.label_std_yolo.setText(f"{std:.4f}")
        self.label_max_yolo.setText(f"{max_err:.4f}")

        # métricas globales — columna KITTI
        self.label_img_kitti.setText(str(muestras))
        self.label_n_kitti.setText(str(n))
        self.label_mae_kitti.setText(f"{maee:.4f}")
        self.label_rmse_kitti.setText(f"{rmsee:.4f}")
        self.label_std_kitti.setText(f"{stdd:.4f}")
        self.label_max_kitti.setText(f"{maxe:.4f}")

        # métricas por rango
        for celdas, fila in zip(self._rango_rows, rangos_data):
            if fila['n'] == 0:
                celdas['n'].setText("0")
                for key in ("mae_yolo", "rmse_yolo", "mae_kitti", "rmse_kitti"):
                    celdas[key].setText("—")
            else:
                celdas['n'].setText(str(fila['n']))
                celdas['mae_yolo'].setText(f"{fila['mae_yolo']:.3f}")
                celdas['rmse_yolo'].setText(f"{fila['rmse_yolo']:.3f}")
                celdas['mae_kitti'].setText(f"{fila['mae_kitti']:.3f}")
                celdas['rmse_kitti'].setText(f"{fila['rmse_kitti']:.3f}")

        self.btn_iniciarval.setText("Iniciar Validación")
        self.btn_iniciarval.setEnabled(True)
