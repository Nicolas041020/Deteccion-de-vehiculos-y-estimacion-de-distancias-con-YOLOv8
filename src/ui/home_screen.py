from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout 
from PyQt6.QtCore import Qt, pyqtSignal

class HomeScreen(QWidget):
    navegar = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):

        self.button_realtime = QPushButton("Detección en Tiempo Real")
        self.button_upload = QPushButton("Subir Video/Imagen")

        layout_botones = QVBoxLayout()
        layout_botones.addWidget(self.button_realtime)
        layout_botones.addWidget(self.button_upload)

        #Configurar Estilo de los botones
        self.button_realtime.setMaximumWidth(200)
        self.button_upload.setMaximumWidth(200)

        layout_botones.setSpacing(10)
        layout_botones.setContentsMargins(0, 0, 0, 0)
        layout_botones.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout_botones)


        self.button_realtime.clicked.connect(lambda: self.navegar.emit(1))
        self.button_upload.clicked.connect(lambda: self.navegar.emit(2))

