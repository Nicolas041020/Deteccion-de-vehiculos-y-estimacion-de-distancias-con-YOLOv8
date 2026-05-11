from PyQt6.QtWidgets import QMainWindow,QWidget, QStackedWidget
from src.ui.home_screen import HomeScreen
from src.ui.realtime_screen import RealTimeScreen
from src.ui.upload_screen import UploadScreen


class MainWindow(QMainWindow):  
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección vehícular con YOLOv8")
        self.setMinimumSize(800, 600)
        self._init_ui()

    def _init_ui(self):
        self.stack = QStackedWidget()
        self.home_screen = HomeScreen()
        self.realtime_screen = RealTimeScreen()
        self.upload_screen = UploadScreen()

        self.stack.addWidget(self.home_screen)
        self.stack.addWidget(self.realtime_screen)
        self.stack.addWidget(self.upload_screen)
        self.home_screen.navegar.connect(self.cambiar_pantalla)
        self.realtime_screen.navegar.connect(self.cambiar_pantalla)
        self.upload_screen.navegar.connect(self.cambiar_pantalla)
        self.setCentralWidget(self.stack)

    def cambiar_pantalla(self, index):
        self.stack.setCurrentIndex(index)

