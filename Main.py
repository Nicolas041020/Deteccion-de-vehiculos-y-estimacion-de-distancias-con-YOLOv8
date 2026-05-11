from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
import sys

if __name__ == "__main__":
    print("Main thread is running")
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec())