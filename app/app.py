import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QComboBox
)
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtMultimedia import QSoundEffect


LABEL_STATES = {
    0: ("Neutral", QColor(0, 128, 0)),
    1: ("High Grip - Accelerate", QColor(0, 255, 0)),
    2: ("Low Grip", QColor(255, 165, 0)),
    3: ("Grip Loss", QColor(255, 0, 0))
}

def interpolate_color(value, max_val=0.7):
    ratio = min(max(value / max_val, 0.0), 1.0)
    hue = (120 - 120 * ratio) / 360.0  # green to red
    color = QColor.fromHsvF(hue, 0.85, 0.9)  # più desaturato, stile vintage
    return color

class WheelVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.slips = [0.0] * 4
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f3e8d3; border: 2px solid #a18860;")

    def update_slips(self, slips):
        self.slips = slips
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        car_w = w * 0.2
        car_h = h * 0.8
        axle_offset = h * 0.22
        wheel_r = min(w, h) * 0.10

        center_x = w // 2
        center_y = h // 2

        # Car body
        car_rect_x = center_x - car_w / 2
        car_rect_y = center_y - car_h / 2
        painter.setBrush(QColor(40, 40, 40))
        painter.setPen(QPen(QColor(100, 200, 255), 2)) 
        painter.drawRoundedRect(int(car_rect_x + 17), int(car_rect_y), int(car_w - 30), int(car_h), 10, 10)

        # Axles
        painter.setBrush(QColor(90, 90, 90))
        painter.setPen(QPen(QColor(100, 200, 255), 2)) 
        painter.drawRoundedRect(int(car_rect_x), int(center_y - axle_offset - 5), int(car_w), 10, 3, 3)
        painter.drawRoundedRect(int(car_rect_x), int(center_y + axle_offset - 5), int(car_w), 10, 3, 3)

        # Wheels
        wheel_positions = [
            (car_rect_x - wheel_r * 2, center_y - axle_offset - wheel_r),
            (car_rect_x + car_w , center_y - axle_offset - wheel_r),
            (car_rect_x - wheel_r * 2, center_y + axle_offset - wheel_r),
            (car_rect_x + car_w, center_y + axle_offset - wheel_r),
        ]

        font_size = max(8, int(wheel_r * 0.35))
        for i, (x, y) in enumerate(wheel_positions):
            slip = self.slips[i]
            color = interpolate_color(slip)

            painter.setBrush(color)
            painter.setPen(QPen(Qt.black, 1.2))
            painter.drawEllipse(int(x), int(y), int(wheel_r * 2), int(wheel_r * 2))

            painter.setPen(QPen(Qt.black))
            painter.setFont(QFont("Georgia", font_size, QFont.Bold))
            painter.drawText(
                int(x + wheel_r / 2), int(y + wheel_r * 2 + font_size + 10), f"{slip:.2f}"
            )

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slip Dashboard")
        self.setMinimumSize(600, 500)

        self.setWindowIcon(QIcon("img.png"))

        self.current_label_pred = None  # salva l'ultima label mostrata

        # Carica i suoni
        self.sounds = {
            0: QSoundEffect(),
            1: QSoundEffect(),
            2: QSoundEffect(),
            3: QSoundEffect()
        }
        self.sounds[1].setSource(QUrl.fromLocalFile("high_grip.wav"))
        self.sounds[2].setSource(QUrl.fromLocalFile("limit_grip.wav"))
        self.sounds[3].setSource(QUrl.fromLocalFile("loss_grip.wav"))

        for s in self.sounds.values():
            s.setVolume(0.5)

        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Roboto', sans-serif;
            }

            QComboBox {
                border: 2px solid #0078d7;
                border-radius: 10px;
                padding: 10px;
                background-color: #f0f4f8;
                font-size: 14px;
                color: #333333;
                padding-right: 30px; 
            }

            QComboBox:hover {
                background-color: #e6f0fa;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #0078d7;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }

            QComboBox::down-arrow {
                image: url(arrow.svg);  
                width: 25%;
                height: 25%;
                margin-right: 7px;
            }

            QComboBox QAbstractItemView {
                border: 1px solid #0078d7;
                selection-background-color: #0078d7;
                selection-color: white;
                padding: 5px;
                background-color: #ffffff;
                outline: none;
            }
        """)


        self.visualizer = WheelVisualizer()

        self.status_label = QLabel("Status: ---")
        self.status_label.setFont(QFont("Roboto", 16, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: #f0f4f8;
            border: 2px solid #0078d7;
            border-radius: 12px;
            padding: 16px;
            color: #333333;
        """)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Simple Model", "LSTM", "Transformers", "CNN 1D", "CNN 1D Sequential"])
        self.model_selector.setFont(QFont("Roboto", 14))

        layout = QVBoxLayout()
        layout.addWidget(self.model_selector)
        layout.addWidget(self.visualizer, stretch=1)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)

    def update_data(self):
        slips = [0,0,0,0] #[random.uniform(0, 0.7) for _ in range(4)]
        label_pred = 3 #random.choice([0, 1, 2])

        self.visualizer.update_slips(slips)

        label_text, label_color = LABEL_STATES[label_pred]
        self.status_label.setText(f"{label_text}")
        self.status_label.setStyleSheet(
            f"""
            background-color: #f0f4f8;
            border: 2px solid #0078d7;
            border-radius: 12px;
            padding: 16px;
            color: rgb({label_color.red()}, {label_color.green()}, {label_color.blue()});
            """
        )

        
        self.current_label_pred = label_pred
        self.sounds[label_pred].play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
