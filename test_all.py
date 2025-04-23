from env.shared_memory_physics import read_telemetry
from vision.screen_capture import capture_screen
from control.controller import send_action
from env.shared_memory_graphics import read_graphics
from utils.classify_vehicle_behavior import classify_vehicle_state

import cv2
import time

if __name__ == "__main__":
    print("[Test] Lettura dati e visualizzazione live")
    # cv2.namedWindow("Assetto POV", cv2.WINDOW_NORMAL)
    for _ in range(10000):
        telem = read_telemetry()
        print("Telemetry:", telem)
        graphics = read_graphics()
        print("Graphics:", graphics)
        result = classify_vehicle_state(
            telem["steer"], telem["speed"], telem["g_force"],
            telem["wheel_slip"], telem["yaw_rate"], telem["gas"]
        )
        print("Classificazione:", result)

        # frame = capture_screen()
        # cv2.imshow("Assetto POV", frame)

        # send_action([0.0, 0.5, 0.0])  # Sterzo 0, Gas 50%, Freno 0%

        if cv2.waitKey(1) == 27:
            break
        time.sleep(0.5)
    cv2.destroyAllWindows()
