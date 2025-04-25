import vgamepad as vg
import cv2
import time

from env.shared_memory_physics import read_telemetry
from vision.screen_capture import capture_screen
from control.controller import send_action
from env.shared_memory_graphics import read_graphics
from data.restart import restart_game

gamepad=vg.VX360Gamepad()


if __name__ == "__main__":
    print("[Test] Lettura dati e visualizzazione live")
    cv2.namedWindow("Assetto POV", cv2.WINDOW_NORMAL)
    for _ in range(10000):
        
        telem = read_telemetry()
        graphics = read_graphics()

        frame = capture_screen()
        cv2.imshow("Assetto POV", frame)
        send_action([0.0, 1, 0.0], gamepad)  # Sterzo 0, Gas 50%, Freno 0%

        if cv2.waitKey(1) == 27:
            break
        time.sleep(3)

        restart_game()
        
    cv2.destroyAllWindows()
