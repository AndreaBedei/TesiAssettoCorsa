import mss
import numpy as np
import cv2

sct = mss.mss()
monitor = sct.monitors[1]  # Primo monitor

def capture_screen():
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
