from pynput.keyboard import Controller, Key
import time

# Initialize the keyboard controller
keyboard = Controller()

# Function to simulate key presses
def restart_game():
    # Simulate Ctrl+O
    with keyboard.pressed(Key.ctrl):
        keyboard.press('o')
        time.sleep(0.5)
        keyboard.release('o')
        keyboard.press('k')
        time.sleep(0.5)
        keyboard.release('k')
        


