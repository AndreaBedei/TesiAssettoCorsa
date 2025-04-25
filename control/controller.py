
# action: [steer (-1,1), throttle (0,1), brake (0,1)]
def send_action(action, gamepad):
    steer, throttle, brake = action

    # Sterzo: asse sinistro X (da -32768 a +32767)
    steer_val = int(steer * 32767)
    gamepad.left_joystick(x_value=steer_val, y_value=0)

    # Acceleratore: grilletto destro (0-255)
    throttle_val = int(throttle * 255)
    gamepad.right_trigger(value=throttle_val)

    # Freno: grilletto sinistro (0-255)
    brake_val = int(brake * 255)
    gamepad.left_trigger(value=brake_val)

    # Invia aggiornamenti al gamepad
    gamepad.update()
