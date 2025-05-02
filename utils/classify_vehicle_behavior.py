def classify_vehicle_state(speed, g_force, wheel_slip):
    SLIP_LIMIT = 1.15  # slip considered excessive
    SLIP_HIGH_GRIP = 0.6 # under this value slip considered high grip

    g_lateral = abs(g_force[0])
    front_slip = (wheel_slip[0] + wheel_slip[1]) / 2
    rear_slip = (wheel_slip[2] + wheel_slip[3]) / 2

    speed_ms = speed / 3.6
    dynamic_g_curve = min(1.2, max(0.1, (speed_ms**2) / (100 * 9.81)))

    if (front_slip > SLIP_LIMIT or rear_slip > SLIP_LIMIT) and g_lateral > dynamic_g_curve:
        return "grip_loss"
    elif (front_slip < SLIP_HIGH_GRIP and rear_slip < SLIP_HIGH_GRIP) and g_lateral > dynamic_g_curve:
        return "high_grip_accelerate"
    
    elif (front_slip > SLIP_HIGH_GRIP and front_slip < SLIP_LIMIT or rear_slip > SLIP_HIGH_GRIP and rear_slip < SLIP_LIMIT) and g_lateral > dynamic_g_curve:
        return "low_grip_accelerate"
    else:
        return "neutral"

