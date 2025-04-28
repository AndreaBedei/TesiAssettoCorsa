def classify_vehicle_state(speed, g_force, wheel_slip):
    # === Costanti configurabili ===
    SLIP_LIMIT = 1.15             # pattinamento considerato eccessivo
    SLIP_HIGH_GRIP = 0.6         # sotto questo valore, alto grip

    # === Calcoli ===
    g_lateral = abs(g_force[0])  # solo componente laterale

    front_slip = (wheel_slip[0] + wheel_slip[1]) / 2
    rear_slip = (wheel_slip[2] + wheel_slip[3]) / 2

    # Converti velocità da km/h a m/s
    speed_ms = speed / 3.6

    # Dinamica G curva: cresce con il quadrato della velocità
    dynamic_g_curve = min(1.2, max(0.1, (speed_ms**2) / (100 * 9.81)))

    # === Caso 1: perdita di aderenza ===
    if (front_slip > SLIP_LIMIT or rear_slip > SLIP_LIMIT) and g_lateral > dynamic_g_curve:
        return "grip_loss"

    # === Caso 2: sei in curva e puoi accelerare di più ===
    elif (front_slip < SLIP_HIGH_GRIP and rear_slip < SLIP_HIGH_GRIP) and g_lateral > dynamic_g_curve:
        return "high_grip_accelerate"
    
    elif (front_slip > SLIP_HIGH_GRIP and front_slip < SLIP_LIMIT or rear_slip > SLIP_HIGH_GRIP and rear_slip < SLIP_LIMIT) and g_lateral > dynamic_g_curve:
        return "low_grip_accelerate"
    else:
        return "neutral"

