def classify_vehicle_state(
    steer, speed, g_force,
    wheel_slip, yaw_rate, gas
):
    
    # === Costanti configurabili ===
    # Coefficiente che converte l'angolo di sterzo in yaw rate atteso.
    # Tipico: 0.4 - 0.8 | Possibile: 0.0 - 1.0
    STEER_YAW_COEFFICIENT = 0.6
    YAW_TOLERANCE = 0.13          # Se yaw diff > 0.13 è rilevante
    SLIP_DIFF_THRESHOLD = 0.3     # Differenza slip anteriore vs posteriore significativa
    WHEEL_SLIP_LIMIT = 1.0        # Pattinamento oltre 1.0 è eccessivo
    LATERAL_G_LIMIT = 0.9         # Alto grip se > 0.9 G
    SPEED_THRESHOLD = 30          # Analisi solo sopra i 30 km/h
    GAS_THRESHOLD = 0.2           # Gas > 0.2 per sovrasterzo

    # === Calcoli base ===
    expected_yaw_rate = (steer * STEER_YAW_COEFFICIENT) * (speed / 100)
    yaw_diff = yaw_rate - expected_yaw_rate


    front_slip = (wheel_slip[0] + wheel_slip[1]) / 2
    rear_slip = (wheel_slip[2] + wheel_slip[3]) / 2
    slip_diff = front_slip - rear_slip

    g_lateral = abs(g_force[0])

    # === Condizioni generali da ignorare (velocità troppo bassa) ===
    if speed < SPEED_THRESHOLD:
        return "neutro"
    
    # Normalizza slip escludendo valori palesemente corrotti (es: > 100 m/s)
    filtered_wheel_slip = [min(slip, 5.0) for slip in wheel_slip]
    front_slip = (filtered_wheel_slip[0] + filtered_wheel_slip[1]) / 2
    rear_slip = (filtered_wheel_slip[2] + filtered_wheel_slip[3]) / 2
    slip_diff = front_slip - rear_slip
    avg_slip = sum(filtered_wheel_slip) / len(filtered_wheel_slip)

    # === Logiche di comportamento ===
    if yaw_diff > YAW_TOLERANCE and slip_diff > SLIP_DIFF_THRESHOLD and front_slip > WHEEL_SLIP_LIMIT:
        return "sottosterzo"
    elif yaw_diff < -YAW_TOLERANCE and slip_diff < -SLIP_DIFF_THRESHOLD and rear_slip > WHEEL_SLIP_LIMIT and gas > GAS_THRESHOLD:
        return "sovrasterzo"
    elif avg_slip < WHEEL_SLIP_LIMIT and g_lateral > LATERAL_G_LIMIT:
        return "neutro_alto_grip"
    else:
        return "neutro"
