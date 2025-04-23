def classify_vehicle_state(
    steer, speed, g_force,
    wheel_slip, yaw_rate, gas
):
    # === Costanti configurabili ===
    STEER_YAW_COEFFICIENT = 0.6       # Fattore conversione steer → yaw atteso
    YAW_TOLERANCE = 0.05              # Tolleranza differenza yaw atteso vs reale
    SLIP_DIFF_THRESHOLD = 0.02        # Differenza significativa slip tra assi
    WHEEL_SLIP_LIMIT = 0.2            # Soglia oltre la quale inizia il pattinamento
    LATERAL_G_LIMIT = 1.2             # G laterali oltre cui si ha grip forte
    SPEED_THRESHOLD = 30              # Per ignorare situazioni lente
    GAS_THRESHOLD = 0.2               # Per capire se si sta accelerando davvero

    # === Calcoli base ===
    expected_yaw_rate = (steer * STEER_YAW_COEFFICIENT) * (speed / 100)
    yaw_diff = yaw_rate - expected_yaw_rate

    front_slip = (wheel_slip[0] + wheel_slip[1]) / 2
    rear_slip = (wheel_slip[2] + wheel_slip[3]) / 2
    slip_diff = front_slip - rear_slip

    g_lateral = abs(g_force[1])

    # === Condizioni generali da ignorare (velocità troppo bassa) ===
    if speed < SPEED_THRESHOLD:
        return "neutro"

    # === Logiche di comportamento ===
    if yaw_diff > YAW_TOLERANCE and slip_diff > SLIP_DIFF_THRESHOLD and front_slip > WHEEL_SLIP_LIMIT:
        return "sottosterzo"
    elif yaw_diff < -YAW_TOLERANCE and slip_diff < -SLIP_DIFF_THRESHOLD and rear_slip > WHEEL_SLIP_LIMIT and gas > GAS_THRESHOLD:
        return "sovrasterzo"
    elif all(slip < WHEEL_SLIP_LIMIT for slip in wheel_slip) and g_lateral > LATERAL_G_LIMIT:
        return "neutro_alto_grip"
    else:
        return "neutro"
