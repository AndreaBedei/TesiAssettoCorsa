def classify_vehicle_state(
    steer, speed, g_force,
    wheel_slip, yaw_rate, gas
):
    
    # === Costanti configurabili ===
    # Coefficiente che converte l'angolo di sterzo in yaw rate atteso.
    # Tipico: 0.4 - 0.8 | Possibile: 0.0 - 1.0
    STEER_YAW_COEFFICIENT = 0.6

    # Tolleranza ammessa tra yaw rate reale e atteso per considerare il comportamento "normale".
    # Tipico: 0.03 - 0.08 | Possibile: 0.0 - 0.2
    YAW_TOLERANCE = 0.05

    # Differenza minima tra slip anteriore e posteriore per considerare uno sbilanciamento.
    # Tipico: 0.01 - 0.05 | Possibile: 0.0 - 0.2
    SLIP_DIFF_THRESHOLD = 0.02

    # Soglia oltre la quale una ruota è considerata in pattinamento.
    # Tipico: 0.15 - 0.25 | Possibile: 0.0 - 1.0
    WHEEL_SLIP_LIMIT = 0.2

    # G laterali (in curva) oltre cui si considera "alto grip".
    # Tipico: 1.0 - 1.5 G | Possibile: 0.0 - 3.0 (auto da corsa)
    LATERAL_G_LIMIT = 1.2

    # Velocità minima per considerare l'analisi dinamica valida (in km/h).
    # Tipico: 20 - 40 | Possibile: 0 - 400
    SPEED_THRESHOLD = 30

    # Valore del gas (0-1) oltre cui si considera che si sta accelerando davvero.
    # Tipico: 0.1 - 0.3 | Possibile: 0.0 - 1.0
    GAS_THRESHOLD = 0.2

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
    
    avg_slip = sum(wheel_slip) / len(wheel_slip)
    print(str(g_lateral))

    # === Logiche di comportamento ===
    if yaw_diff > YAW_TOLERANCE and slip_diff > SLIP_DIFF_THRESHOLD and front_slip > WHEEL_SLIP_LIMIT:
        return "sottosterzo"
    elif yaw_diff < -YAW_TOLERANCE and slip_diff < -SLIP_DIFF_THRESHOLD and rear_slip > WHEEL_SLIP_LIMIT and gas > GAS_THRESHOLD:
        return "sovrasterzo"
    elif avg_slip < WHEEL_SLIP_LIMIT and g_lateral > LATERAL_G_LIMIT:
        return "neutro_alto_grip"
    else:
        return "neutro"
