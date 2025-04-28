# def classify_vehicle_state(
#     steer, speed, g_force,
#     wheel_slip, yaw_rate, gas
# ):
#     # === Costanti configurabili ===
#     STEER_YAW_COEFFICIENT = 0.6
#     BASE_YAW_TOLERANCE = 0.10
#     SLIP_DIFF_THRESHOLD = 0.25
#     WHEEL_SLIP_LIMIT = 0.5
#     BASE_G_LIMIT = 0.7
#     SPEED_THRESHOLD = 20

#     # === Precondizione: se la velocità è troppo bassa, inutile valutare ===
#     if speed < SPEED_THRESHOLD:
#         return "neutro"
    
#     # === Calcoli derivati ===
#     expected_yaw_rate = (steer * STEER_YAW_COEFFICIENT) * (speed / 100)
#     yaw_diff = yaw_rate - expected_yaw_rate

#     # Filtra pattinamenti anomali
#     filtered_wheel_slip = [min(slip, 5.0) for slip in wheel_slip]
#     front_slip = (filtered_wheel_slip[0] + filtered_wheel_slip[1]) / 2
#     rear_slip = (filtered_wheel_slip[2] + filtered_wheel_slip[3]) / 2
#     slip_diff = front_slip - rear_slip
#     avg_slip = sum(filtered_wheel_slip) / len(filtered_wheel_slip)

#     # G laterale assoluto
#     g_lateral = abs(g_force[0])

#     # === Adattamento dinamico ===
#     # Yaw diff aumenta con la velocità (maggiore yaw a parità di sterzo)
#     yaw_tolerance = BASE_YAW_TOLERANCE * (1 + speed / 100)

#     # G limite cresce con la velocità (più velocità => più G tollerabile)
#     dynamic_g_limit = BASE_G_LIMIT + (speed / 100) * 0.3  # es. da 0.7 a 1.3 G

#     # === Logica ===
#     # yaw_diff → quanto lo sterzo effettivo (yaw rate) differisce da quello atteso
#     # slip_diff → quanto le ruote anteriori slittano più (o meno) delle posteriori
#     # g_lateral → quanto grip c'è in curva (più G laterale = più grip)
#     # avg_slip → quanto stanno pattinando in media le ruote
#     # front_slip / rear_slip → pattinamento anteriore/posteriore separati
#     if (
#         yaw_diff > yaw_tolerance and 
#         slip_diff > SLIP_DIFF_THRESHOLD and 
#         front_slip > WHEEL_SLIP_LIMIT
#     ):
#         return "sottosterzo"
    
#     elif (
#         yaw_diff < -yaw_tolerance and 
#         slip_diff < -SLIP_DIFF_THRESHOLD and 
#         rear_slip > WHEEL_SLIP_LIMIT
#     ):
#         return "sovrasterzo"
    
#     elif avg_slip < WHEEL_SLIP_LIMIT and g_lateral > dynamic_g_limit:
#         return "neutro_alto_grip"
    
#     # Nuovo: rileva potenziale perdita anche a ruote dritte ma yaw alto
#     elif yaw_rate > 0.2 and avg_slip > WHEEL_SLIP_LIMIT:
#         return "perdita_aderenza_curva_ampia"

#     else:
#         return "neutro"

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

