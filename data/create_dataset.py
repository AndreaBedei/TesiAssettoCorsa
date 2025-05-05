
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.shared_memory_physics import read_telemetry
from env.shared_memory_graphics import read_graphics
from utils.classify_vehicle_behavior import classify_vehicle_state

import cv2
import time
import csv

if __name__ == "__main__":
    csv_filename = 'vehicle_telemetry_andrea_abu_36G_S_1_ora.csv'
    fields = [
        "gas", "brake", "rpm", "steer", "speed", "g_force_x", "g_force_y", "g_force_z",
        "wheel_slip_front_left", "wheel_slip_front_right", "wheel_slip_rear_left", "wheel_slip_rear_right",
        "pressure_front_left", "pressure_front_right", "pressure_rear_left", "pressure_rear_right",
        "tyre_temp_front_left", "tyre_temp_front_right", "tyre_temp_rear_left", "tyre_temp_rear_right",
        "air_temp", "road_temp", "yaw_rate", "current_time_str", "normalized_car_position", "wind_speed", "wind_direction", "result"
    ]

    # Crea il file CSV con l'intestazione (se non esiste)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

    print("Creazione dataset")

    # Ciclo per raccogliere e scrivere i dati ogni volta
    for i in range(20731):
        telem = read_telemetry()
        graphics = read_graphics()
        
        # Classificazione del comportamento del veicolo
        result = classify_vehicle_state(
            telem["speed"], telem["g_force"],
            telem["wheel_slip"]
        )

        # # Scrittura del record nel CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writerow({
                "gas": telem["gas"],
                "brake": telem["brake"],
                "rpm": telem["rpm"],
                "steer": telem["steer"],
                "speed": telem["speed"],
                "g_force_y": telem["g_force"][0], # Inversione a causa di un bug
                "g_force_z": telem["g_force"][1], # Inversione a causa di un bug
                "g_force_x": telem["g_force"][2], # Inversione a causa di un bug
                "wheel_slip_front_left": telem["wheel_slip"][0],
                "wheel_slip_front_right": telem["wheel_slip"][1],
                "wheel_slip_rear_left": telem["wheel_slip"][2],
                "wheel_slip_rear_right": telem["wheel_slip"][3],
                "pressure_front_left": telem["pressure"][0],
                "pressure_front_right": telem["pressure"][1],
                "pressure_rear_left": telem["pressure"][2],
                "pressure_rear_right": telem["pressure"][3],
                "tyre_temp_front_left": telem["tyre_temp"][0],
                "tyre_temp_front_right": telem["tyre_temp"][1],
                "tyre_temp_rear_left": telem["tyre_temp"][2],
                "tyre_temp_rear_right": telem["tyre_temp"][3],
                "air_temp": telem["air_temp"],
                "road_temp": telem["road_temp"],
                "yaw_rate": telem["yaw_rate"],
                "current_time_str": graphics["current_time_str"].rstrip('\x00'),
                "normalized_car_position": graphics["normalized_car_position"],
                "wind_speed": graphics["wind_speed"],
                "wind_direction": graphics["wind_direction"],
                "result": result
            })
        
        # Interruzione del ciclo al tasto ESC
        if cv2.waitKey(1) == 27:
            break
        
        time.sleep(0.10)

    print("Dataset creato")