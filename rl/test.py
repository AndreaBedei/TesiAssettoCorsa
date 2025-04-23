import ctypes
import mmap
import struct
import time

class SPageFilePhysics(ctypes.Structure):
    _fields_ = [
        ("packetId", ctypes.c_int),
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("fuel", ctypes.c_float),
        ("gear", ctypes.c_int),
        ("rpm", ctypes.c_int),
        ("steerAngle", ctypes.c_float),
        ("speedKmh", ctypes.c_float),
        ("velocity", ctypes.c_float * 3),  # x, y, z
        ("accG", ctypes.c_float * 3),
        ("wheelSlip", ctypes.c_float * 4),
        ("wheelLoad", ctypes.c_float * 4),
        ("wheelsPressure", ctypes.c_float * 4),
        ("wheelsAngularSpeed", ctypes.c_float * 4),
        ("tyreWear", ctypes.c_float * 4),
        ("tyreDirtyLevel", ctypes.c_float * 4),
        ("tyreCoreTemperature", ctypes.c_float * 4),
        ("camberRAD", ctypes.c_float * 4),
        ("suspensionTravel", ctypes.c_float * 4),
        ("drsAvailable", ctypes.c_int),
        ("drsEnabled", ctypes.c_int),
        ("engineBrake", ctypes.c_int),
        ("ersRecoveryLevel", ctypes.c_int),
        ("ersPowerLevel", ctypes.c_int),
        ("ersHeatCharging", ctypes.c_int),
        ("ersIsCharging", ctypes.c_int),
        ("kersCurrentKJ", ctypes.c_float),
        ("engineIdleRpm", ctypes.c_int),
        ("engineMaxRpm", ctypes.c_int),
        ("currentLapTime", ctypes.c_float),
        ("lastLapTime", ctypes.c_float),
        ("bestLapTime", ctypes.c_float),
        ("lapCount", ctypes.c_int),
        ("gasPedalPosition", ctypes.c_float),
    ]

def read_physics():
    try:
        # Nome della shared memory di Assetto Corsa
        filename = "Local\\acpmf_physics"
        map_file = mmap.mmap(-1, ctypes.sizeof(SPageFilePhysics), filename, access=mmap.ACCESS_READ)
        
        while True:
            map_file.seek(0)
            data = map_file.read(ctypes.sizeof(SPageFilePhysics))
            physics = SPageFilePhysics.from_buffer_copy(data)
            
            print(f"Velocità: {physics.speedKmh:.2f} km/h | Giri: {physics.rpm} | Marcia: {physics.gear}")
            print(f"Temperatura gomme: {[f'{temp:.2f}' for temp in physics.tyreCoreTemperature]} °C")
            print(f"Forza G: X={physics.accG[0]:.2f}, Y={physics.accG[1]:.2f}, Z={physics.accG[2]:.2f}")
            print(f"Pressione gomme: {[f'{pressure:.2f}' for pressure in physics.wheelsPressure]} bar")
            print(f"Usura gomme: {[f'{wear:.2f}' for wear in physics.tyreWear]}")
            print(f"Carico ruote: {[f'{load:.2f}' for load in physics.wheelLoad]} N")
            time.sleep(1)
            print("\n" + "-"*50 + "\n")

    except Exception as e:
        print("Errore nella lettura:", e)

if __name__ == "__main__":
    read_physics()
