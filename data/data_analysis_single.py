import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columns = [
    'g_force_x', 'g_force_y', 'g_force_z',
    'wheel_slip_front_left', 'wheel_slip_front_right',
    'wheel_slip_rear_left', 'wheel_slip_rear_right',
    'pressure_front_left', 'pressure_front_right',
    'pressure_rear_left', 'pressure_rear_right',
    'tyre_temp_front_left', 'tyre_temp_front_right',
    'tyre_temp_rear_left', 'tyre_temp_rear_right',
    'air_temp', 'road_temp', 'yaw_rate'
]

def clip_outliers_iqr(df, features, multiplier=4):
    df_clipped = df.copy()
    for feature in features:
        if pd.api.types.is_numeric_dtype(df_clipped[feature]):
            Q1 = df_clipped[feature].quantile(0.25)
            Q3 = df_clipped[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df_clipped[feature] = df_clipped[feature].clip(lower=lower_bound, upper=upper_bound)
    return df_clipped

# Load the telemetry data
df = pd.read_csv("vehicle_telemetry_andrea_abu_36G_S_1_ora.csv")

# Display first few records to verify
print(df.head())

# General information about the dataset
print("\nDataset Info:")
print(df.info())

# Statistical summary (mean, std, min, max)
print("\nDescriptive Statistics:")
print(df.describe())

# Distribution of target variable
print("\nLabel distribution:")
print(df["result"].value_counts())

# Correlation Analysis
important_vars = ["gas", "brake", "speed", "yaw_rate", "g_force_x", "g_force_y", "g_force_z"]

corr_matrix = df[important_vars + ["normalized_car_position"] + ["steer"]].corr()

# Clipping outliers in all physical variables
df = clip_outliers_iqr(df, columns)

# # Plot Distributions of Important Variables To Understand the natural spread and potential abnormalities
# df[important_vars].hist(bins=30, figsize=(15, 10))
# plt.suptitle("Distribution of Key Physical Variables", fontsize=16)
# plt.tight_layout()
# plt.show()

# # Correlation Matrix Heatmap to understand relationships between forces, control inputs, and speed
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Matrix of Key Variables")
# plt.show()

# # Tire Temperature Analysis to see mean temperature across four tires
# tyre_temp_cols = [
#     "tyre_temp_front_left", "tyre_temp_front_right",
#     "tyre_temp_rear_left", "tyre_temp_rear_right"
# ]
# df["avg_tyre_temp"] = df[tyre_temp_cols].mean(axis=1)

# # Overall distribution of tire temperatures
# plt.figure(figsize=(10,6))
# sns.histplot(df["avg_tyre_temp"], kde=True)
# plt.title("Distribution of Average Tire Temperature °C")
# plt.show()

# # Tire temperature segmented by grip classification
# plt.figure(figsize=(10,6))
# sns.boxplot(data=df, x="result", y="avg_tyre_temp")
# plt.title("Average Tire Temperature by Grip Class")
# plt.show()

# # Wheel Slip Analysis to slip averages between front and rear tires
# df["slip_front_mean"] = (df["wheel_slip_front_left"] + df["wheel_slip_front_right"]) / 2
# df["slip_rear_mean"] = (df["wheel_slip_rear_left"] + df["wheel_slip_rear_right"]) / 2
# df["slip_mean"] = (df["slip_front_mean"] + df["slip_rear_mean"]) / 2

# # Compare slip distributions front vs rear
# plt.figure(figsize=(14,6))
# sns.histplot(df["slip_front_mean"], kde=True, color='blue', label='Front Slip')
# sns.histplot(df["slip_rear_mean"], kde=True, color='red', label='Rear Slip')
# plt.title("Distribution of Front vs Rear Wheel Slip")
# plt.xlabel("Wheel Slip Value")
# plt.legend()
# plt.grid()
# plt.show()

# # Front slip classified by grip
# plt.figure(figsize=(12,6))
# sns.boxplot(x="result", y="slip_front_mean", data=df)
# plt.title("Front Wheel Slip by Grip Classification")
# plt.grid()
# plt.show()

# # Rear slip classified by grip
# plt.figure(figsize=(12,6))
# sns.boxplot(x="result", y="slip_rear_mean", data=df)
# plt.title("Rear Wheel Slip by Grip Classification")
# plt.grid()
# plt.show()

# # Tire Pressure Analysis
# df["pressure_front_mean"] = (df["pressure_front_left"] + df["pressure_front_right"]) / 2
# df["pressure_rear_mean"] = (df["pressure_rear_left"] + df["pressure_rear_right"]) / 2

# # Distribution of front vs rear tire pressure
# plt.figure(figsize=(14,6))
# sns.histplot(df["pressure_front_mean"], kde=True, color='green', label='Front Pressure')
# sns.histplot(df["pressure_rear_mean"], kde=True, color='orange', label='Rear Pressure')
# plt.title("Distribution of Front vs Rear Tire Pressure PSI")
# plt.xlabel("Pressure")
# plt.legend()
# plt.grid()
# plt.show()

# # Front pressure classified by grip
# plt.figure(figsize=(12,6))
# sns.boxplot(x="result", y="pressure_front_mean", data=df)
# plt.title("Front Tire Pressure by Grip Classification")
# plt.grid()
# plt.show()

# # Tire Temperature Detailed Front vs Rear
# df["tyre_temp_front_mean"] = (df["tyre_temp_front_left"] + df["tyre_temp_front_right"]) / 2
# df["tyre_temp_rear_mean"] = (df["tyre_temp_rear_left"] + df["tyre_temp_rear_right"]) / 2
# plt.figure(figsize=(14,6))
# sns.histplot(df["tyre_temp_front_mean"], kde=True, color='purple', label='Front Tire Temp')
# sns.histplot(df["tyre_temp_rear_mean"], kde=True, color='pink', label='Rear Tire Temp')
# plt.title("Distribution of Front vs Rear Tire Temperatures °C")
# plt.xlabel("Temperature (°C)")
# plt.legend()
# plt.grid()
# plt.show()

# # Front tire temperature by grip classification
# plt.figure(figsize=(12,6))
# sns.boxplot(x="result", y="tyre_temp_front_mean", data=df)
# plt.title("Front Tire Temperature by Grip Classification")
# plt.grid()
# plt.show()

# custom_palette = {
#     "grip_loss": "red",
#     "neutral": "darkgreen",
#     "high_grip_accelerate": "lightgreen",
#     "low_grip_accelerate": "orange"
# }

# # Tire Pressure vs Tire Temperature The Relationship between pressure and temperature
# plt.figure(figsize=(12,6))
# sns.scatterplot(x="pressure_rear_mean", y="tyre_temp_rear_mean", hue="result", data=df, palette=custom_palette)
# plt.title("Rear Tire Pressure vs Rear Tire Temperature by Grip Classification")
# plt.xlabel("Pressure")
# plt.ylabel("Temperature °C")
# plt.grid()
# plt.show()

# # Speed Analysis, in particular the speed distribution by grip label
# plt.figure(figsize=(12,6))
# sns.boxplot(x="result", y="speed", data=df)
# plt.title("Vehicle Speed by Grip Classification")
# plt.grid()
# plt.show()

# # Speed along the track position (0=start, 1=end)
# plt.figure(figsize=(12,6))
# sns.scatterplot(x="normalized_car_position", y="speed", hue="result", data=df, palette=custom_palette)
# plt.title("Speed along the Track vs Grip Classification")
# plt.xlabel("Normalized Track Position")
# plt.ylabel("Speed (km/h)")
# plt.grid()
# plt.show()

# # Lateral G-Force analysis across track
# plt.figure(figsize=(12,6))
# sns.scatterplot(x="normalized_car_position", y="g_force_y", hue="result", data=df, palette=custom_palette)
# plt.title("Lateral G-Force along the Track by Grip Classification")
# plt.xlabel("Normalized Track Position")
# plt.ylabel("Lateral G-Force (g)")
# plt.grid()
# plt.show()

# # Mean Wheel Slip along the track
# plt.figure(figsize=(12,6))
# sns.scatterplot(x="normalized_car_position", y="slip_mean", hue="result", data=df, palette=custom_palette)
# plt.title("Mean Wheel Slip along the Track by Grip Classification")
# plt.xlabel("Normalized Track Position")
# plt.ylabel("Mean Wheel Slip")
# plt.grid()
# plt.show()


# Calcolo dello slip medio per ogni record
df["slip_mean"] = (
    df["wheel_slip_front_left"] + df["wheel_slip_front_right"] +
    df["wheel_slip_rear_left"] + df["wheel_slip_rear_right"]
) / 4

# Creazione di una colonna temporale sequenziale
df["time_index"] = range(len(df))

# Modifica dello slip medio per simulare una curva in salita
# df["slip_mean"] = df["slip_mean"] + (df["time_index"] * 0.000001)

# Calcolo della media dello slip ogni 100 dati
window_size = 4000
df["slip_mean_rolling"] = df["slip_mean"].rolling(window=window_size).mean()

# Tracciamento del grafico dello slip medio
plt.figure(figsize=(12, 6))
plt.plot(df["time_index"], df["slip_mean_rolling"], label=f"Mean Wheel Slip (Rolling {window_size})", color="blue")
plt.title(f"Mean Wheel Slip Over Time (Rolling Average of {window_size} Records)")
plt.xlabel("Time Index (Sequential Records)")
plt.ylabel("Mean Wheel Slip")
plt.grid()
plt.legend()
plt.show()