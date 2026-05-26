import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("D02 Color IMG Centerline\Plot_profile_1.csv")

x = df["Distance_(pixels)"]
y = df["Gray_Value"]

window_size = 200
y_smooth = y.rolling(window=window_size, center=True).mean()

plt.figure(figsize=(10, 5))
plt.plot(x, y, alpha=0.3, label="Raw data")
plt.plot(x, y_smooth, linewidth=2, label="Smoothed data")
plt.xlabel("Distance (pixels)")
plt.ylabel("Gray Value")
plt.legend()
plt.grid(True)
plt.show()
