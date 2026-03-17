import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load data
# -------------------------------------------------
df_BOS = pd.read_csv(r'CC Data/displacement_vectors4.csv', delimiter=',')

# -------------------------------------------------
# Extract raw BOS data
# -------------------------------------------------
x = df_BOS['x'].to_numpy()
y = df_BOS['y'].to_numpy()
u = df_BOS['x-displacement'].to_numpy()
v = df_BOS['y-displacement'].to_numpy()

# -------------------------------------------------
# Build working DataFrame
# -------------------------------------------------
data = pd.DataFrame({
    "x": x,
    "y": y,
    "ux": u,
    "uy": v
})

data = data.sort_values(["x", "y"]).reset_index(drop=True)

# Unique x-columns
x_unique = np.sort(data["x"].unique())

# Fractions of local nozzle half-height
fractions = [0.00, 0.25, 0.50, 0.75, 1.00]

# tolerance on y/y_max(x)
tol = 0.01

# Store results
results = {f: {"x": [], "y_target": [], "ux": [], "uy": []} for f in fractions}

# -------------------------------------------------
# Sample along y / y_max(x) = const
# using upper half only (y >= 0)
# -------------------------------------------------
for x0 in x_unique:
    col_all = data[np.isclose(data["x"], x0)].copy().sort_values("y")
    col_upper = col_all[col_all["y"] >= 0].copy().sort_values("y")

    if len(col_all) < 2:
        continue

    y_all = col_all["y"].to_numpy()
    ux_all = col_all["ux"].to_numpy()
    uy_all = col_all["uy"].to_numpy()

    if len(col_upper) < 2:
        continue

    y_col = col_upper["y"].to_numpy()
    ux_col = col_upper["ux"].to_numpy()
    uy_col = col_upper["uy"].to_numpy()

    y_max_local = np.max(y_col)

    if y_max_local <= 0:
        continue

    for f in fractions:
        if f == 0.0:
            if 0.0 < y_all.min() or 0.0 > y_all.max():
                continue

            y_target = 0.0
            ux_target = np.interp(0.0, y_all, ux_all)
            uy_target = np.interp(0.0, y_all, uy_all)

        else:
            y_target = f * y_max_local

            if y_target < y_col.min() or y_target > y_col.max():
                continue

            ux_target = np.interp(y_target, y_col, ux_col)
            uy_target = np.interp(y_target, y_col, uy_col)

        results[f]["x"].append(x0)
        results[f]["y_target"].append(y_target)
        results[f]["ux"].append(ux_target)
        results[f]["uy"].append(uy_target)

# -------------------------------------------------
# Plot displacement vs x
# -------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for f in fractions:
    x_plot = np.array(results[f]["x"])
    ux_plot = np.array(results[f]["ux"])
    uy_plot = np.array(results[f]["uy"])

    idx = np.argsort(x_plot)
    x_plot = x_plot[idx]
    ux_plot = ux_plot[idx]
    uy_plot = uy_plot[idx]

    label = f"y / y_max(x) = {f:.2f}"

    axes[0].plot(x_plot, ux_plot, label=label)
    axes[1].plot(x_plot, uy_plot, label=label)

axes[0].set_ylabel("x-displacement")
axes[0].set_title("x-displacement vs x")
axes[0].grid(True)
axes[0].legend()

axes[1].set_ylabel("y-displacement")
axes[1].set_xlabel("x")
axes[1].set_title("y-displacement vs x")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Optional: show the actual sampling curves in the nozzle
# -------------------------------------------------
plt.figure(figsize=(10, 4))
plt.scatter(x, y, s=2, color='lightgray', label='BOS grid')

for f in fractions:
    x_plot = np.array(results[f]["x"])
    y_plot = np.array(results[f]["y_target"])

    idx = np.argsort(x_plot)
    plt.plot(x_plot[idx], y_plot[idx], label=f"y / y_max(x) = {f:.2f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Sampling curves used for displacement extraction")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


rows = []

for f in fractions:
    for i in range(len(results[f]["x"])):
        rows.append({
            "fraction": f,
            "x": results[f]["x"][i],
            "y": results[f]["y_target"][i],
            "ux": results[f]["ux"][i],
            "uy": results[f]["uy"][i]
        })

df_out = pd.DataFrame(rows)
df_out.to_csv("CC_streamline/upper_results.csv", index=False)
