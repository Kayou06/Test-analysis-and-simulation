import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def compare_streamlines(
    upper_csv_path,
    lower_csv_path
):

    # -------------------------------------------------
    # Load results
    # -------------------------------------------------
    upper = pd.read_csv(upper_csv_path)
    lower = pd.read_csv(lower_csv_path)

    # -------------------------------------------------
    # Merge datasets (match same x and fraction)
    # -------------------------------------------------
    merged = pd.merge(
        upper,
        lower,
        on=["fraction", "x"],
        suffixes=("_upper", "_lower")
    )

    # -------------------------------------------------
    # Compute averaged fields
    # -------------------------------------------------
    # merged["ux_avg"] = (merged["ux_upper"] + merged["ux_lower"]) / 2
    # merged["uy_avg"] = (merged["uy_upper"] - merged["uy_lower"]) / 2

    fractions = sorted(merged["fraction"].unique())

    # -------------------------------------------------
    # Plot averaged displacement vs x
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for f in fractions:
        
        if math.isclose(f, 0.5, rel_tol=1e-5):
            
            subset = merged[merged["fraction"] == f]

            x = subset["x"].to_numpy()
            idx = np.argsort(x)
            x_sorted = x[idx]

            # Extract raw upper and lower data
            ux_up = subset["ux_upper"].to_numpy()[idx]
            uy_up = subset["uy_upper"].to_numpy()[idx]
            
            ux_low = subset["ux_lower"].to_numpy()[idx]
            uy_low = subset["uy_lower"].to_numpy()[idx]

            # Plot Upper (Dashed line)
            axes[0].plot(x_sorted, ux_up, '--', linewidth=2, label=f"Upper {f} Raw")
            axes[1].plot(x_sorted, uy_up, '--', linewidth=2, label=f"Upper {f} Raw")

            # Plot Lower (Dotted line)
            axes[0].plot(x_sorted, ux_low, ':', linewidth=2, label=f"Lower {f} Raw")
            # Note: You might want to plot -uy_low if you want it to visually mirror the upper y-displacement
            axes[1].plot(x_sorted, uy_low, ':', linewidth=2, label=f"Lower {f} Raw")

    axes[0].set_ylabel("x-displacement")
    axes[0].set_title("Streamline Displacements (y/y_max = ±0.5)")
    axes[0].grid(True)
    axes[0].legend()

    # Formatting Bottom Plot
    axes[1].set_ylabel("y-displacement")
    axes[1].set_xlabel("x")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
