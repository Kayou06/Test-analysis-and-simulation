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
    num_fracs = len(fractions)

    # -------------------------------------------------
    # Dynamically create a grid: Rows = num_fracs, Columns = 2
    # -------------------------------------------------
    # figsize scales automatically so the graphs don't get squished
    fig, axes = plt.subplots(nrows=num_fracs, ncols=2, figsize=(12, 4 * num_fracs), sharex=True)

    # Edge case: If there's only 1 fraction, wrap axes in a list so the indexing still works
    if num_fracs == 1:
        axes = np.array([axes])

    # Loop through the fractions, keeping track of the row index (row_idx)
    for row_idx, f in enumerate(fractions):
        subset = merged[merged["fraction"] == f]

        x = subset["x"].to_numpy()
        idx = np.argsort(x)
        x_sorted = x[idx]

        # Extract raw upper and lower data
        ux_up = subset["ux_upper"].to_numpy()[idx]
        uy_up = subset["uy_upper"].to_numpy()[idx]
        
        ux_low = subset["ux_lower"].to_numpy()[idx]
        uy_low = subset["uy_lower"].to_numpy()[idx]

        # Target the specific subplots for this row
        ax_x = axes[row_idx, 0] # Left column: X-displacement
        ax_y = axes[row_idx, 1] # Right column: Y-displacement

        # Plot Upper (Dashed line)
        ax_x.plot(x_sorted, ux_up, '--', linewidth=2, label=f"Upper {f}")
        ax_y.plot(x_sorted, uy_up, '--', linewidth=2, label=f"Upper {f}")

        # Plot Lower (Dotted line)
        ax_x.plot(x_sorted, ux_low, ':', linewidth=2, label=f"Lower {f}")
        ax_y.plot(x_sorted, uy_low, ':', linewidth=2, label=f"Lower {f}")

        # Format the X-displacement plot (Left Column)
        ax_x.set_ylabel("x-displacement")
        ax_x.set_title(f"X-Displacements (y/y_max = ±{f})")
        ax_x.grid(True)
        ax_x.legend()

        # Format the Y-displacement plot (Right Column)
        ax_y.set_ylabel("y-displacement")
        ax_y.set_title(f"Y-Displacements (y/y_max = ±{f})")
        ax_y.grid(True)
        ax_y.legend()

        # Add x-axis labels ONLY to the very bottom row to keep it clean
        if row_idx == num_fracs - 1:
            ax_x.set_xlabel("x")
            ax_y.set_xlabel("x")

    # Draw the final master grid outside the loop!
    plt.tight_layout()
    plt.show()
        
compare_streamlines(upper_csv_path= f"CC_streamline/upper_results_1.csv",
     lower_csv_path=f"CC_streamline/lower_results_1.csv"
 )
