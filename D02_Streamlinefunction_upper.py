import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def streamline_upper(
    csv_path,
    output_csv_path,
    fractions=None,
    make_plots=True
):
    """
    Extract BOS displacement values along curves defined by y / y_max(x) = const
    using the upper half of the nozzle only (y >= 0).

    Parameters
    ----------
    csv_path : str
        Path to input CSV file.
    output_csv_path : str or None
        Path to save output CSV. If None, no file is saved.
    fractions : list[float] or None
        Fractions of local half-height to sample.
        Default = [0.00, 0.25, 0.50, 0.75, 1.00]
    make_plots : bool
        If True, generate plots.

    Returns
    -------
    results : dict
        Dictionary with sampled results for each fraction.
    df_out : pandas.DataFrame
        Flattened output table with columns:
        fraction, x, y, ux, uy
    """

    if fractions is None:
        fractions = [0.00, 0.25, 0.50, 0.75, 1.00]

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    df_BOS = pd.read_csv(csv_path, delimiter=",")

    # -------------------------------------------------
    # Extract raw BOS data
    # -------------------------------------------------
    x = df_BOS["x"].to_numpy()
    y = df_BOS["y"].to_numpy()
    u = df_BOS["x-displacement"].to_numpy()
    v = df_BOS["y-displacement"].to_numpy()

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
    # Make output DataFrame
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Save CSV if requested
    # -------------------------------------------------
    if output_csv_path is not None:
        df_out.to_csv(output_csv_path, index=False)

    # -------------------------------------------------
    # Plot displacement vs x
    # -------------------------------------------------
    if make_plots:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for f in fractions:
            x_plot = np.array(results[f]["x"])
            ux_plot = np.array(results[f]["ux"])
            uy_plot = np.array(results[f]["uy"])

            if len(x_plot) == 0:
                continue

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
        # Plot actual sampling curves in nozzle
        # -------------------------------------------------
        plt.figure(figsize=(10, 4))
        plt.scatter(x, y, s=2, color="lightgray", label="BOS grid")

        for f in fractions:
            x_plot = np.array(results[f]["x"])
            y_plot = np.array(results[f]["y_target"])

            if len(x_plot) == 0:
                continue

            idx = np.argsort(x_plot)
            plt.plot(x_plot[idx], y_plot[idx], label=f"y / y_max(x) = {f:.2f}")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Sampling curves used for displacement extraction")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

    return results, df_out