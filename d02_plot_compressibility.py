import matplotlib.pyplot as plt

rho0 = [40.2777, 75.0403, 139.006, 209.161, 91.6089, 59.8612, 41.4280]
Z = [0.7925, 0.6357, 0.4983, 0.3571, 0.6430, 0.7404, 0.8166]
labels = [1, 2, 3, 4, 5, 6, 7]

fig, ax = plt.subplots()

# Use a qualitative colour map
cmap = plt.get_cmap("tab10")

for i, (xi, yi, label) in enumerate(zip(Z, rho0, labels)):
    colour = cmap(i)

    # Thin vertical line
    ax.vlines(
        xi,
        ymin=0,
        ymax=yi,
        linewidth=0.8,
        color=colour,
        alpha=0.8
    )

    # Point
    ax.scatter(
        xi,
        yi,
        s=35,
        color=colour,
        zorder=3,
        label=f"Image {label}"
    )

    # Point label
    ax.annotate(
        label,
        xy=(xi, yi),
        xytext=(5, -8),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel(r"Compressibility Factor ($Z$)")
ax.set_ylabel(r"Density ($\rho_0$) [kg/m$^3$]")
ax.set_title("Density vs Compressibility Factor for Different Configurations")
ax.grid(True, alpha=0.3)

# Legend
ax.legend(
    title="Configurations",
    loc="best",
    fontsize=8
)

fig.savefig("compressibility_factor_image2.png", dpi=300, bbox_inches="tight")
plt.show()
