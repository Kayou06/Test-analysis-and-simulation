import numpy as np
import pandas as pd




for i in range(1, 8):
    data = np.load(f"CC Data/displacement_vectors{i}.npy")
    df = pd.DataFrame(data, columns=["x", "y", "x-displacement", "y-displacement"])
    df.to_csv(f"CC Data/displacement_vectors{i}.csv", index=False)
