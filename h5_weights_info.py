import h5py
import numpy as np
import os
import pandas as pd

# Path to weights file
weights_path = "model/model_efficientnet_b0/ckp.weights.h5"

# File size on disk
file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

rows = []
total_params = 0
total_memory_bytes = 0

with h5py.File(weights_path, 'r') as f:
    def collect_attrs(name, obj):
        global total_params, total_memory_bytes
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            params_count = int(np.prod(shape))
            dtype = str(obj.dtype)
            dtype_size = np.dtype(obj.dtype).itemsize  # bytes per parameter
            memory_bytes = params_count * dtype_size

            total_params += params_count
            total_memory_bytes += memory_bytes

            rows.append({
                "Layer Path": name,
                "Shape": str(shape),
                "Param Count": params_count,
                "Data Type": dtype,
                "Memory (MB)": memory_bytes / (1024 * 1024)
            })

    f.visititems(collect_attrs)

# Create DataFrame and sort
df = pd.DataFrame(rows)
df = df.sort_values(by="Param Count", ascending=False).reset_index(drop=True)

# Print table
print(f"Weight file: {weights_path}")
print(f"File size on disk: {file_size_mb:.2f} MB\n")
print(df.to_string(index=False))
print(f"\nTotal parameters: {total_params:,}")
print(f"Total memory usage: {total_memory_bytes / (1024 * 1024):.2f} MB (based on actual dtype)")
