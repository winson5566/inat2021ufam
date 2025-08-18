# inspect_pb.py (兼容 Keras 3 SavedModel + TFSMLayer + dtype=str 问题)
import os
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

# 映射 dtype 字符串 到字节大小
DTYPE_SIZE = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "int8":    1,
    "uint8":   1,
    "int16":   2,
    "uint16":  2,
    "int32":   4,
    "uint32":  4,
    "int64":   8,
    "uint64":  8,
    "bool":    1,
    "complex64": 8,
}


def inspect_saved_model(saved_model_dir):
    print(f"📦 Inspecting SavedModel at: {saved_model_dir}")
    model = TFSMLayer(saved_model_dir, call_endpoint="serving_default")
    print("✅ Loaded SavedModel using TFSMLayer")

    weights = model.variables  # All internal tf.Variable tensors

    total_params = 0
    total_bytes = 0
    rows = []

    for var in weights:
        name = var.name
        shape = var.shape
        dtype = str(var.dtype).lower()
        param_count = np.prod(shape)

        if dtype not in DTYPE_SIZE:
            print(f"⚠️ Unsupported dtype: {dtype}, skipping {name}")
            continue

        mem_bytes = param_count * DTYPE_SIZE[dtype]

        rows.append({
            "Name": name,
            "Shape": str(tuple(shape)),
            "DType": dtype,
            "Param Count": param_count,
            "Memory (MB)": mem_bytes / (1024 * 1024)
        })

        total_params += param_count
        total_bytes += mem_bytes

    print("\n📋 Weights Summary:")
    for row in sorted(rows, key=lambda x: x["Param Count"], reverse=True):
        print(f"{row['Name']:<60} Shape: {row['Shape']:<20} DType: {row['DType']:<8} "
              f"Params: {row['Param Count']:,}  Memory: {row['Memory (MB)']:.2f} MB")

    print("\n🧠 Total Parameters     :", f"{total_params:,}")
    print("📦 Total Weights Memory:", f"{total_bytes / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inspect_pb.py <saved_model_dir>")
        sys.exit(1)
    inspect_saved_model(sys.argv[1])
