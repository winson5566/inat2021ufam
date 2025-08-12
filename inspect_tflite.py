# inspect_tflite.py
import os
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None

# 兼容不同的 tflite flatbuffer Python 代码结构
try:
    # 情况A：from tflite import Model 返回的是“类”
    from tflite import Model as TFLiteModel
    from tflite import TensorType
    MODEL_IS_CLASS = True
except Exception:
    # 情况B：from tflite import Model 返回的是“模块”，类名也叫 Model
    from tflite import Model as TFLiteModelModule
    from tflite import TensorType
    MODEL_IS_CLASS = False

DTYPE_SIZE = {
    TensorType.FLOAT32: 4,
    TensorType.FLOAT16: 2,
    TensorType.FLOAT64: 8,
    TensorType.INT8:    1,
    TensorType.UINT8:   1,
    TensorType.INT16:   2,
    TensorType.UINT16:  2,
    TensorType.INT32:   4,
    TensorType.UINT32:  4,
    TensorType.INT64:   8,
    TensorType.UINT64:  8,
    TensorType.BOOL:    1,
    TensorType.COMPLEX64: 8,
}

TYPE_NAME = {getattr(TensorType, k): k for k in dir(TensorType) if k.isupper()}

def tensor_shape(tensor):
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

def get_root_model(buf):
    """同时兼容两种导入风格，返回 flatbuffer 根模型对象。"""
    if MODEL_IS_CLASS:
        return TFLiteModel.GetRootAsModel(buf, 0)
    else:
        return TFLiteModelModule.Model.GetRootAsModel(buf, 0)

def main(tflite_path: str):
    with open(tflite_path, "rb") as f:
        buf = f.read()

    model = get_root_model(buf)

    if model.SubgraphsLength() == 0:
        raise RuntimeError("No subgraphs found in TFLite model.")
    subgraph = model.Subgraphs(0)

    rows = []
    total_params = 0
    total_bytes = 0

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        name = tensor.Name().decode("utf-8") if tensor.Name() else f"tensor_{i}"
        ttype = tensor.Type()
        dtype = TYPE_NAME.get(ttype, str(ttype))
        shape = tensor_shape(tensor)
        buf_idx = tensor.Buffer()

        # 取对应的 buffer；注意 API 为 model.Buffers(i)
        if buf_idx < 0 or buf_idx >= model.BuffersLength():
            continue
        buf_obj = model.Buffers(buf_idx)

        # **用 DataLength() 拿字节长度**，更稳妥
        nbytes = 0
        if hasattr(buf_obj, "DataLength"):
            nbytes = int(buf_obj.DataLength())
        else:
            # 极少数打包没有 DataLength，则尝试 DataAsNumpy()
            try:
                data = buf_obj.DataAsNumpy()
                nbytes = int(data.nbytes) if hasattr(data, "nbytes") else int(len(data))
            except Exception:
                nbytes = 0

        # 只统计“有数据”的常量张量（权重/常量），忽略中间激活/占位
        if nbytes > 0 and ttype in DTYPE_SIZE:
            itemsize = DTYPE_SIZE[ttype]
            # 从字节数反推参数量（兼容不同布局/稀疏情况）
            params_cnt = nbytes // itemsize

            rows.append({
                "Tensor Name": name,
                "Shape": str(tuple(shape)),
                "DType": dtype,
                "Param Count": params_cnt,
                "Memory (MB)": nbytes / (1024 * 1024),
            })
            total_params += params_cnt
            total_bytes += nbytes

    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

    print(f"TFLite file: {tflite_path}")
    print(f"File size on disk: {file_size_mb:.2f} MB\n")

    if pd is not None and rows:
        df = pd.DataFrame(rows).sort_values("Param Count", ascending=False).reset_index(drop=True)
        df["Param Count"] = df["Param Count"].map(lambda x: f"{x:,}")
        df["Memory (MB)"] = df["Memory (MB)"].map(lambda x: f"{x:.2f}")
        print(df.to_string(index=False))
    else:
        rows_sorted = sorted(rows, key=lambda r: r["Param Count"], reverse=True)
        for r in rows_sorted:
            print(f"{r['Tensor Name']}\n  Shape: {r['Shape']}  DType: {r['DType']}"
                  f"  Params: {r['Param Count']:,}  Memory: {r['Memory (MB)']:.2f} MB")

    print(f"\nTotal parameters (by bytes/dtype): {total_params:,}")
    print(f"Total constant memory: {total_bytes / (1024 * 1024):.2f} MB")
    if abs(total_bytes/(1024*1024) - file_size_mb) > 1.0:
        print("(Note: file size != constant memory; 文件还包含算子/图结构/量化表等额外开销)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)
    main(sys.argv[1])
