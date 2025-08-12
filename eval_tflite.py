# eval_tflite.py
import os
import numpy as np
from absl import app, flags
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

import dataloader  # 复用你的 TFRecord 读取

FLAGS = flags.FLAGS
flags.DEFINE_string('tflite_path', None, 'Path to .tflite model.')
flags.DEFINE_string('test_files', None, 'TFRecord file pattern.')
flags.DEFINE_integer('num_classes', None, 'Number of classes.')
flags.DEFINE_integer('batch_size', 32, 'Dataset batch size (推理仍逐样本喂给 TFLite).')
flags.DEFINE_integer('input_size', 224, 'Model input size (HxW).')
flags.DEFINE_string('results_file', None, 'Prefix to save results (no extension).')
flags.DEFINE_integer('log_frequence', 500, 'Log every n steps.')
flags.mark_flag_as_required('tflite_path')
flags.mark_flag_as_required('test_files')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('results_file')

def build_dataset():
    """使用你项目的 dataloader 构建测试数据集。"""
    input_data = dataloader.TFRecordWBBoxInputProcessor(
        file_pattern=FLAGS.test_files,
        batch_size=FLAGS.batch_size,
        is_training=False,
        output_size=FLAGS.input_size,
        num_classes=FLAGS.num_classes,
        num_instances=0,
        provide_instance_id=True,        # 需要拿到 label 与 id
        provide_coordinates_input=False  # .tflite 一般无坐标分支
    )
    dataset, _, _ = input_data.make_source_dataset()
    return dataset

def _get_io_quant_params(in_det):
    """从 input_details 里尽可能稳健地取出量化参数（仅当 int8 输入）。"""
    # TFLite 可能把量化信息放在两处：quantization_parameters 或 quantization
    scales = []
    zero_points = []
    if 'quantization_parameters' in in_det:
        qp = in_det['quantization_parameters']
        scales = list(qp.get('scales', []))
        zero_points = list(qp.get('zero_points', []))
    if (not scales) and ('quantization' in in_det):
        q = in_det['quantization']
        if isinstance(q, (list, tuple)) and len(q) == 2 and q[0] not in (None, 0.0):
            scales = [q[0]]
            zero_points = [q[1]]
    return scales, zero_points

def main(_):
    # 1) 加载 TFLite 模型
    num_threads = max(1, (os.cpu_count() or 1) // 2)
    interpreter = Interpreter(model_path=FLAGS.tflite_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_idx  = in_det['index']
    out_idx = out_det['index']
    in_dtype = in_det['dtype']
    out_dtype = out_det['dtype']

    # 仅当输入类型是 int8 时读取量化参数
    in_scale = None
    in_zero  = None
    has_quant = False
    if in_dtype == np.int8:
        scales, zero_points = _get_io_quant_params(in_det)
        if not scales:
            raise ValueError(
                "Input dtype is int8 but no quantization parameters found. "
                "这通常说明模型不是全整型 I/O，请检查导出的 tflite。"
            )
        in_scale = float(scales[0])
        in_zero  = int(zero_points[0])
        has_quant = True

    # 打印 I/O 概况
    print("==== TFLite Model I/O ====")
    print(f"Input  name: {in_det.get('name')}, shape: {in_det.get('shape')}, dtype: {in_dtype}")
    print(f"Output name: {out_det.get('name')}, shape: {out_det.get('shape')}, dtype: {out_dtype}")
    print(f"Quantized input: {has_quant}  (scale={in_scale}, zero_point={in_zero})" if has_quant else "Quantized input: False")
    print(f"Using XNNPACK/CPU threads: {num_threads}")
    print("==========================")

    # 2) 构建测试集
    ds = build_dataset()

    y_true, y_pred = [], []
    count = 0

    # 3) 逐 batch 迭代；对 TFLite 建议逐样本喂入
    for batch, meta in ds:
        # 你的 dataloader 通常返回 (label, instance_id)
        label, instance_id = meta
        imgs = batch.numpy().astype(np.float32)     # 假设 dataloader 已做与训练一致的预处理（如 /255）
        labels = tf.argmax(label, axis=1).numpy()

        for i in range(imgs.shape[0]):
            x = imgs[i][np.newaxis, ...]  # [1, H, W, C]
            if in_dtype == np.float32:
                input_data = x
            elif in_dtype == np.int8:
                # 量化到 int8：注意要与代表性数据集/训练前处理保持一致尺度
                input_data = np.round(x / in_scale + in_zero).astype(np.int8)
            else:
                raise ValueError(f"Unsupported input dtype: {in_dtype}")

            interpreter.set_tensor(in_idx, input_data)
            interpreter.invoke()
            out = interpreter.get_tensor(out_idx)[0]  # [num_classes]
            pred = int(np.argmax(out))                # argmax 不需要反量化
            y_pred.append(pred)

        y_true += list(labels)

        count += 1
        if count % FLAGS.log_frequence == 0:
            print(f"Finished eval step {count}")

    # 4) 评测与保存
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    os.makedirs(os.path.dirname(FLAGS.results_file), exist_ok=True)
    with open(f"{FLAGS.results_file}.accuracy", "w") as f:
        f.write(str(acc))
    with open(f"{FLAGS.results_file}.conf_matrix", "w") as f:
        f.write(str(cm))
    with open(f"{FLAGS.results_file}.classification_report", "w") as f:
        f.write(rep)

    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    app.run(main)
