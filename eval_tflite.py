import os
import time
import numpy as np
from absl import app, flags
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

# 复用 TFRecord 读取逻辑
import dataloader

FLAGS = flags.FLAGS
flags.DEFINE_string('tflite_path', None, 'Path to .tflite model.')
flags.DEFINE_string('test_files', None, 'TFRecord file pattern')
flags.DEFINE_integer('num_classes', None, 'Number of classes.')
flags.DEFINE_integer('batch_size', 64, 'Dataset batch size.')
flags.DEFINE_integer('input_size', 224, 'Model input size (HxW).')
flags.DEFINE_integer('log_frequence', 500, 'Log every n samples.')
flags.DEFINE_string('results_file', None, 'Append summary line to this path.')
flags.DEFINE_enum('input_norm', 'none', ['none', 'minus1_1', 'imagenet'],
                  'Input normalization for float32 input tensors.')
flags.DEFINE_integer('output_index', 0, 'Use this output index if > -1.')
flags.DEFINE_string('output_name_contains', None, 'If set, pick first output whose name contains this substring.')
flags.DEFINE_integer('top_k', 5, 'Top-K accuracy to compute.')

flags.mark_flag_as_required('tflite_path')
flags.mark_flag_as_required('test_files')
flags.mark_flag_as_required('num_classes')

def _maybe_dequantize_output(output_details, y):
    y = y.astype(np.float32, copy=False)
    qparams = output_details.get("quantization_parameters", {})
    scales = qparams.get("scales", None)
    zero_points = qparams.get("zero_points", None)
    quantized_dimension = qparams.get("quantized_dimension", None)
    if scales is not None and len(scales) > 0:
        if len(scales) == 1:
            s = float(scales[0])
            z = float(zero_points[0]) if zero_points is not None and len(zero_points) > 0 else 0.0
            if s != 0.0:
                y = s * (y - z)
            return y
        else:
            axis = int(quantized_dimension if quantized_dimension is not None else -1)
            s = np.asarray(scales, dtype=np.float32)
            z = np.asarray(zero_points if zero_points is not None else np.zeros_like(s), dtype=np.float32)
            shape = [1] * y.ndim
            shape[axis] = s.shape[0]
            return s.reshape(shape) * (y - z.reshape(shape))
    if 'quantization' in output_details and output_details['quantization'] is not None:
        scale, zero_point = output_details['quantization']
        if scale not in (None, 0.0):
            y = scale * (y - float(zero_point))
    return y

def _apply_input_norm_f32(x_f32_0_255):
    mode = FLAGS.input_norm
    if mode == 'minus1_1':
        x = x_f32_0_255 / 255.0
        return x * 2.0 - 1.0
    elif mode == 'imagenet':
        x = x_f32_0_255 / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x - mean) / std
    return x_f32_0_255

def _fix_to_uint8(x_np: np.ndarray) -> np.ndarray:
    if x_np.dtype.kind == 'f':
        if float(x_np.max()) <= 1.0 and float(x_np.min()) >= 0.0:
            x_np = np.round(x_np * 255.0)
        x_np = np.clip(x_np, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        x_np = np.clip(x_np, 0, 255).astype(np.uint8, copy=False)
    return x_np

def _prepare_input(interpreter, x_np: np.ndarray):
    input_details = interpreter.get_input_details()[0]
    in_index = input_details['index']
    in_dtype = input_details['dtype']
    wanted_shape = list(input_details['shape'])

    x_u8 = _fix_to_uint8(x_np)
    target_shape = [1, x_u8.shape[0], x_u8.shape[1], x_u8.shape[2]]
    if wanted_shape != target_shape:
        interpreter.resize_tensor_input(in_index, target_shape, strict=False)
        interpreter.allocate_tensors()

    if in_dtype == np.uint8:
        x_for_model = x_u8[None, ...]
    elif in_dtype == np.float32:
        x_f32 = x_u8.astype(np.float32, copy=False)
        x_f32 = _apply_input_norm_f32(x_f32)
        x_for_model = x_f32[None, ...]
    elif in_dtype == np.int8:
        scale, zero_point = input_details.get('quantization', (None, None))
        if scale in (None, 0.0):
            x_q = x_u8.astype(np.int32) - 128
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        else:
            x_q = np.round(x_u8.astype(np.float32) / float(scale) + float(zero_point))
            x_q = np.clip(x_q, -128, 127).astype(np.int8, copy=False)
        x_for_model = x_q[None, ...]
    else:
        x_f32 = x_u8.astype(np.float32, copy=False)
        x_f32 = _apply_input_norm_f32(x_f32)
        x_for_model = x_f32[None, ...]
    interpreter.set_tensor(in_index, x_for_model)

def _select_output_details(interpreter):
    outs = interpreter.get_output_details()
    if FLAGS.output_name_contains:
        key = FLAGS.output_name_contains.lower()
        for od in outs:
            if key in od.get('name', '').lower():
                return od
    if FLAGS.output_index is not None and 0 <= FLAGS.output_index < len(outs):
        return outs[FLAGS.output_index]
    return outs[0]

def _predict_batch_top_k(interpreter, out_details, images: np.ndarray, k: int):
    preds_top_k = np.empty((images.shape[0], k), dtype=np.int64)
    for i in range(images.shape[0]):
        _prepare_input(interpreter, images[i])
        interpreter.invoke()
        y = interpreter.get_tensor(out_details['index'])
        y = _maybe_dequantize_output(out_details, y)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        preds_top_k[i] = np.argpartition(-y, range(k))[:k]
    return preds_top_k

def build_test_dataset():
    proc = dataloader.TFRecordWBBoxInputProcessor(
        file_pattern=FLAGS.test_files,
        batch_size=FLAGS.batch_size,
        num_classes=FLAGS.num_classes,
        num_instances=0,
        is_training=False,
        use_eval_preprocess=True,
        output_size=FLAGS.input_size,
        resize_with_pad=False,
        provide_instance_id=False,
        provide_validity_info_output=False,
        provide_coordinates_input=False,
        provide_coord_date_encoded_input=False,
        batch_drop_remainder=False,
        seed=1234
    )
    dataset, _, _ = proc.make_source_dataset()
    return dataset

def main(_):
    ds = build_test_dataset()
    assert os.path.exists(FLAGS.tflite_path), f"tflite not found: {FLAGS.tflite_path}"
    interpreter = Interpreter(model_path=FLAGS.tflite_path)
    interpreter.allocate_tensors()
    out_details = _select_output_details(interpreter)

    correct_top1 = 0
    correct_top_k = 0
    total = 0
    t0 = time.time()

    for step, batch in enumerate(ds):
        images_np = batch[0].numpy()
        labels_np = batch[1].numpy()
        true_ids = labels_np.argmax(axis=1).astype(np.int64)

        pred_top_k = _predict_batch_top_k(interpreter, out_details, images_np, k=FLAGS.top_k)
        correct_top1 += int((pred_top_k[:, 0] == true_ids).sum())
        correct_top_k += sum(true in topk for true, topk in zip(true_ids, pred_top_k))
        total += images_np.shape[0]

        if FLAGS.log_frequence and total % FLAGS.log_frequence < images_np.shape[0]:
            acc1 = correct_top1 / total if total else 0.0
            acck = correct_top_k / total if total else 0.0
            print(f"[{total:6d} samples] acc@1 = {acc1:.4f} | acc@{FLAGS.top_k} = {acck:.4f}")

    dt = time.time() - t0
    acc1 = correct_top1 / total if total else 0.0
    acck = correct_top_k / total if total else 0.0
    print("=" * 60)
    print(f"Evaluated samples : {total}")
    print(f"Top-1 Accuracy    : {acc1:.6f}")
    print(f"Top-{FLAGS.top_k} Accuracy  : {acck:.6f}")
    print(f"Elapsed time (s)  : {dt:.2f}")
    print(f"Avg samples/sec   : {total / dt:.2f}" if dt > 0 else "Avg samples/sec   : n/a")

    if FLAGS.results_file:
        line = f"{os.path.basename(FLAGS.tflite_path)},acc@1={acc1:.6f},acc@{FLAGS.top_k}={acck:.6f},samples={total},secs={dt:.2f}\n"
        os.makedirs(os.path.dirname(os.path.abspath(FLAGS.results_file)), exist_ok=True)
        with open(FLAGS.results_file, "a", encoding="utf-8") as f:
            f.write(line)

if __name__ == "__main__":
    app.run(main)
