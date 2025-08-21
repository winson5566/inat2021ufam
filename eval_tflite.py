# eval_tflite.py
import os
import time
import numpy as np
from absl import app, flags
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

# 数据与模型
import dataloader
# 和训练一致：用训练脚本里的 FCNet 定义
from FCNetModel import FCNet

FLAGS = flags.FLAGS

# ---------- 基础评估参数 ----------
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

# ---------- Geo Prior 相关参数（传入 geo_prior_ckpt_dir 即启用） ----------
flags.DEFINE_string('geo_prior_ckpt_dir', None, 'Directory of geo prior checkpoints')
flags.DEFINE_bool('use_bn_geo_prior', False, 'Use BatchNorm in geo prior (must match training)')
flags.DEFINE_integer('embed_dim', 256, 'Embedding dim for geo prior (must match training)')

flags.mark_flag_as_required('tflite_path')
flags.mark_flag_as_required('test_files')
flags.mark_flag_as_required('num_classes')

# ---------------- 工具函数 ----------------
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

def _mix_predictions(cnn_probs: np.ndarray, prior_probs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    论文原始融合：逐元素乘 + valid 门控
    """
    if valid.ndim == 1:
        valid = valid[:, None]
    return cnn_probs * prior_probs * valid + (1.0 - valid) * cnn_probs

def _predict_batch_top_k(interpreter, out_details, images: np.ndarray, k: int):
    """
    返回 (topk_indices, probs)，probs 为去量化后的模型输出（B,C），用于后续与 prior 融合。
    """
    B = images.shape[0]
    topk = np.empty((B, k), dtype=np.int64)
    probs = None
    for i in range(B):
        reallocated = _prepare_input(interpreter, images[i])
        # 如需更稳，可在 reallocated 时重新获取 out_details；多数固定输入模型不必
        interpreter.invoke()
        y = interpreter.get_tensor(out_details['index'])
        y = _maybe_dequantize_output(out_details, y)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        if probs is None:
            probs = np.empty((B, y.shape[-1]), dtype=np.float32)
        probs[i] = y.astype(np.float32, copy=False)
        topk[i] = np.argpartition(-y, range(k))[:k]
    return topk, probs

# ---------------- 数据集构建 ----------------
def build_test_dataset(include_geo_data: bool):
    """
    include_geo_data=True 时：
        ((image, prior_input), (label, valid))
    否则：
        (image, label)
    """
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
        provide_validity_info_output=include_geo_data,
        provide_coordinates_input=False,
        provide_coord_date_encoded_input=include_geo_data,
        batch_drop_remainder=False,
        seed=1234
    )
    dataset, _, _ = proc.make_source_dataset()
    return dataset

# 从数据里窥探 geo prior 向量真实维度
def _peek_prior_dim(dataset):
    for item in dataset.take(1):
        # ((image, prior_input), (label, valid))
        (images_tf, prior_input_tf), _ = item
        return int(prior_input_tf.shape[-1])
    raise RuntimeError("Empty dataset: cannot infer geo prior feature dimension.")

# ---------------- Geo Prior 模型加载（与训练同构） ----------------
def _load_geo_prior_model(prior_dim: int):
    if FLAGS.geo_prior_ckpt_dir is None:
        return None

    rand_sample_generator = dataloader.RandSpatioTemporalGenerator()
    model = FCNet(
        num_inputs=prior_dim,                 # ★ 和训练一致：用真实维度
        embed_dim=FLAGS.embed_dim,            # 默认 256；需与训练一致
        num_classes=FLAGS.num_classes,        # 必须一致（你的训练是 10000）
        rand_sample_generator=rand_sample_generator,
        num_users=0,                          # 训练脚本默认 use_photographers=False
        use_bn=FLAGS.use_bn_geo_prior         # 如训练时开了 BN，这里要 True
    )

    # 先 build：创建变量（Keras3 必须）
    dummy = tf.zeros((1, prior_dim), dtype=tf.float32)
    _ = model(dummy, training=False)

    ckpt_dir = FLAGS.geo_prior_ckpt_dir
    candidates = [
        os.path.join(ckpt_dir, "ckp.weights.h5"),  # 训练脚本明确保存了这个
        os.path.join(ckpt_dir, "ckp.keras"),
    ]
    if os.path.exists(os.path.join(ckpt_dir, "ckp.index")):
        candidates.append(os.path.join(ckpt_dir, "ckp"))  # TF Checkpoint 前缀

    last_err = None
    for p in candidates:
        if not os.path.exists(p):
            continue
        print(f"🔄 Loading geo prior weights from: {p}")
        try:
            model.load_weights(p)  # 严格加载（最佳）
            print("✅ Geo prior weights loaded (strict).")
            return model
        except Exception as e:
            print(f"[GeoPrior] strict load failed: {e}")
            last_err = e
            # 兜底：允许跳过不匹配，先让管线可跑；后续再把参数对齐到严格加载成功
            try:
                model.load_weights(p, skip_mismatch=True)
                print("⚠️ Geo prior weights loaded with skip_mismatch=True (some layers unmatched).")
                return model
            except Exception as e2:
                print(f"[GeoPrior] skip_mismatch load failed: {e2}")
                last_err = e2

    raise FileNotFoundError(
        f"No geo-prior weights could be loaded from {ckpt_dir}.\n"
        f"Tried: {', '.join(candidates)}\n"
        f"Last error: {last_err}"
    )

# ---------------- 主流程 ----------------
def main(_):
    use_geo = FLAGS.geo_prior_ckpt_dir is not None
    ds = build_test_dataset(include_geo_data=use_geo)

    assert os.path.exists(FLAGS.tflite_path), f"tflite not found: {FLAGS.tflite_path}"
    interpreter = Interpreter(model_path=FLAGS.tflite_path)
    interpreter.allocate_tensors()
    out_details = _select_output_details(interpreter)

    if use_geo:
        # 从数据里推断 prior 维度，然后据此构建并加载权重
        prior_dim = _peek_prior_dim(ds)
        print(f"[GeoPrior] inferred prior feature dim = {prior_dim}")
        geo_model = _load_geo_prior_model(prior_dim)
    else:
        geo_model = None

    correct_top1 = 0
    correct_top_k = 0
    total = 0
    t0 = time.time()

    for step, item in enumerate(ds):
        if use_geo:
            # ((images, prior_input), (labels, valid))
            (images_tf, prior_input_tf), (labels_tf, valid_tf) = item
            images_np = images_tf.numpy()
            labels_np = labels_tf.numpy()
            valid_np  = valid_tf.numpy().astype(np.float32)
            true_ids = labels_np.argmax(axis=1).astype(np.int64)

            # CNN (TFLite)
            pred_top_k, cnn_probs = _predict_batch_top_k(interpreter, out_details, images_np, k=FLAGS.top_k)

            # Geo Prior（确保 float32）
            prior_input_tf = tf.cast(prior_input_tf, tf.float32)
            prior_probs = geo_model(prior_input_tf, training=False).numpy().astype(np.float32)

            # 融合：乘法 + valid 门控
            fused_probs = _mix_predictions(cnn_probs, prior_probs, valid_np)

            # 基于融合后重新 top-k
            fused_topk = np.argpartition(-fused_probs, range(FLAGS.top_k), axis=1)[:, :FLAGS.top_k]

            # 指标
            correct_top1 += int((np.argmax(fused_probs, axis=1) == true_ids).sum())
            correct_top_k += sum(true in topk for true, topk in zip(true_ids, fused_topk))
            total += images_np.shape[0]
        else:
            images_np, labels_np = item[0].numpy(), item[1].numpy()
            true_ids = labels_np.argmax(axis=1).astype(np.int64)

            pred_top_k, _ = _predict_batch_top_k(interpreter, out_details, images_np, k=FLAGS.top_k)
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
        line = f"{os.path.basename(FLAGS.tflite_path)},acc@1={acc1:.6f},acc@{FLAGS.top_k}={acck:.6f},samples={total},secs={dt:.2f},geo={'on' if use_geo else 'off'}\n"
        os.makedirs(os.path.dirname(os.path.abspath(FLAGS.results_file)), exist_ok=True)
        with open(FLAGS.results_file, "a", encoding="utf-8") as f:
            f.write(line)

if __name__ == "__main__":
    app.run(main)
