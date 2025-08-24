# eval_h5.py
import os
import time
import importlib
import numpy as np
from absl import app, flags
import tensorflow as tf

# 数据与模型
import dataloader
from FCNetModel import FCNet  # Geo Prior

FLAGS = flags.FLAGS

# ---------- 基础评估参数 ----------
flags.DEFINE_string('model_path', None,
                    'Path to model directory/file. '
                    'Can be: a directory containing export/ or ckp.weights.h5, '
                    'a SavedModel directory, a .keras file, or a full .h5 model.')
flags.DEFINE_string('weights_filename', 'ckp.weights.h5',
                    'When model_path is a directory without export/, load this weights file inside it.')
flags.DEFINE_string('model_builder_module', 'models',
                    'Python module that contains `create()` (your builder shown above).')
flags.DEFINE_string('model_name', None,
                    'Model name for builder.create(), e.g., efficientnet-b0 / mobilenet-v3. '
                    'If not set, will try to infer from model_path.')
flags.DEFINE_integer('num_classes', None, 'Number of classes.')
flags.DEFINE_integer('input_size', 224, 'Model input size (HxW).')
flags.DEFINE_integer('batch_size', 64, 'Dataset batch size.')
flags.DEFINE_string('test_files', None, 'TFRecord file pattern.')
flags.DEFINE_integer('log_frequence', 500, 'Log every n samples.')
flags.DEFINE_string('results_file', None, 'Append summary line to this path.')
flags.DEFINE_enum('input_norm', 'none', ['none', 'minus1_1', 'imagenet'],
                  'Input normalization for float32 input tensors.')
flags.DEFINE_integer('top_k', 5, 'Top-K accuracy to compute.')

flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('test_files')

# ---------- Geo Prior 相关参数 ----------
flags.DEFINE_string('geo_prior_ckpt_dir', None, 'Directory of geo prior checkpoints')
flags.DEFINE_bool('use_bn_geo_prior', False, 'Use BatchNorm in geo prior (must match training)')
flags.DEFINE_integer('embed_dim', 256, 'Embedding dim for geo prior (must match training)')

# ---------------- 工具函数 ----------------
def _apply_input_norm_f32(x_f32):
    """
    与 TFLite 对齐：
    - 若原始是 [0,1] 且 input_norm='none'，转换为 [0,255] 再喂模型（和 eval_tflite.py 一致）
    - 若原始是 [0,255]，保持不变
    - 'minus1_1' / 'imagenet' 按各自规范处理
    """
    x = x_f32.astype(np.float32, copy=False)
    x_max = float(np.max(x)); x_min = float(np.min(x))

    # 判定原始量纲
    is_0_255 = (x_max > 1.5 or x_min < 0.0)

    mode = FLAGS.input_norm
    if mode == 'minus1_1':
        # 统一到 [0,1] 后再到 [-1,1]
        x = (x if not is_0_255 else np.clip(x, 0.0, 255.0) / 255.0)
        return x * 2.0 - 1.0
    elif mode == 'imagenet':
        x = (x if not is_0_255 else np.clip(x, 0.0, 255.0) / 255.0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x - mean) / std

    # mode == 'none'：与 TFLite 的 'none' 对齐
    if is_0_255:
        return np.clip(x, 0.0, 255.0)              # 已经是 0~255
    else:
        return np.clip(x * 255.0, 0.0, 255.0)      # 0~1 → 0~255



def _mix_predictions(cnn_probs: np.ndarray, prior_probs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.ndim == 1:
        valid = valid[:, None]
    return cnn_probs * prior_probs * valid + (1.0 - valid) * cnn_probs

_PEEKED = {'printed': False}  # 只打印一次调试信息
def _predict_batch_top_k(model, images_np: np.ndarray, k: int):
    x_raw = images_np.astype(np.float32, copy=False)
    if not _PEEKED['printed']:
        print(f"🔎 raw batch min/max: {float(x_raw.min()):.3f}/{float(x_raw.max()):.3f}")
    x = _apply_input_norm_f32(x_raw)
    if not _PEEKED['printed']:
        print(f"🔎 norm batch min/max: {float(x.min()):.3f}/{float(x.max()):.3f}")
        try:
            print(f"🧩 model inputs: {len(model.inputs)} | shapes: {[tuple(t.shape) for t in model.inputs]}")
        except Exception:
            pass
        _PEEKED['printed'] = True

    # 关键：很多自定义 builder 是 inputs=[image_input]
    y = model([x], training=False).numpy().astype(np.float32, copy=False)
    topk = np.argpartition(-y, range(k), axis=1)[:, :k]
    return topk, y


# ---------------- 数据集 ----------------
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

def _peek_prior_dim(dataset):
    for item in dataset.take(1):
        (images_tf, prior_input_tf), _ = item
        return int(prior_input_tf.shape[-1])
    raise RuntimeError("Empty dataset: cannot infer geo prior feature dimension.")

# ---------------- Geo Prior 模型加载 ----------------
def _load_geo_prior_model(prior_dim: int):
    if FLAGS.geo_prior_ckpt_dir is None:
        return None
    rand_sample_generator = dataloader.RandSpatioTemporalGenerator()
    model = FCNet(
        num_inputs=prior_dim,
        embed_dim=FLAGS.embed_dim,
        num_classes=FLAGS.num_classes,
        rand_sample_generator=rand_sample_generator,
        num_users=0,
        use_bn=FLAGS.use_bn_geo_prior
    )
    _ = model(tf.zeros((1, prior_dim), dtype=tf.float32), training=False)

    ckpt_dir = FLAGS.geo_prior_ckpt_dir
    candidates = [
        os.path.join(ckpt_dir, "ckp.weights.h5"),
        os.path.join(ckpt_dir, "ckp.keras"),
    ]
    if os.path.exists(os.path.join(ckpt_dir, "ckp.index")):
        candidates.append(os.path.join(ckpt_dir, "ckp"))

    last_err = None
    for p in candidates:
        if not os.path.exists(p):
            continue
        print(f"🔄 Loading geo prior weights from: {p}")
        try:
            model.load_weights(p)
            print("✅ Geo prior weights loaded (strict).")
            return model
        except Exception as e:
            print(f"[GeoPrior] strict load failed: {e}")
            last_err = e
            try:
                model.load_weights(p, skip_mismatch=True)
                print("⚠️ Geo prior weights loaded with skip_mismatch=True (some layers unmatched).")
                return model
            except Exception as e2:
                print(f"[GeoPrior] skip_mismatch load failed: {e2}")
                last_err = e2
    raise FileNotFoundError(f"No geo-prior weights in {ckpt_dir}. Tried {candidates}. Last error: {last_err}")

# ---------------- CNN 模型加载 ----------------
def _infer_model_name_from_path(path: str):
    base = os.path.basename(os.path.normpath(path)).lower()
    # 常见别名归一化
    base = base.replace('efficientnet_', 'efficientnet-').replace('mobilenet_', 'mobilenet-')
    # 例如 model_efficientnet_b0 -> efficientnet-b0
    if 'efficientnet-b0' in base: return 'efficientnet-b0'
    if 'efficientnet-b2' in base: return 'efficientnet-b2'
    if 'efficientnet-b3' in base: return 'efficientnet-b3'
    if 'efficientnet-b4' in base: return 'efficientnet-b4'
    if 'mobilenet-v2'   in base: return 'mobilenet-v2'
    if 'mobilenet-v3'   in base: return 'mobilenet-v3'
    return None

def _load_full_model_from_path(p: str):
    # SavedModel 目录 / .keras / 完整 .h5
    try:
        model = tf.keras.models.load_model(p, compile=False)
        print(f"✅ Loaded full model from: {p}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load full model from {p}: {e}")

def _load_cnn_model():
    path = FLAGS.model_path
    if os.path.isdir(path):
        # 1) 目录下若有 export/ 则当作 SavedModel
        export_dir = os.path.join(path, 'export')
        if os.path.isdir(export_dir):
            try:
                return _load_full_model_from_path(export_dir)
            except Exception as e:
                print(f"[WARN] Failed to load SavedModel from {export_dir}: {e}")

        # 2) 否则找权重文件（默认 ckp.weights.h5），需要用你的 builder 构建骨架
        weights_path = os.path.join(path, FLAGS.weights_filename)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights file not found: {weights_path}")

        model_name = FLAGS.model_name or _infer_model_name_from_path(path)
        if not model_name:
            raise ValueError("--model_name not provided and cannot infer from model_path.")

        # 导入你的 builder.create()
        mod = importlib.import_module(FLAGS.model_builder_module)
        if not hasattr(mod, 'create'):
            raise AttributeError(f'Module "{FLAGS.model_builder_module}" has no function `create`.')
        builder_create = getattr(mod, 'create')

        print(f"🔧 Building model via {FLAGS.model_builder_module}.create(model_name='{model_name}')")
        # 关键：不加载 imagenet 以避免无谓下载；权重会覆盖
        model = builder_create(
            model_name=model_name,
            num_classes=FLAGS.num_classes,
            input_size=FLAGS.input_size,
            classifier_activation="softmax",
            unfreeze_layers=-1,               # 推理
            use_coordinates_inputs=False,     # 只喂图像；GeoPrior独立融合
            base_model_weights=None,          # 避免自动下载 imagenet
            seed=None
        )
        print(f"🔄 Loading CNN weights from: {weights_path}")
        try:
            model.load_weights(weights_path)
            print("✅ CNN weights loaded (strict).")
        except Exception as e:
            print(f"[CNN] strict load failed: {e}")
            print("   → Retrying with skip_mismatch=True ...")
            model.load_weights(weights_path, skip_mismatch=True)
            print("⚠️ CNN weights loaded with skip_mismatch=True (some layers unmatched).")
        return model

    # 3) 非目录：当作 .keras / SavedModel / 完整 .h5
    return _load_full_model_from_path(path)

# ---------------- 主流程 ----------------
def main(_):
    use_geo = FLAGS.geo_prior_ckpt_dir is not None
    ds = build_test_dataset(include_geo_data=use_geo)

    # CNN
    cnn_model = _load_cnn_model()

    # Geo Prior（可选）
    if use_geo:
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

            pred_top_k, cnn_probs = _predict_batch_top_k(cnn_model, images_np, k=FLAGS.top_k)

            prior_input_tf = tf.cast(prior_input_tf, tf.float32)
            prior_probs = geo_model(prior_input_tf, training=False).numpy().astype(np.float32)

            fused_probs = _mix_predictions(cnn_probs, prior_probs, valid_np)
            fused_topk = np.argpartition(-fused_probs, range(FLAGS.top_k), axis=1)[:, :FLAGS.top_k]

            correct_top1 += int((np.argmax(fused_probs, axis=1) == true_ids).sum())
            correct_top_k += sum(true in topk for true, topk in zip(true_ids, fused_topk))
            total += images_np.shape[0]
        else:
            images_np, labels_np = item[0].numpy(), item[1].numpy()
            true_ids = labels_np.argmax(axis=1).astype(np.int64)

            pred_top_k, _ = _predict_batch_top_k(cnn_model, images_np, k=FLAGS.top_k)
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
        line = f"{os.path.basename(FLAGS.model_path)},acc@1={acc1:.6f},acc@{FLAGS.top_k}={acck:.6f},samples={total},secs={dt:.2f},geo={'on' if use_geo else 'off'}\n"
        os.makedirs(os.path.dirname(os.path.abspath(FLAGS.results_file)), exist_ok=True)
        with open(FLAGS.results_file, "a", encoding="utf-8") as f:
            f.write(line)

if __name__ == "__main__":
    app.run(main)
