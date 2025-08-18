import os
import json
import numpy as np
from PIL import Image
from absl import app, flags
from tensorflow.lite.python.interpreter import Interpreter

FLAGS = flags.FLAGS
flags.DEFINE_string('tflite_model_path', None, 'Path to .tflite model file.')
flags.DEFINE_string('image_path', None, 'Path to input image.')
flags.DEFINE_string('categories_json', None, 'Path to categories.json.')
flags.DEFINE_integer('top_k', 5, 'Top-K predictions to display.')
flags.DEFINE_enum('input_norm', 'none', ['none', 'minus1_1', 'imagenet'],
                  'Float32 input normalization (must match training/export).')
flags.DEFINE_integer('output_index', 0, 'If >=0, pick this output.')
flags.DEFINE_string('output_name_contains', None, 'Pick first output whose name contains this substring.')
flags.DEFINE_boolean('center_crop_to_square', True, 'Center-crop to square before resize (match eval pipeline).')
flags.DEFINE_boolean('prefer_common_name', True, 'Prefer common_name when available for display.')

flags.mark_flag_as_required('tflite_model_path')
flags.mark_flag_as_required('image_path')
flags.mark_flag_as_required('categories_json')

# ----------------- helpers (aligned with your eval script) -----------------
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
# --------------------------------------------------------------------------

def load_categories(categories_json, prefer_common_name=True):
    """
    读取 inat2021/categories.json（列表，每项含 id/name/common_name/...）
    返回：
      - idx_to_name: index(int) -> 展示名（优先 common_name，否则 name）
      - idx_to_tax:  index(int) -> 'kingdom > phylum > class > order > family > genus [species]'
      - meta:        其它信息（id->原始字典）
    """
    with open(categories_json, 'r', encoding='utf-8') as f:
        cats = json.load(f)

    if not isinstance(cats, list):
        raise ValueError("categories.json 不是列表结构。")

    idx_to_name, idx_to_tax = {}, {}
    id_to_raw = {}

    for i, c in enumerate(cats):
        if not isinstance(c, dict):
            continue
        idx = int(c.get('id', i))
        sci_name = c.get('name', f'class_{idx}')
        com_name = c.get('common_name') or None
        show_name = (com_name if (prefer_common_name and com_name) else sci_name)

        parts = [
            c.get('kingdom', '?'),
            c.get('phylum', '?'),
            c.get('class', '?'),
            c.get('order', '?'),
            c.get('family', '?'),
            c.get('genus', '?')
        ]
        tax = " > ".join(parts)
        sp = c.get('specific_epithet')
        if sp:
            tax = f"{tax} > {sp}"

        idx_to_name[idx] = show_name
        idx_to_tax[idx] = tax
        id_to_raw[idx] = c

    meta = {"id_to_raw": id_to_raw}
    return idx_to_name, idx_to_tax, meta

def load_and_preprocess_image(path, size, center_crop=True):
    img = Image.open(path).convert('RGB')
    if center_crop:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top  = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img)  # 0..255 uint8
    return arr

def main(_):
    # Load TFLite
    assert os.path.exists(FLAGS.tflite_model_path), f"not found: {FLAGS.tflite_model_path}"
    interpreter = Interpreter(model_path=FLAGS.tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = _select_output_details(interpreter)
    input_size = int(input_details[0]['shape'][1])

    # Categories
    idx_to_name, idx_to_tax, meta = load_categories(
        FLAGS.categories_json,
        prefer_common_name=FLAGS.prefer_common_name
    )

    # Image
    arr = load_and_preprocess_image(
        FLAGS.image_path,
        size=input_size,
        center_crop=FLAGS.center_crop_to_square
    )

    # Prepare input & infer
    _prepare_input(interpreter, arr)
    interpreter.invoke()

    # Get & dequantize output
    y = interpreter.get_tensor(output_details['index'])
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    y = _maybe_dequantize_output(output_details, y)

    # Top-K
    k = min(FLAGS.top_k, y.shape[-1])
    topk_idx = np.argpartition(-y, range(k))[:k]
    topk_sorted = topk_idx[np.argsort(-y[topk_idx])]

    print(f"\n✅ Top-{k} predictions for {FLAGS.image_path}:\n")
    for rank, cls in enumerate(topk_sorted, 1):
        score = float(y[cls])
        raw = meta["id_to_raw"].get(int(cls), {})
        disp = idx_to_name.get(int(cls), f"(index {int(cls)})")
        sci = raw.get("name")
        com = raw.get("common_name")
        if com and sci and com != sci:
            disp_line = f"{disp}  (Scientific: {sci})"
        else:
            disp_line = disp

        print(f"{rank}. {disp_line}  [index={int(cls)}]")
        print(f"   Taxonomy:    {idx_to_tax.get(int(cls), '(taxonomy unavailable)')}")
        print(f"   Score/Logit: {score:.6f}\n")

if __name__ == '__main__':
    app.run(main)
