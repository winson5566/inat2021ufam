# app.py
import os
import time
import json
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string, send_from_directory, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.lite.python.interpreter import Interpreter

# ---------------- Config (fixed) ----------------
MODEL_DIR         = "model/model_efficientnet_b0/export"
MODEL_OPTIONS = {
    "TFLite INT8 (PTQ)":  "model_efficientnet_b0_inat2021_drq.tflite",
    "TFLite FP16 (PTQ)":  "model_efficientnet_b0_inat2021_fp16.tflite",
    "TFLite FP32 (PTQ)":  "model_efficientnet_b0_inat2021_fp32.tflite",
}
DEFAULT_MODEL_KEY = "TFLite FP32 (PTQ)"  # 默认选中
CATEGORIES_JSON   = "inat2021/categories.json"
TOP_K             = 1
INPUT_NORM        = "none"
CENTER_CROP       = True
PREFER_COMMON_NAME= True
UPLOAD_DIR        = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# ------------------------------------------------

# ---------------- Helpers -----------------------
def _maybe_dequantize_output(output_details, y):
    y = y.astype(np.float32, copy=False)
    qparams = output_details.get("quantization_parameters", {})
    scales = qparams.get("scales", None)
    zero_points = qparams.get("zero_points", None)
    quantized_dimension = output_details.get("quantized_dimension", None)
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
    if INPUT_NORM == 'minus1_1':
        x = x_f32_0_255 / 255.0
        return x * 2.0 - 1.0
    elif INPUT_NORM == 'imagenet':
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

def load_categories(categories_json, prefer_common_name=True):
    with open(categories_json, 'r', encoding='utf-8') as f:
        cats = json.load(f)
    idx_to_name, idx_to_tax, id_to_raw = {}, {}, {}
    for i, c in enumerate(cats):
        idx = int(c.get('id', i))
        sci_name = c.get('name', f'class_{idx}')
        com_name = c.get('common_name') or None
        show_name = (com_name if (prefer_common_name and com_name) else sci_name)
        parts = [
            c.get('kingdom', '?'), c.get('phylum', '?'), c.get('class', '?'),
            c.get('order', '?'),   c.get('family', '?'), c.get('genus', '?')
        ]
        tax = " > ".join(parts)
        sp = c.get('specific_epithet')
        if sp:
            tax = f"{tax} > {sp}"
        idx_to_name[idx] = show_name
        idx_to_tax[idx] = tax
        id_to_raw[idx] = c
    return idx_to_name, idx_to_tax, id_to_raw

def load_and_preprocess_image(path, size, center_crop=True):
    img = Image.open(path).convert('RGB')
    if center_crop:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top  = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img)
# ------------------------------------------------

# ---------------- Flask App ----------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")  # 用于 session 记忆“上次图片”

# 类别表只需加载一次
IDX2NAME, IDX2TAX, ID2RAW = load_categories(CATEGORIES_JSON, prefer_common_name=PREFER_COMMON_NAME)

# 模型解释器缓存：{model_key: (Interpreter, input_size, output_details)}
_INTERPRETER_CACHE = {}

def get_interpreter(model_key: str):
    """按选择的模型返回解释器与输入尺寸、输出细节。缓存避免重复加载。"""
    if model_key not in MODEL_OPTIONS:
        model_key = DEFAULT_MODEL_KEY
    if model_key in _INTERPRETER_CACHE:
        return _INTERPRETER_CACHE[model_key]

    model_path = os.path.join(MODEL_DIR, MODEL_OPTIONS[model_key])
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_size = int(interpreter.get_input_details()[0]['shape'][1])
    output_details = interpreter.get_output_details()[0]
    _INTERPRETER_CACHE[model_key] = (interpreter, input_size, output_details)
    return _INTERPRETER_CACHE[model_key]

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Species Classifier</title>
<style>
  :root{--bg:#0b0f19;--card:#111827;--text:#e5e7eb;--muted:#9ca3af;--accent:#60a5fa;--border:#1f2937}
  *{box-sizing:border-box}
  body{margin:0;background:#0b0f19;color:var(--text);font:14px system-ui,Segoe UI,Roboto,Helvetica,Arial}
  .wrap{max-width:980px;margin:0 auto;padding:12px 14px}
  header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:10px;flex-wrap:wrap}
  header h1{font-size:18px;margin:0}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px}
  form{display:flex;gap:10px;align-items:center;flex-wrap:wrap}

  /* ---- Custom file button ---- */
  input[type=file]{display:none}
  .file-btn{
    background:var(--accent);color:#0b0f19;font-weight:700;border:0;
    border-radius:9px;padding:8px 12px;cursor:pointer;display:inline-block
  }

  select,button{border-radius:9px;border:1px solid var(--border);padding:8px 10px}
  button{background:var(--accent);color:#0b0f19;font-weight:700;border:0;cursor:pointer}

  .kv{margin:0;padding:0;list-style:none}
  .kv li{margin:6px 0}
  .muted{color:var(--muted)}

  /* ---- Table smaller + centered ---- */
  table {width:100%;border-collapse:collapse;margin:12px 0 0 0;font-size:12px}
  th, td {border:1px solid var(--border);padding:8px;text-align:center}
  th {background:#0f172a}

  /* ---- Result layout with fixed thumbnail ---- */
  .result-grid{display:grid;gap:12px;margin-top:12px}
  @media(min-width:800px){.result-grid{grid-template-columns:260px 1fr}}
  .thumbbox{
    width:240px;height:240px;border:1px solid var(--border);border-radius:10px;
    background:#0b1224;display:flex;align-items:center;justify-content:center;overflow:hidden
  }
  .thumbbox img{max-width:100%;max-height:100%;object-fit:contain;display:block}

  /* ---- Drag & Drop zone ---- */
  .dropzone{
    border:2px dashed var(--border);border-radius:10px;padding:10px;
    color:var(--muted);user-select:none
  }
  .dropzone.highlight{border-color:var(--accent);color:var(--text);background:#0b1224}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>COSC681 AI Project: Species Classifier</h1>
    <form id="frm" method="post" enctype="multipart/form-data">
      <label class="muted">Model:</label>
      <select name="model_key" required>
        {% for k in model_keys %}
          <option value="{{k}}" {% if k==chosen_model %}selected{% endif %}>{{k}}</option>
        {% endfor %}
      </select>

      <!-- custom file button (no filename text shown) -->
      <label for="file" class="file-btn">Choose Image</label>
      <input id="file" type="file" name="file" accept="image/*">

      <button type="submit">Classify</button>

      <!-- drag & drop zone -->
      <div id="dz" class="dropzone" style="flex-basis:100%">Drag & Drop image here (or click “Choose Image”)</div>
    </form>
  </header>

  <!-- 静态对比表（固定内容，非图片） -->
  <div class="card">
    <div class="muted" style="margin-bottom:6px">Model Accuracy Comparison</div>
    <table>
      <thead>
        <tr>
          <th>Model Type</th>
          <th>Params</th>
          <th>Model Size (MB)</th>
          <th>Top1</th>
          <th>Top3</th>
          <th>Top5</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>FP32 (Baseline)</td><td>16.86M</td><td>64.31 MB</td><td>76.1%</td><td>87.8%</td><td>91.0%</td>
        </tr>
        <tr>
          <td>TFLite FP32(PTQ)</td><td>16.80M</td><td>64.07 MB</td><td>73.8%</td><td>86.5%</td><td>89.9%</td>
        </tr>
        <tr>
          <td>TFLite FP16(PTQ)</td><td>16.80M</td><td>32.04 MB</td><td>73.8%</td><td>86.5%</td><td>89.8%</td>
        </tr>
        <tr>
          <td>TFLite INT8(PTQ)</td><td>16.80M</td><td>16.15 MB</td><td>70.8%</td><td>84.2%</td><td>88.7%</td>
        </tr>
      </tbody>
    </table>
  </div>

  {% if result %}
  <div class="card result-grid">
    <div class="thumbbox">
      <img src="{{ result.upload_url }}" alt="uploaded image">
    </div>
    <div>
      <h2 style="margin:0 0 6px 0;font-size:18px;line-height:1.2">{{ result.disp }}</h2>
      <ul class="kv">
        <li><span class="muted">Scientific name:</span> {{ result.sci }}</li>
        <li><span class="muted">Taxonomy:</span> {{ result.tax }}</li>
        <li><span class="muted">Confidence:</span> {{ result.score }}</li>
        <li><span class="muted">Model:</span> {{ result.model_key }}</li>
        <li><span class="muted">Inference time:</span> {{ result.ms }} ms</li>
      </ul>
    </div>
  </div>
  {% else %}
  <div class="card" style="text-align:center;color:var(--muted);margin-top:12px">Upload an image to get a prediction.</div>
  {% endif %}
</div>

<script>
  // Drag & Drop -> set to hidden file input, then auto-submit the form
  const frm  = document.getElementById('frm');
  const dz   = document.getElementById('dz');
  const file = document.getElementById('file');

  // 选择文件后自动提交（点“Choose Image”选择）
  file.addEventListener('change', () => {
    if (file.files && file.files.length > 0) {
      frm.submit();
    }
  });

  // 拖拽交互高亮
  ['dragenter','dragover'].forEach(evt=>{
    dz.addEventListener(evt, e=>{
      e.preventDefault(); e.stopPropagation();
      dz.classList.add('highlight');
    });
  });
  ['dragleave','drop'].forEach(evt=>{
    dz.addEventListener(evt, e=>{
      e.preventDefault(); e.stopPropagation();
      dz.classList.remove('highlight');
    });
  });

  // 放下文件：灌入隐藏 input，再自动提交
  dz.addEventListener('drop', e=>{
    const dt = e.dataTransfer;
    if (!dt || !dt.files || dt.files.length===0) return;
    const f = dt.files[0];
    const dtr = new DataTransfer();
    dtr.items.add(f);
    file.files = dtr.files;   // attach to hidden input so it posts with the form
    frm.submit();             // auto submit
  });
</script>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    chosen_model = DEFAULT_MODEL_KEY
    result = None

    if request.method == "POST":
        chosen_model = request.form.get("model_key", DEFAULT_MODEL_KEY)
        f = request.files.get("file")
        fname = None

        # 如果本次有新文件，就保存并覆盖“上次文件”
        if f and f.filename:
            fname = secure_filename(f.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            f.save(save_path)
            session['last_upload'] = fname
        else:
            # 没有新文件 -> 复用上次文件
            last = session.get('last_upload')
            if last:
                fname = last
            else:
                # 没有任何可用图片
                return render_template_string(
                    HTML,
                    result=None,
                    model_keys=list(MODEL_OPTIONS.keys()),
                    chosen_model=chosen_model
                )

        # 取对应模型的解释器
        interpreter, input_size, out_details = get_interpreter(chosen_model)

        # 用“当前/上次”图片进行推理
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        arr = load_and_preprocess_image(img_path, size=input_size, center_crop=CENTER_CROP)
        _prepare_input(interpreter, arr)

        # 仅统计前向推理时延
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        y = interpreter.get_tensor(out_details['index'])
        if y.ndim==2 and y.shape[0]==1:
            y = y[0]
        y = _maybe_dequantize_output(out_details, y)

        k = min(TOP_K, y.shape[-1])
        topk_idx = np.argpartition(-y, range(k))[:k]
        topk_sorted = topk_idx[np.argsort(-y[topk_idx])]

        cls = int(topk_sorted[0]); score = float(y[cls])
        score_pct = f"{score*100:.1f}%"
        raw = ID2RAW.get(cls, {})
        disp = IDX2NAME.get(cls, f"(index {cls})")
        sci  = raw.get("name")
        tax  = IDX2TAX.get(cls, "?")

        result = {
            "disp": disp, "sci": sci, "tax": tax,
            "score": score_pct, "ms": f"{infer_ms:.2f}",
            "model_key": chosen_model,
            "upload_url": url_for("uploaded_file", filename=fname)
        }

    return render_template_string(
        HTML,
        result=result,
        model_keys=list(MODEL_OPTIONS.keys()),
        chosen_model=chosen_model
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__=="__main__":
    app.run(debug=True)
