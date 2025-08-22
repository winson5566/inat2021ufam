# app.py
import os
import time
import json
import glob
import numpy as np
import random
from PIL import Image
from flask import Flask, request, render_template_string, send_from_directory, url_for, session, jsonify
from werkzeug.utils import secure_filename
from tensorflow.lite.python.interpreter import Interpreter

# ---------------- Config (fixed) ----------------
MODEL_DIR         = "model/model_efficientnet_b0/export"
MODEL_OPTIONS = {
    "EfficientNet-B0 INT8 (TFLite PTQ)":  "model_efficientnet_b0_inat2021_drq.tflite",
    "EfficientNet-B0 FP16 (TFLite PTQ)":  "model_efficientnet_b0_inat2021_fp16.tflite",
    "EfficientNet-B0 FP32 (TFLite PTQ)":  "model_efficientnet_b0_inat2021_fp32.tflite",
}
DEFAULT_MODEL_KEY = "EfficientNet-B0 FP32 (TFLite PTQ)"  # 默认选中
CATEGORIES_JSON   = "inat2021/categories.json"
CATEGORIES_DETAIL_JSON = "inat2021/categories_detail.json"  # 新增：带 summary 的明细
CATEGORY_IMG_DIR  = "inat2021/categories"  # 示例图片目录
TOP_K             = 1
INPUT_NORM        = "none"
CENTER_CROP       = True
PREFER_COMMON_NAME= True
UPLOAD_DIR        = "uploads"
TEST_DIR          = "test"
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

def load_categories_detail(detail_json_path):
    """读取 categories_detail.json，返回 {id: summary}"""
    if not os.path.isfile(detail_json_path):
        return {}
    try:
        with open(detail_json_path, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        out = {}
        for item in arr:
            try:
                k = int(item.get("id"))
                summary = item.get("summary") or ""
                out[k] = summary
            except Exception:
                continue
        return out
    except Exception:
        return {}

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

def list_test_images():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.bmp")
    paths = []
    if os.path.isdir(TEST_DIR):
        for pat in exts:
            paths.extend(glob.glob(os.path.join(TEST_DIR, pat)))
    fnames = [os.path.basename(p) for p in paths]
    random.shuffle(fnames)  # 打乱顺序
    return fnames

def _example_image_url_for_cls(cls: int):
    raw = ID2RAW.get(cls, {})
    img_name = raw.get("image_dir_name")
    if not img_name:
        return None
    direct_path = os.path.join(CATEGORY_IMG_DIR, img_name)
    if os.path.isfile(direct_path):
        return url_for("category_image", filename=img_name)
    for ext in ("", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        p = os.path.join(CATEGORY_IMG_DIR, img_name + ext) if ext and not img_name.lower().endswith(ext) else os.path.join(CATEGORY_IMG_DIR, img_name)
        if os.path.isfile(p):
            return url_for("category_image", filename=os.path.basename(p))
    return None
# ------------------------------------------------

# ---------------- Flask App ----------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")  # 用于 session

IDX2NAME, IDX2TAX, ID2RAW = load_categories(CATEGORIES_JSON, prefer_common_name=PREFER_COMMON_NAME)
ID2DETAIL = load_categories_detail(CATEGORIES_DETAIL_JSON)  # 新增

_INTERPRETER_CACHE = {}

def get_interpreter(model_key: str):
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

def _trim_detail(text: str, max_len: int = 1000) -> str:
    """后端先做一次长度控制；前端再用 line-clamp 二次兜底。"""
    if not text:
        return ""
    t = text.strip()
    return (t[:max_len].rstrip() + "…") if len(t) > max_len else t

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SnapNature</title>
<style>
  :root{ --bg:#0b0f19; --card:rgba(17,24,39,0.55); --text:#e5e7eb; --muted:#9ca3af; --accent:#60a5fa; --border:rgba(31,41,55,0.45) }
  *{box-sizing:border-box}
  body{ margin:0;color:var(--text);font:14px system-ui,Segoe UI,Roboto,Helvetica,Arial;min-height:100vh;
        background: radial-gradient(1200px 800px at 20% -10%, #0b1224 0%, #0b0f19 60%, #0b0f19 100%); overflow-x:hidden; }

  .fx{ position:fixed; inset:0; pointer-events:none; z-index:-1; overflow:hidden; }
  .fx-grid{ position:absolute; inset:-50%;
            background: repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 60px),
                        repeating-linear-gradient(90deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 60px); }
  .fx-grain{ position:absolute; inset:-20%; background-image: radial-gradient(rgba(255,255,255,.03) 1px, transparent 1px);
             background-size: 3px 3px; opacity:.6; mix-blend-mode:soft-light; filter:contrast(120%) brightness(105%); }
  .blob{ position:absolute; width:48vw; height:48vw; max-width:820px; max-height:820px; filter: blur(80px); opacity:.5; border-radius:50%; animation: float 26s ease-in-out infinite; }
  .blob.b1{ left:-10vw; top:-10vh; background:radial-gradient(circle at 30% 30%, #2563eb, transparent 60%);}
  .blob.b2{ right:-8vw; bottom:-16vh; background:radial-gradient(circle at 60% 60%, #10b981, transparent 60%); animation-duration: 34s;}
  @keyframes float{ 0%{ transform: translate3d(0,0,0) scale(1);} 50%{ transform: translate3d(2vw,1vh,0) scale(1.05);} 100%{ transform: translate3d(0,0,0) scale(1);} }

  .wrap{max-width:1100px;margin:0 auto;padding:12px 14px;position:relative}
  header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:10px;flex-wrap:wrap}
  header h1{ font-size:36px; margin:4px 0 0; text-align:center; flex:1 0 100%; font-weight:800; letter-spacing:.2px; text-shadow: 0 2px 20px rgba(96,165,250,0.12); }

  .card{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px;
         backdrop-filter: blur(10px) saturate(130%); -webkit-backdrop-filter: blur(10px) saturate(130%);
         box-shadow: 0 12px 30px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.03); }

  select{ border-radius:10px;border:1px solid var(--border);padding:10px 12px;background:rgba(11,18,36,.7); color:var(--text); backdrop-filter: blur(6px); }
  .btn-snap{ padding:10px 16px;border:none;border-radius:12px;background:linear-gradient(90deg,#60a5fa,#34d399); color:#0b0f19;font-weight:800;cursor:pointer; }
  .kv{margin:0;padding:0;list-style:none} .kv li{margin:6px 0} .muted{color:var(--muted)}
  table {width:100%;border-collapse:collapse;margin:12px 0 0 0;font-size:12px}
  th, td {border:1px solid var(--border);padding:8px;text-align:center}
  th {background:rgba(15,23,42,.6)}

  .result-grid{display:grid;gap:12px;margin-top:12px}

  /* 四列：左图 | VS | 右图 | 信息 */
  .showdown{
    position:relative; display:grid;
    grid-template-columns: 1fr 96px 1fr 1.6fr;
    align-items:center; gap:12px; margin-top:12px;
  }
  @media(max-width:999px){
    .showdown{ grid-template-columns: 1fr 80px 1fr; }
    .infocol{ grid-column: 1 / -1; }
  }

  /* 图片框 */
  .thumbbox{
    position:relative; width:240px; height:240px;
    border:1px solid var(--border); border-radius:12px;
    background:rgba(11,18,36,.6);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding:8px; margin:0 auto; box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
    overflow:hidden;
  }
  .thumbimg{ flex:1 1 auto; width:100%; height: calc(100% - 22px); display:flex; align-items:center; justify-content:center; overflow:hidden; }
  .thumbimg img{ max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:block; }
  .caption{ margin-top:6px; height:22px; line-height:22px; font-size:12px; color:var(--muted); }

  /* 中间 VS + 置信度 */
  .center-stack{ display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; }
  .vs-badge{ display:inline-flex; align-items:center; justify-content:center; width:56px; height:56px; border-radius:50%;
             background:radial-gradient(circle at 30% 30%, #111827, #0b0f19 70%); border:2px solid rgba(96,165,250,.6); font-weight:900; }
  .confidence-badge{ padding:4px 10px; border-radius:999px; background:linear-gradient(90deg,#34d399,#60a5fa); color:#0b0f19; font-weight:800; font-size:14px; }

  /* 右侧信息列：上下两个面板 */
  .infocol{
    display:grid; gap:12px; align-self:stretch;
    grid-auto-rows: minmax(0, auto);
  }
  .panel{
    border:1px solid var(--border); border-radius:12px; padding:12px;
    background:rgba(11,18,36,.55);
  }
  .panel h2{ margin:0 0 6px 0; font-size:18px; line-height:1.2 }
  .panel h3{ margin:0 0 6px 0; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.4px }

  /* Detail 行：三行截断 */
  .detail-clamp{
    display:-webkit-box;
    -webkit-line-clamp:8;
    -webkit-box-orient:vertical;
    overflow:hidden;
  }

  input[type=file]{display:none}
  .dropzone{ border:2px dashed var(--border);border-radius:12px;padding:28px;color:var(--muted);user-select:none;cursor:pointer;flex-basis:100%;
             display:flex;align-items:center;justify-content:center;text-align:center;gap:10px;background:rgba(11,18,36,.5); }
  .dz-inner{display:flex;align-items:center;gap:12px;flex-wrap:wrap;justify-content:center}
  .dz-icon{width:36px;height:36px;opacity:.85}
  .dz-text{font-size:14px}
  .dz-text a{color:var(--accent);text-decoration:none}

  .form-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}

  .rows-wrap{margin-top:12px; border:1px solid var(--border); border-radius:12px; background:rgba(11,18,36,.55); overflow:hidden; }
  .gallery-title{color:var(--muted);font-size:14px;margin:10px 12px 6px;text-align:center;font-weight:600;letter-spacing:.2px}
  .row-track{ display:flex; gap:8px; padding:6px 8px 6px; width:max-content; animation: scroll-left 80s linear infinite; will-change: transform; white-space:nowrap; }
  .row-track.slower{ animation-duration: 96s; }
  .rows-wrap:hover .row-track{ animation-play-state: paused; }
  @keyframes scroll-left{ 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }

  .tile{ display:inline-block; width:112px;height:88px; border-radius:10px; overflow:hidden; border:1px solid var(--border); background:rgba(11,18,36,.6); cursor:pointer; }
  .tile img{ width:100%;height:100%;object-fit:cover;display:block; }
</style>
</head>
<body>

<div class="fx" aria-hidden="true">
  <div class="fx-grid"></div>
  <div class="fx-grain"></div>
  <div class="blob b1"></div>
  <div class="blob b2"></div>
</div>

<div class="wrap">
  <header>
    <h1>COSC681 AI Project: SnapNature</h1>

    <details class="card" style="width:100%; margin-top:8px;">
      <summary style="cursor:pointer; list-style:none; display:flex; align-items:center; justify-content:center; gap:8px; font-size:16px; font-weight:700;">
        <span class="muted">Model Accuracy Comparison</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style="opacity:.7">
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </summary>
      <div style="margin-top:8px">
        <table>
          <thead>
            <tr>
              <th>Model Type</th><th>Params</th><th>Model Size (MB)</th><th>Top1</th><th>Top3</th><th>Top5</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>EfficientNet-B0 FP32 (Baseline)</td><td>16.86M</td><td>64.31 MB</td><td>76.1%</td><td>87.8%</td><td>91.0%</td></tr>
            <tr><td>EfficientNet-B0 FP32 (TFLite PTQ)</td><td>16.80M</td><td>64.07 MB</td><td>73.8%</td><td>86.5%</td><td>89.9%</td></tr>
            <tr><td>EfficientNet-B0 FP16 (TFLite PTQ)</td><td>16.80M</td><td>32.04 MB</td><td>73.8%</td><td>86.5%</td><td>89.8%</td></tr>
            <tr><td>EfficientNet-B0 INT8 (TFLite PTQ)</td><td>16.80M</td><td>16.15 MB</td><td>70.8%</td><td>84.2%</td><td>88.7%</td></tr>
          </tbody>
        </table>
      </div>
    </details>

    <form id="frm" method="post" enctype="multipart/form-data" style="width:100%">
      <div class="form-row" style="margin-top:10px;">
        <label class="muted">Model:</label>
        <select name="model_key" id="modelKey" required>
          {% for k in model_keys %}
            <option value="{{k}}" {% if k==chosen_model %}selected{% endif %}>{{k}}</option>
          {% endfor %}
        </select>
        <button type="submit" name="action" value="snap" class="btn-snap" title="Re-run on the last image">Snap</button>
      </div>
      <div style="flex-basis:100%; height:12px"></div>
      <div id="dz" class="dropzone">
        <div class="dz-inner">
          <svg class="dz-icon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M19 15a4 4 0 0 0-3.8-4 5 5 0 0 0-9.2 2A3 3 0 0 0 6 20h12a3 3 0 0 0 1-5zM12 8l3 3h-2v3h-2v-3H9l3-3z"/>
          </svg>
          <div class="dz-text">Drag an image here or <a href="#" id="uploadLink">upload a file</a></div>
        </div>
        <input id="file" type="file" name="file" accept="image/*">
      </div>

      {% if test_files and test_files|length > 0 %}
      <div class="rows-wrap" aria-label="Auto-scrolling test images">
        <div class="row-track" id="row1">
          {% for fname in test_files[::2] %}
            <button class="tile test-tile" type="button" data-fname="{{fname}}" title="{{fname}}">
              <img src="{{ url_for('test_file', filename=fname) }}" loading="lazy" alt="{{fname}}">
            </button>
          {% endfor %}
          {% for fname in test_files[::2] %}
            <button class="tile test-tile" type="button" data-fname="{{fname}}" title="{{fname}}">
              <img src="{{ url_for('test_file', filename=fname) }}" loading="lazy" alt="{{fname}}">
            </button>
          {% endfor %}
        </div>
        <div class="row-track slower" id="row2" style="margin-top:8px;">
          {% for fname in test_files[1::2] %}
            <button class="tile test-tile" type="button" data-fname="{{fname}}" title="{{fname}}">
              <img src="{{ url_for('test_file', filename=fname) }}" loading="lazy" alt="{{fname}}">
            </button>
          {% endfor %}
          {% for fname in test_files[1::2] %}
            <button class="tile test-tile" type="button" data-fname="{{fname}}" title="{{fname}}">
              <img src="{{ url_for('test_file', filename=fname) }}" loading="lazy" alt="{{fname}}">
            </button>
          {% endfor %}
        </div>
      </div>
      {% endif %}
    </form>
  </header>

  <div id="resultHost">
    {% if result %}
    <div class="card" id="resultCard">
      <div class="showdown">
        <!-- 左：Uploaded -->
        <div class="thumbbox">
          <div class="thumbimg"><img src="{{ result.upload_url }}" alt="uploaded image"></div>
          <div class="caption">Uploaded</div>
        </div>

        <!-- 中：VS + 置信度（已移除模型名与推理时间两行） -->
        <div class="center-stack" aria-label="vs and confidence">
          <div class="vs-badge">VS</div>
          <div class="confidence-badge">{{ result.score }}</div>
        </div>

        <!-- 右：Example（若无则占位） -->
        {% if result.example_url %}
        <div class="thumbbox">
          <div class="thumbimg"><img src="{{ result.example_url }}" alt="example species image"></div>
          <div class="caption">Example</div>
        </div>
        {% else %}
        <div class="thumbbox" style="opacity:.6">
          <div class="thumbimg"><img src="" alt="" style="display:none"></div>
          <div class="caption">Example</div>
        </div>
        {% endif %}

        <!-- 右侧上下两个面板 -->
        <div class="infocol">
          <!-- 上：物种信息 -->
          <section class="panel" aria-label="Species information">
            <h2>{{ result.disp }}</h2>
            <ul class="kv">
              <li><strong class="muted">Scientific name:</strong> <span>{{ result.sci }}</span></li>
              <li><strong class="muted">Taxonomy:</strong> <span>{{ result.tax }}</span></li>
              <li><strong class="muted">Detail:</strong> <span class="detail-clamp">{{ result.detail }}</span></li>
            </ul>
          </section>
          <!-- 下：推理信息 -->
          <section class="panel" aria-label="Inference information">
            <h3>Inference Detail</h3>
            <ul class="kv">
              <li><span class="muted">Model:</span> <span>{{ result.model_key }}</span></li>
              <li><span class="muted">Inference time:</span> <span>{{ result.ms }} ms</span></li>
              <li><span class="muted">Confidence:</span> <span>{{ result.score }}</span></li>
            </ul>
          </section>
        </div>
      </div>
    </div>
    {% else %}
    <div class="card" id="resultCard" style="text-align:center;color:var(--muted);margin-top:12px">
      Upload an image to get a prediction.
    </div>
    {% endif %}
  </div>
</div>

<script>
  const frm   = document.getElementById('frm');
  const dz    = document.getElementById('dz');
  const file  = document.getElementById('file');
  const link  = document.getElementById('uploadLink');
  const model = document.getElementById('modelKey');
  const host  = document.getElementById('resultHost');

  file.addEventListener('change', () => { if (file.files && file.files.length > 0) frm.submit(); });
  dz.addEventListener('click', (e) => { if (e.target.id !== 'uploadLink') file.click(); });
  link.addEventListener('click', (e) => { e.preventDefault(); file.click(); });

  ['dragenter','dragover'].forEach(evt=>{
    dz.addEventListener(evt, e=>{ e.preventDefault(); e.stopPropagation(); dz.classList.add('highlight'); });
  });
  ['dragleave','drop'].forEach(evt=>{
    dz.addEventListener(evt, e=>{ e.preventDefault(); e.stopPropagation(); dz.classList.remove('highlight'); });
  });

  dz.addEventListener('drop', e=>{
    const dt = e.dataTransfer;
    if (!dt || !dt.files || dt.files.length===0) return;
    const f = dt.files[0];
    const dtr = new DataTransfer();
    dtr.items.add(f);
    file.files = dtr.files;
    frm.submit();
  });

  function bindTiles(){
    document.querySelectorAll('.test-tile').forEach(btn=>{
      btn.addEventListener('click', async ()=>{
        const fname = btn.dataset.fname;
        const modelKey = model.value || '{{ chosen_model }}';
        try{
          const res = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ test_name: fname, model_key: modelKey })
          });
          const data = await res.json();
          if(data && data.ok){
            const r = data.result;
            host.innerHTML = `
              <div class="card" id="resultCard">
                <div class="showdown">
                  <div class="thumbbox">
                    <div class="thumbimg"><img src="${r.upload_url}" alt="uploaded image"></div>
                    <div class="caption">Uploaded</div>
                  </div>

                  <div class="center-stack" aria-label="vs and confidence">
                    <div class="vs-badge">VS</div>
                    <div class="confidence-badge">${r.score}</div>
                  </div>

                  ${r.example_url ? `
                  <div class="thumbbox">
                    <div class="thumbimg"><img src="${r.example_url}" alt="example species image"></div>
                    <div class="caption">Example</div>
                  </div>` : `
                  <div class="thumbbox" style="opacity:.6">
                    <div class="thumbimg"><img src="" alt="" style="display:none"></div>
                    <div class="caption">Example</div>
                  </div>`}

                  <div class="infocol">
                    <section class="panel" aria-label="Species information">
                      <h2>${r.disp}</h2>
                      <ul class="kv">
                        <li><strong class="muted">Scientific name:</strong> <span>${r.sci ?? ''}</span></li>
                        <li><strong class="muted">Taxonomy:</strong> <span>${r.tax}</span></li>
                        <li><strong class="muted">Detail:</strong> <span class="detail-clamp">${r.detail ?? ''}</span></li>
                      </ul>
                    </section>
                    <section class="panel" aria-label="Inference information">
                      <h3>Inference Detail</h3>
                      <ul class="kv">
                        <li><span class="muted">Model:</span> <span>${r.model_key}</span></li>
                        <li><span class="muted">Inference time:</span> <span>${r.ms} ms</span></li>
                        <li><span class="muted">Confidence:</span> <span>${r.score}</span></li>
                      </ul>
                    </section>
                  </div>
                </div>
              </div>`;
          } else {
            console.error(data);
          }
        }catch(err){ console.error(err); }
      });
    });
  }
  bindTiles();
</script>

</body>
</html>
"""

def _run_inference(interpreter, input_size, out_details, img_path):
    arr = load_and_preprocess_image(img_path, size=input_size, center_crop=CENTER_CROP)
    _prepare_input(interpreter, arr)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0
    y = interpreter.get_tensor(out_details['index'])
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    y = _maybe_dequantize_output(out_details, y)
    k = min(TOP_K, y.shape[-1])
    topk_idx = np.argpartition(-y, range(k))[:k]
    topk_sorted = topk_idx[np.argsort(-y[topk_idx])]
    cls = int(topk_sorted[0])
    score = float(y[cls])
    return infer_ms, cls, score

def _predict_on_path(model_key, img_path, img_url):
    interpreter, input_size, out_details = get_interpreter(model_key)
    infer_ms, cls, score = _run_inference(interpreter, input_size, out_details, img_path)
    score_pct = f"{score*100:.1f}%"
    raw = ID2RAW.get(cls, {})
    disp = IDX2NAME.get(cls, f"(index {cls})")
    sci  = raw.get("name")
    tax  = IDX2TAX.get(cls, "?")
    ex_url = _example_image_url_for_cls(cls)
    detail_full = ID2DETAIL.get(cls, "")
    detail = _trim_detail(detail_full)  # 截断
    return {
        "disp": disp, "sci": sci, "tax": tax, "detail": detail,
        "score": score_pct, "ms": f"{infer_ms:.2f}",
        "model_key": model_key, "upload_url": img_url,
        "example_url": ex_url
    }

@app.route("/", methods=["GET","POST"])
def index():
    chosen_model = DEFAULT_MODEL_KEY
    result = None
    if request.method == "POST":
        chosen_model = request.form.get("model_key", DEFAULT_MODEL_KEY)
        f = request.files.get("file")
        test_name = request.form.get("test_name")
        img_path = None
        img_url  = None
        if f and f.filename:
            fname = secure_filename(f.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            f.save(save_path)
            img_path = save_path
            img_url  = url_for("uploaded_file", filename=fname)
            session['last_path'] = img_path
            session['last_url']  = img_url
        elif test_name:
            safe_name = os.path.basename(test_name)
            test_path = os.path.join(TEST_DIR, safe_name)
            if os.path.isfile(test_path):
                img_path = test_path
                img_url  = url_for("test_file", filename=safe_name)
                session['last_path'] = img_path
                session['last_url']  = img_url
        else:
            last_path = session.get('last_path'); last_url = session.get('last_url')
            if last_path and os.path.isfile(last_path):
                img_path = last_path; img_url = last_url
            else:
                return render_template_string(HTML, result=None, model_keys=list(MODEL_OPTIONS.keys()),
                                              chosen_model=chosen_model, test_files=list_test_images())
        result = _predict_on_path(chosen_model, img_path, img_url)

    return render_template_string(HTML, result=result, model_keys=list(MODEL_OPTIONS.keys()),
                                  chosen_model=chosen_model, test_files=list_test_images())

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key", DEFAULT_MODEL_KEY)
    test_name = data.get("test_name")
    if not test_name:
        last_path = session.get('last_path'); last_url  = session.get('last_url')
        if not (last_path and os.path.isfile(last_path)):
            return jsonify(ok=False, error="no_image")
        try:
            result = _predict_on_path(model_key, last_path, last_url)
            return jsonify(ok=True, result=result)
        except Exception as e:
            return jsonify(ok=False, error=str(e))
    safe_name = os.path.basename(test_name)
    img_path = os.path.join(TEST_DIR, safe_name)
    if not os.path.isfile(img_path):
        return jsonify(ok=False, error="file_not_found")
    img_url = url_for("test_file", filename=safe_name)
    session['last_path'] = img_path; session['last_url'] = img_url
    try:
        result = _predict_on_path(model_key, img_path, img_url)
        return jsonify(ok=True, result=result)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/test/<path:filename>")
def test_file(filename):
    return send_from_directory(TEST_DIR, filename)

@app.route("/categories/<path:filename>")
def category_image(filename):
    return send_from_directory(CATEGORY_IMG_DIR, filename)

if __name__=="__main__":
    app.run(debug=True)
