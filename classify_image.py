import numpy as np
from PIL import Image
import json
from tensorflow.lite.python.interpreter import Interpreter

# Paths
TFLITE_MODEL_PATH = "model/model_mobile_v3/export/MobileNet_v3_small_inat2021.tflite"
IMAGE_PATH = "test/sasanqua_camellia.png"
CATEGORIES_JSON = "inat2021/categories.json"
TOP_K = 5

# Load iNat2021 categories (id → full taxonomy info)
with open(CATEGORIES_JSON, "r") as f:
    categories = json.load(f)

id_to_info = {item["id"]: item for item in categories}

# Load model
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_size = input_shape[1]  # assumes square input


# Preprocess image
def preprocess_image(image_path, size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


input_data = preprocess_image(IMAGE_PATH, input_size)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (num_classes,)

# Get top-k predictions
top_k_indices = np.argsort(output)[-TOP_K:][::-1]  # descending order

print(f"✅ Top-{TOP_K} predictions for {IMAGE_PATH}:\n")
for rank, class_id in enumerate(top_k_indices, start=1):
    confidence = float(output[class_id])
    if class_id in id_to_info:
        info = id_to_info[class_id]
        name = info["name"]
        tax = f"{info['kingdom']} > {info['phylum']} > {info['class']} > {info['order']} > {info['family']} > {info['genus']}"
    else:
        name = "(Unknown)"
        tax = "(No taxonomy info available)"

    print(f"{rank}. {name} (ID: {class_id})")
    print(f"   Confidence: {confidence:.4f}")
    print(f"   Taxonomy:   {tax}\n")
