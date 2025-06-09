import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import json

# ==== TFLite Interpreter Import ====
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ==== CONFIG ====
MODEL_PATH = "model/model_mobile_v3/export/MobileNet_v3_small_inat2021.tflite"
CATEGORIES_JSON = "inat2021/categories.json"
INPUT_SIZE = 224
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# ===============

# Load category info
with open(CATEGORIES_JSON, "r") as f:
    categories = json.load(f)
id_to_info = {item["id"]: item for item in categories}

# Load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Predict from image
def predict(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    class_id = int(np.argmax(output))
    confidence = float(output[class_id])
    return class_id, confidence

# UI functions
def update_video():
    ret, frame = cap.read()
    if not ret:
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(30, update_video)

def capture_and_classify():
    ret, frame = cap.read()
    if not ret:
        return
    class_id, conf = predict(frame)
    if class_id in id_to_info:
        name = id_to_info[class_id]["name"]
        taxonomy = f"{id_to_info[class_id]['kingdom']} > {id_to_info[class_id]['phylum']} > {id_to_info[class_id]['class']} > {id_to_info[class_id]['order']}"
        result_label.config(text=f"{name} ({conf:.2%})")
        taxonomy_label.config(text=taxonomy)
    else:
        result_label.config(text="Unknown")
        taxonomy_label.config(text="")

# ==== UI SETUP ====
root = tk.Tk()
root.title("AI Camera")
root.geometry("700x600")  # 宽 x 高
root.resizable(False, False)

video_label = tk.Label(root)
video_label.pack()

btn = tk.Button(root, text="📷 Capture & Classify", command=capture_and_classify, font=("Helvetica", 14))
btn.pack(pady=5)

result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"))
result_label.pack()

taxonomy_label = tk.Label(root, text="", font=("Helvetica", 11), wraplength=680, justify="center")
taxonomy_label.pack()

# ==== START VIDEO ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
update_video()

root.mainloop()
cap.release()
