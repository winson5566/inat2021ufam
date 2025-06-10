import numpy as np
from PIL import Image
import json
from absl import app, flags
from tensorflow.lite.python.interpreter import Interpreter

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('tflite_model_path', None, help='Path to TFLite model file.')
flags.DEFINE_string('image_path', None, help='Path to input image.')
flags.DEFINE_string('categories_json', None, help='Path to categories.json.')
flags.DEFINE_integer('top_k', 5, help='Top K predictions to display.')

flags.mark_flag_as_required('tflite_model_path')
flags.mark_flag_as_required('image_path')
flags.mark_flag_as_required('categories_json')

# Preprocessing
def preprocess_image(image_path, size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main(_):
    # Load categories
    with open(FLAGS.categories_json, "r") as f:
        categories = json.load(f)
    id_to_info = {item["id"]: item for item in categories}

    # Load TFLite model
    interpreter = Interpreter(model_path=FLAGS.tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]

    # Preprocess image
    input_data = preprocess_image(FLAGS.image_path, input_size)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Top-K predictions
    top_k_indices = np.argsort(output)[-FLAGS.top_k:][::-1]
    print(f"✅ Top-{FLAGS.top_k} predictions for {FLAGS.image_path}:\n")

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

if __name__ == '__main__':
    app.run(main)
