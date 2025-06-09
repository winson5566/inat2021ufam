import tensorflow as tf
from absl import app, flags

# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_size', default=224, help='Input image size.')
flags.DEFINE_integer('num_classes', default=1000, help='Number of classes.')
flags.DEFINE_string('weights_path', None, help='Path to .h5 weights file.')
flags.DEFINE_string('saved_model_dir', None, help='Output directory for SavedModel.')
flags.DEFINE_string('tflite_path', None, help='Output path for TFLite file.')
flags.DEFINE_integer('seed', default=42, help='Random seed.')

flags.mark_flag_as_required('weights_path')
flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('tflite_path')

def build_model():
    # Load base MobileNetV3Small model without top layer
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = True

    # Build full classification model
    inputs = tf.keras.Input(shape=(FLAGS.input_size, FLAGS.input_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def main(_):
    tf.random.set_seed(FLAGS.seed)

    model = build_model()
    model.load_weights(FLAGS.weights_path)
    print("✅ Weights loaded successfully.")

    model.export(FLAGS.saved_model_dir)  # <-- use export here
    print(f"✅ SavedModel exported to: {FLAGS.saved_model_dir}/")

    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(FLAGS.tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to: {FLAGS.tflite_path}")

if __name__ == '__main__':
    app.run(main)
