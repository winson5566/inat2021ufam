import tensorflow as tf
from absl import app, flags

# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', default='mobilenetv3', help='Model to use: mobilenetv3 or efficientnetb0')
flags.DEFINE_integer('input_size', default=224, help='Input image size.')
flags.DEFINE_integer('num_classes', default=1000, help='Number of classes.')
flags.DEFINE_string('weights_path', None, help='Path to .h5 weights file.')
flags.DEFINE_string('saved_model_dir', None, help='Output directory for SavedModel.')
flags.DEFINE_string('tflite_path', None, help='Output path for TFLite file.')
flags.DEFINE_integer('seed', default=42, help='Random seed.')

flags.mark_flag_as_required('weights_path')
flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('tflite_path')


def build_model(model_name='mobilenetv3', input_size=224, num_classes=1000):
    input_shape = (input_size, input_size, 3)

    if model_name == 'mobilenet_v3':
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
    elif model_name == 'efficientnet_b0':
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    base_model.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def main(_):
    tf.random.set_seed(FLAGS.seed)

    model = build_model(
        model_name=FLAGS.model_name,
        input_size=FLAGS.input_size,
        num_classes=FLAGS.num_classes
    )

    model.load_weights(FLAGS.weights_path)
    print("✅ Weights loaded successfully.")

    model.export(FLAGS.saved_model_dir)
    print(f"✅ SavedModel exported to: {FLAGS.saved_model_dir}/")

    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(FLAGS.tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to: {FLAGS.tflite_path}")


if __name__ == '__main__':
    app.run(main)
