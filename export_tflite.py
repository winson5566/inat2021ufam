import os
import shutil
import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', default='efficientnet_b0',
                    help="Backbone: 'efficientnet_b0' or 'mobilenet_v3'")
flags.DEFINE_integer('input_size', default=224, help='Input image size.')
flags.DEFINE_integer('num_classes', default=10000, help='Number of classes.')
flags.DEFINE_string('weights_path', None, help='Path to .h5 weights file.')
flags.DEFINE_string('saved_model_dir', None, help='Output directory for SavedModel.')
flags.DEFINE_string('tflite_path', None, help='Output path for TFLite file.')
flags.DEFINE_integer('seed', default=42, help='Random seed.')

# 新增：选择导出精度
flags.DEFINE_enum('export_precision', default='fp32',
                  enum_values=['fp32', 'fp16', 'drq'],
                  help="TFLite export precision: 'fp32' (no quant), 'fp16' (half precision), 'drq' (Dynamic Range Quantization)")

flags.mark_flag_as_required('weights_path')
flags.mark_flag_as_required('saved_model_dir')
flags.mark_flag_as_required('tflite_path')


def build_model(model_name='efficientnet_b0', input_size=224, num_classes=10000):
    input_shape = (input_size, input_size, 3)

    if model_name == 'mobilenet_v3':
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape, include_top=False, weights=None
        )
    elif model_name == 'efficientnet_b0':
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights=None
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def export_saved_model(model, export_dir):
    # 可选：清空旧目录，避免残留（若不想清空，可注释掉）
    # if os.path.isdir(export_dir):
    #     shutil.rmtree(export_dir)
    model.export(export_dir)  # Keras 3 API
    print(f"✅ SavedModel exported to: {export_dir}/")


def convert_to_tflite(saved_model_dir: str, out_path: str, mode: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if mode == 'fp32':
        # 不做任何优化：保持 FP32
        print("➡️ Export mode: FP32 (no quantization).")
        # no optimizations set

    elif mode == 'fp16':
        # FP16 权重量化（模型体积减半，很多设备仍以 FP32/FP16 运行）
        print("➡️ Export mode: FP16 (float16 weights).")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 关键：允许 float16 类型
        converter.target_spec.supported_types = [tf.float16]
        # 维持浮点 I/O，兼容性更好（如需严格 FP16 I/O，TFLite 目前仍以 float32 接口为主）
        # converter.inference_input_type = tf.float32
        # converter.inference_output_type = tf.float32

    elif mode == 'drq':
        # 动态范围量化（权重 INT8，激活浮点；无需代表性数据）
        print("➡️ Export mode: DRQ (Dynamic Range Quantization: int8 weights, float activations).")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # 不提供 representative_dataset，即为 DRQ；I/O 默认 float32

    else:
        raise ValueError(f"Unknown export mode: {mode}")

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to: {out_path}")


def main(_):
    tf.random.set_seed(FLAGS.seed)

    print("🛠️ Building model...")
    model = build_model(
        model_name=FLAGS.model_name,
        input_size=FLAGS.input_size,
        num_classes=FLAGS.num_classes
    )

    print(f"🔄 Loading weights from: {FLAGS.weights_path}")
    model.load_weights(FLAGS.weights_path)
    print("✅ Weights loaded successfully.")

    export_saved_model(model, FLAGS.saved_model_dir)
    convert_to_tflite(FLAGS.saved_model_dir, FLAGS.tflite_path, FLAGS.export_precision)


if __name__ == '__main__':
    app.run(main)
