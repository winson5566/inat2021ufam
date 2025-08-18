# Copyright 2021 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Tool to evaluate classifiers."""

import os
import random

from absl import app, flags
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf

import dataloader
import geoprior
import model_builder

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))
flags.DEFINE_integer('input_size', default=224,
    help=('Input size of the model'))
flags.DEFINE_integer('num_classes', default=None,
    help=('Number of classes of the model.'))
flags.DEFINE_integer('batch_size', default=32,
    help=('Batch size used during prediction.'))
flags.DEFINE_string('ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))
flags.DEFINE_string('test_files', default=None,
    help=('A file pattern for TFRecord files'))
flags.DEFINE_bool('use_coordinates_inputs', default=False,
    help=('Use coordinates as aditional input of the model'))
flags.DEFINE_integer('log_frequence', default=500,
    help=('Log prediction every n steps'))
flags.DEFINE_string('results_file', default=None,
    help=('File name where the results will be stored.'))
flags.DEFINE_string('geo_prior_ckpt_dir', default=None,
    help=('Location of the checkpoint files for the geo prior model'))
flags.DEFINE_integer('geo_prior_input_size', default=6,
    help=('Input size for the geo prior model'))
flags.DEFINE_bool('use_bn_geo_prior', default=False,
    help=('Include Batch Normalization to the geo prior model'))
flags.DEFINE_integer('embed_dim', default=256,
    help=('Embedding dimension for geo prior model'))
flags.DEFINE_integer('top_k', default=1,
    help=('Top-k accuracy to report.'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer('random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('num_classes')
flags.mark_flag_as_required('test_files')
flags.mark_flag_as_required('results_file')

# ----------------------------
# Helper functions
# ----------------------------
def _decode_labels(y):
  y = tf.convert_to_tensor(y)
  if y.shape.rank == 2:
    return tf.argmax(y, axis=1).numpy().tolist()
  elif y.shape.rank == 1:
    return y.numpy().tolist()
  else:
    return tf.reshape(y, [-1]).numpy().tolist()

def _decode_preds(logits_or_probs):
  return tf.argmax(logits_or_probs, axis=1).numpy().tolist()

def _decode_topk_preds(logits_or_probs, k):
  return tf.math.top_k(logits_or_probs, k=k).indices.numpy().tolist()

def _fix_input_dtype_for_eval(x):
  x = tf.convert_to_tensor(x)
  if x.shape.rank == 4:
    if x.dtype.is_floating:
      x_max = tf.reduce_max(x)
      x_min = tf.reduce_min(x)
      if (x_max <= 1.0) and (x_min >= 0.0):
        x = tf.cast(tf.round(x * 255.0), tf.uint8)
      else:
        x = tf.cast(tf.clip_by_value(x, 0.0, 255.0), tf.uint8)
    elif x.dtype != tf.uint8:
      x = tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8)
  return x

def _load_model():
  model = model_builder.create(
      model_name=FLAGS.model_name,
      num_classes=FLAGS.num_classes,
      input_size=FLAGS.input_size,
      use_coordinates_inputs=FLAGS.use_coordinates_inputs,
      unfreeze_layers=0)

  for fname in ["ckp.weights.h5", "ckp.h5", "ckp"]:
    p = os.path.join(FLAGS.ckpt_dir, fname)
    if os.path.exists(p):
      print(f"🔄 Loading weights from: {p}")
      model.load_weights(p)
      print("✅ Weights loaded.")
      break
  else:
    raise FileNotFoundError(
        f"No checkpoint file found under {FLAGS.ckpt_dir} "
        "tried: ckp.weights.h5, ckp.h5, ckp"
    )
  return model

def _load_geo_prior_model():
  if FLAGS.geo_prior_ckpt_dir is not None:
    rand_sample_generator = dataloader.RandSpatioTemporalGenerator()
    geo_prior_model = geoprior.FCNet(
      num_inputs=FLAGS.geo_prior_input_size,
      embed_dim=FLAGS.embed_dim,
      num_classes=FLAGS.num_classes,
      use_bn=FLAGS.use_bn_geo_prior,
      rand_sample_generator=rand_sample_generator)
    checkpoint_path = os.path.join(FLAGS.geo_prior_ckpt_dir, "ckp")
    geo_prior_model.load_weights(checkpoint_path)
    return geo_prior_model
  else:
    return None

def build_input_data():
  include_geo_data = FLAGS.geo_prior_ckpt_dir is not None
  input_data = dataloader.TFRecordWBBoxInputProcessor(
    file_pattern=FLAGS.test_files,
    batch_size=FLAGS.batch_size,
    is_training=False,
    output_size=FLAGS.input_size,
    num_classes=FLAGS.num_classes,
    num_instances=0,
    provide_validity_info_output=include_geo_data,
    provide_coord_date_encoded_input=include_geo_data,
    provide_instance_id=True,
    provide_coordinates_input=FLAGS.use_coordinates_inputs)
  dataset, _, _ = input_data.make_source_dataset()
  return dataset

def mix_predictions(cnn_preds, prior_preds, valid):
  valid = tf.expand_dims(valid, axis=-1)
  return cnn_preds*prior_preds*valid + (1 - valid)*cnn_preds

def predict_w_geo_prior(batch, metadata, model, geo_prior_model):
  cnn_input = batch[:-1]
  prior_input = batch[-1]
  label, valid, _ = metadata
  fixed_input = list(cnn_input)
  fixed_input[0] = _fix_input_dtype_for_eval(fixed_input[0])
  cnn_preds = model(fixed_input, training=False)
  prior_preds = geo_prior_model(prior_input, training=False)
  preds = mix_predictions(cnn_preds, prior_preds, valid)
  return label, preds

def top_k_accuracy(y_true, y_pred_topk, k):
  match_count = sum([y in topk for y, topk in zip(y_true, y_pred_topk)])
  return match_count / len(y_true)

def predict_classifier(model, geo_prior_model, dataset):
  labels, predictions, topk_predictions, count = [], [], [], 0
  for batch, metadata in dataset:
    if geo_prior_model is not None:
      label, preds = predict_w_geo_prior(batch, metadata, model, geo_prior_model)
    else:
      if isinstance(batch, (list, tuple)):
        fixed_batch = list(batch)
        fixed_batch[0] = _fix_input_dtype_for_eval(fixed_batch[0])
      else:
        fixed_batch = _fix_input_dtype_for_eval(batch)
      preds = model(fixed_batch, training=False)
      label, _ = metadata

    batch_labels = _decode_labels(label)
    batch_preds  = _decode_preds(preds)
    batch_topk_preds = _decode_topk_preds(preds, FLAGS.top_k)

    labels += batch_labels
    predictions += batch_preds
    topk_predictions += batch_topk_preds

    if count % FLAGS.log_frequence == 0 and count > 0:
      acc1 = accuracy_score(labels, predictions)
      topk_acc = top_k_accuracy(labels, topk_predictions, FLAGS.top_k)
      print(f"[INFO] Finished step {count}, top-{FLAGS.top_k} acc={topk_acc:.4f}, top-1 acc={acc1:.4f}")
    count += 1
  return labels, predictions, topk_predictions

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()
  dataset = build_input_data()
  model = _load_model()
  geo_prior_model = _load_geo_prior_model()
  labels, predictions, topk_predictions = predict_classifier(model, geo_prior_model, dataset)

  acc1 = accuracy_score(labels, predictions)
  topk_acc = top_k_accuracy(labels, topk_predictions, FLAGS.top_k)
  conf_matrix = confusion_matrix(labels, predictions)
  report = classification_report(labels, predictions)

  with open(f"{FLAGS.results_file}.top{FLAGS.top_k}_accuracy", "w") as text_file:
    text_file.write("%s" % topk_acc)
  with open(f"{FLAGS.results_file}.accuracy", "w") as text_file:
    text_file.write("%s" % acc1)
  with open(f"{FLAGS.results_file}.conf_matrix", "w") as text_file:
    text_file.write("%s" % conf_matrix)
  with open(f"{FLAGS.results_file}.classification_report", "w") as text_file:
    text_file.write("%s" % report)

  print("Top-%d Accuracy: %s" % (FLAGS.top_k, topk_acc))
  print("Top-1 Accuracy: %s" % acc1)

if __name__ == '__main__':
  app.run(main)
