# Modified version to generate standard TFRecord format: name-00000-of-00084
# Compatible with multi_stage_train.py `?????-of-XXXXX` pattern

import os
import random
import hashlib
import json
import datetime
import calendar
import tensorflow as tf
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('annotations_file', None, 'JSON file in COCO format')
flags.DEFINE_string('dataset_base_dir', None, 'Path to dataset directory')
flags.DEFINE_string('output_dir', None, 'Path prefix for output TFRecord, e.g. /path/to/inat_val.record')
flags.DEFINE_integer('images_per_shard', 1200, 'Images per shard')
flags.DEFINE_bool('shuffle_images', True, 'Shuffle before writing TFRecord')
flags.DEFINE_string('datetime_format', '%Y-%m-%d %H:%M:%S+00:00', 'Date format')
flags.DEFINE_integer('random_seed', 42, 'Random seed')

flags.mark_flag_as_required('annotations_file')
flags.mark_flag_as_required('dataset_base_dir')
flags.mark_flag_as_required('output_dir')

def _date2float(date):
    dt = datetime.datetime.strptime(date, FLAGS.datetime_format).timetuple()
    year_days = 366 if calendar.isleap(dt.tm_year) else 365
    return dt.tm_yday / year_days

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_tf_example(image, dataset_base_dir, annotations, category_index):
    filename = image['file_name'].split('/')[-1]
    image_path = os.path.join(dataset_base_dir, image['file_name'])
    if not tf.io.gfile.exists(image_path):
        return None

    with tf.io.gfile.GFile(image_path, 'rb') as image_file:
        encoded_image_data = image_file.read()
    key = hashlib.sha256(encoded_image_data).hexdigest()

    height = image['height']
    width = image['width']
    date = _date2float(image['date'])
    if image['latitude'] is None:
        latitude = 0.0
        longitude = 0.0
        valid = 0.0
    else:
        latitude = float(image['latitude']) / 90.0
        longitude = float(image['longitude']) / 180.0
        valid = 1.0

    classes = [ann['category_id'] for ann in annotations]
    classes_text = [category_index[cid]['name'].encode('utf8') for cid in classes]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/latitude': _float_feature(latitude),
        'image/longitude': _float_feature(longitude),
        'image/valid': _float_feature(valid),
        'image/date': _float_feature(date),
        'image/filename': _bytes_feature(filename.encode('utf8')),
        'image/source_id': _bytes_feature(str(image['id']).encode('utf8')),
        'image/key/sha256': _bytes_feature(key.encode('utf8')),
        'image/encoded': _bytes_feature(encoded_image_data),
        'image/format': _bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': _float_list_feature([]),
        'image/object/bbox/xmax': _float_list_feature([]),
        'image/object/bbox/ymin': _float_list_feature([]),
        'image/object/bbox/ymax': _float_list_feature([]),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
    }))

    return tf_example

def create_tf_record(images, annotations_index, dataset_base_dir, category_index, output_prefix):
    num_shards = 1 + (len(images) // FLAGS.images_per_shard)
    output_dir = os.path.dirname(output_prefix)
    base_prefix = os.path.basename(output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    writers = []
    for i in range(num_shards):
        filename = f"{base_prefix}-{i:05d}-of-{num_shards:05d}"
        writers.append(tf.io.TFRecordWriter(os.path.join(output_dir, filename)))

    skipped = 0
    for idx, image in enumerate(images):
        annots = annotations_index.get(image['id'], [])
        tf_example = create_tf_example(image, dataset_base_dir, annots, category_index)
        if tf_example:
            shard = idx % num_shards
            writers[shard].write(tf_example.SerializeToString())
        else:
            skipped += 1

    print(f"Skipped {skipped} images (not found or error)")
    for w in writers:
        w.close()

def main(_):
    random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)

    with tf.io.gfile.GFile(FLAGS.annotations_file, 'r') as f:
        json_data = json.load(f)

    images = json_data['images']
    annotations = json_data.get('annotations', [])
    categories = {cat['id']: cat for cat in json_data['categories']}

    if FLAGS.shuffle_images:
        random.shuffle(images)

    annotations_index = {}
    for ann in annotations:
        annotations_index.setdefault(ann['image_id'], []).append(ann)

    create_tf_record(images, annotations_index, FLAGS.dataset_base_dir, categories, FLAGS.output_dir)

if __name__ == '__main__':
    app.run(main)