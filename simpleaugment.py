# Copyright 2021 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import math
import utils

def distort_color(image, seed=None):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.random_brightness(image, max_delta=32. / 255., seed=seed)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed)
    image = tf.image.random_hue(image, max_delta=0.2, seed=seed)
    return tf.clip_by_value(image, 0.0, 1.0)

def wrap(image):
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], dtype=image.dtype)
    return tf.concat([image, extended_channel], axis=2)

def unwrap(image, replace):
    shape = tf.shape(image)
    flat = tf.reshape(image, [-1, 4])
    alpha = flat[:, 3]
    replace = tf.concat([replace, tf.ones([1], image.dtype)], axis=0)
    flat = tf.where(tf.equal(alpha, 0), tf.ones_like(flat) * replace, flat)
    image = tf.reshape(flat, shape)
    return image[:, :, :3]

def rotate_affine(image, degrees, replace):
    radians = degrees * math.pi / 180.0
    cos_val = tf.math.cos(radians)
    sin_val = tf.math.sin(radians)
    transform = [
        cos_val, -sin_val, 0.0,
        sin_val,  cos_val, 0.0,
        0.0, 0.0
    ]
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(wrap(image), 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=[transform],
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_value=replace[0] if isinstance(replace, (list, tuple)) else replace
    )
    return unwrap(tf.cast(rotated[0], tf.uint8), replace)

def random_rotation(image, deg=20, seed=None):
    rotation_theta = utils.deg2rad(deg)
    random_deg = tf.random.uniform([], minval=-rotation_theta * 180.0 / math.pi,
                                        maxval=rotation_theta * 180.0 / math.pi,
                                        seed=seed)
    return rotate_affine(image, random_deg, replace=[128])

def distort_image_with_simpleaugment(image, seed=None):
    tf.compat.v1.logging.info('Using SimpleAug.')
    image = distort_color(image, seed=seed)
    image = random_rotation(image, seed=seed)
    return image
