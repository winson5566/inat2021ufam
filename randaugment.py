# coding=utf-8
# Copyright 2021 Fagner Cunha
# Copyright 2019 The Google NoisyStudent Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Fagner Cunha to add support for Tensorflow 2.0.
'''RandAugment policies for enhanced image preprocessing.

RandAugment Reference: https://arxiv.org/abs/1909.13719
'''
import inspect
import math
import tensorflow as tf
from absl import flags


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

FLAGS = flags.FLAGS

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments')
  )

def blend(image1, image2, factor):
  '''Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 'extrapolates' the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  '''
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, dtype=tf.float32)
  image2 = tf.cast(image2, dtype=tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, dtype=tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image, pad_size, replace=0):
  '''Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  '''
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32, seed=FLAGS.random_seed)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32, seed=FLAGS.random_seed)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims, constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.compat.v1.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace,
      image)
  return image


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.compat.v1.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128.
  added_image = tf.cast(image, tf.int64) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.compat.v1.where(image < threshold, added_image, image)


def color(image, factor):
  '''Equivalent of PIL Color.'''
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  '''Equivalent of PIL Contrast.'''
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image, factor):
  '''Equivalent of PIL Brightness.'''
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  '''Equivalent of PIL Posterize.'''
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)

def rotate(image, degrees, replace=[128]):
    radians = tf.cast(degrees, tf.float32) * tf.constant(math.pi / 180.0, tf.float32)
    cos_val = tf.math.cos(radians)
    sin_val = tf.math.sin(radians)

    transform = tf.stack([
        cos_val, -sin_val, 0.0,
        sin_val,  cos_val, 0.0,
        0.0, 0.0
    ])
    transform = tf.expand_dims(transform, 0)

    image = tf.expand_dims(image, 0)
    result = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape=tf.shape(image)[1:3],
        interpolation='BILINEAR',
        fill_value=replace[0]
    )
    return tf.squeeze(result, 0)



def translate_x(image, pixels, replace):
    transform = tf.convert_to_tensor([[1.0, 0.0, -pixels,
                                       0.0, 1.0, 0.0,
                                       0.0, 0.0]], dtype=tf.float32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tf.cast(image, tf.float32), 0),
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="NEAREST",
        fill_value=replace[0])[0]


def translate_y(image, pixels, replace):
    transform = tf.convert_to_tensor([[1.0, 0.0, 0.0,
                                       0.0, 1.0, -pixels,
                                       0.0, 0.0]], dtype=tf.float32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tf.cast(image, tf.float32), 0),
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="NEAREST",
        fill_value=replace[0])[0]

def shear_x(image, level, replace):
    transform = tf.convert_to_tensor([[1.0, level, 0.0,
                                       0.0, 1.0, 0.0,
                                       0.0, 0.0]], dtype=tf.float32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tf.cast(image, tf.float32), 0),
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="NEAREST",
        fill_value=replace[0])[0]

def shear_y(image, level, replace):
    transform = tf.convert_to_tensor([[1.0, 0.0, 0.0,
                                       level, 1.0, 0.0,
                                       0.0, 0.0]], dtype=tf.float32)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(tf.cast(image, tf.float32), 0),
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="NEAREST",
        fill_value=replace[0])[0]


def autocontrast(image):
  '''Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  '''

  def scale_channel(image):
    '''Scale the 2D image using the autocontrast rule.'''
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), dtype=tf.float32)
    hi = tf.cast(tf.reduce_max(image), dtype=tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, dtype=tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image, factor):
  '''Implements Sharpness function from PIL using TF ops.'''
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.compat.v1.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image):
  '''Implements Equalize function from PIL using TF ops.'''
  def scale_channel(im, c):
    '''Scale the data in the channel to implement equalize.'''
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.compat.v1.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def invert(image):
  '''Inverts the image pixels.'''
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image):
  '''Returns 'image' with an extra channel set to all 1s.'''
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended


def unwrap(image, replace):
  '''Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  '''
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = flattened_image[:, 3]

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.compat.v1.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}


def _randomly_negate_tensor(tensor):
  '''With 50% prob turn the tensor negative.'''
  should_flip = tf.cast(tf.floor(tf.random.uniform(
                                  [], seed=FLAGS.random_seed) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  '''Converts level to ratio by which we shrink the image content.'''
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def level_to_arg(hparams):
  return {
      'AutoContrast': lambda level: (),
      'Equalize': lambda level: (),
      'Invert': lambda level: (),
      'Rotate': _rotate_level_to_arg,
      'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
      'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
      'SolarizeAdd': lambda level: (int((level/_MAX_LEVEL) * 110),),
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': lambda level: (int((level/_MAX_LEVEL) * hparams['cutout_const']),),
      # pylint:disable=g-long-lambda
      'TranslateX': lambda level: _translate_level_to_arg(
          level, hparams['translate_const']),
      'TranslateY': lambda level: _translate_level_to_arg(
          level, hparams['translate_const']),
      # pylint:enable=g-long-lambda
  }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
  '''Return the function that corresponds to `name` and update `level` param.'''
  func = NAME_TO_FUNC[name]
  args = level_to_arg(augmentation_hparams)[name](level)

  sig = inspect.signature(func)

  # Add prob if needed
  if 'prob' in sig.parameters:
    args = tuple([prob] + list(args))

  # Add replace if needed
  if 'replace' in sig.parameters:
    assert 'replace' == list(sig.parameters)[-1]
    args = tuple(list(args) + [replace_value])

  return (func, prob, args)



def distort_image_with_randaugment(image, num_layers, magnitude,
                                   cutout_const=40, translate_const=100):
  '''Applies the RandAugment policy to `image`.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [5, 30].

  Returns:
    The augmented version of `image`.
  '''
  replace_value = [128] * 3
  tf.compat.v1.logging.info('Using RandAug.')
  augmentation_hparams = {
      'cutout_const': cutout_const,
      'translate_const': translate_const}
  available_ops = [
      'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
      'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']

  for layer_num in range(num_layers):
    op_to_select = tf.random.uniform(
        [], maxval=len(available_ops), dtype=tf.int32, seed=FLAGS.random_seed)
    random_magnitude = float(magnitude)
    with tf.name_scope('randaug_layer_{}'.format(layer_num)):
      for (i, op_name) in enumerate(available_ops):
        prob = tf.random.uniform(
          [], minval=0.2, maxval=0.8, dtype=tf.float32, seed=FLAGS.random_seed)
        func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                           replace_value, augmentation_hparams)
        image = tf.cond(
            tf.equal(i, op_to_select),
            lambda selected_func=func, selected_args=args: tf.cast(selected_func(image, *selected_args), tf.uint8),
            lambda: image)
  return image
