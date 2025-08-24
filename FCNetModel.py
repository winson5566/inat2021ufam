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

def _create_res_layer(inputs, embed_dim, use_bn=False):
  x = tf.keras.layers.Dense(embed_dim)(inputs)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(rate=0.5)(x)
  x = tf.keras.layers.Dense(embed_dim)(x)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  outputs = tf.keras.layers.add([inputs, x])

  return outputs

def _create_loc_encoder(inputs, embed_dim, num_res_blocks, use_bn=False):
  x = tf.keras.layers.Dense(embed_dim)(inputs)
  if use_bn:
    x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  for _ in range(num_res_blocks):
    x = _create_res_layer(x, embed_dim)

  return x

def _create_FCNet(num_inputs,
                  num_classes,
                  embed_dim,
                  num_users,
                  num_res_blocks=4,
                  use_bn=False):
  inputs = tf.keras.Input(shape=(num_inputs,))
  loc_embed = _create_loc_encoder(inputs, embed_dim, num_res_blocks, use_bn)
  class_embed_layer = tf.keras.layers.Dense(num_classes,
                                            activation='sigmoid',
                                            use_bias=False)
  class_embed = class_embed_layer(loc_embed)

  if num_users > 0:
    user_emb_layer = tf.keras.layers.Dense(num_users,
                                           activation='sigmoid',
                                           use_bias=False)
    user_emb = user_emb_layer(loc_embed)
    outputs = [class_embed, user_emb]
  else:
    outputs = [class_embed]
    user_emb_layer = None

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  return model, class_embed_layer, user_emb_layer

class FCNet(tf.keras.Model):
  def __init__(self, num_inputs, embed_dim, num_classes, rand_sample_generator,
                num_users=0, num_res_blocks=4, use_bn=False):
    super(FCNet, self).__init__()
    
    model, class_embed, user_emb = _create_FCNet(
                               num_inputs, num_classes, embed_dim, num_users,
                               num_res_blocks=num_res_blocks, use_bn=use_bn)
    self.model = model
    self.class_embed = class_embed
    self.user_emb = user_emb
    self.rand_sample_generator = rand_sample_generator

  def call(self, inputs):
    return self.model(inputs)

  def compile(self, optimizer, loc_o_loss, loc_p_loss=None, p_o_loss=None):
    super(FCNet, self).compile()
    self.fc_optimizer = optimizer
    self.loc_o_loss = loc_o_loss
    self.loc_p_loss = loc_p_loss
    self.p_o_loss = p_o_loss

    if self.user_emb is not None:
      if loc_p_loss is None or p_o_loss is None:
        raise RuntimeError('Users branch loss cannot be None')

  def train_step(self, data):
    x, y = data
    batch_size = tf.shape(x)[0]

    if self.user_emb is not None:
      y_class_true = y[0]
      y_user_true = y[1]
    else:
      y_class_true = y
      y_user_true = None

    rand_samples = self.rand_sample_generator.get_rand_samples(batch_size)
    combined_inputs = tf.concat([x, rand_samples], axis=0)
    
    # The localization loss on the paper for the random points is equivalent to
    # the Binary Cross Entropy considering all labels as zero
    rand_labels = tf.zeros(shape=y_class_true.shape)

    with tf.GradientTape() as tape:
      preds = self(combined_inputs, training=True)
      total_loss = 0

      if self.user_emb is not None:
        y_pred = preds[0]
        y_pred_user = preds[1]

        phot_loss = self.loc_p_loss(y_user_true, y_pred_user[:batch_size])

        # For the random points
        y_pred_user_rand = 1 - y_pred_user[:batch_size]
        phot_loss_rand = self.loc_p_loss(y_user_true, y_pred_user_rand)

        # User class loss is equivalent to filter the category/user affinity
        # using the user labels and then applying the weighted binary cross
        # entropy loss considering the object labels as the true label
        p_c_given_u = tf.matmul(y_user_true,
                                self.user_emb.kernel,
                                transpose_b=True)
        p_c_given_u = tf.matmul(p_c_given_u, self.class_embed.kernel)
        p_c_given_u = tf.sigmoid(p_c_given_u)
        phot_obj_loss = self.p_o_loss(y_class_true, p_c_given_u)

        total_loss = phot_loss + phot_loss_rand + phot_obj_loss

      else:
        y_pred = preds

      obj_loss = self.loc_o_loss(y_class_true, y_pred[:batch_size])
      obj_loss_rand = self.loc_o_loss(rand_labels, y_pred[batch_size:])
      total_loss = total_loss + obj_loss + obj_loss_rand 

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)
    
    self.fc_optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(y, y_pred)
    metrics = {m.name: m.result() for m in self.metrics}
    metrics['loss'] = total_loss
    metrics['obj_loss'] = obj_loss
    metrics['obj_loss_rand'] = obj_loss_rand
    if self.user_emb is not None:
      metrics['phot_loss'] = phot_loss
      metrics['phot_loss_rand'] = phot_loss_rand
      metrics['phot_obj_loss'] = phot_obj_loss

    return metrics

  def test_step(self, data):
    x, y = data
    y_pred = self(x, training=False)

    if self.user_emb is not None:
      y_pred_class = y_pred[0]
      y_true = y[0]
    else:
      y_pred_class = y_pred
      y_true = y

    loss = self.loc_o_loss(y_true, y_pred_class)

    self.compiled_metrics.update_state(y_true, y_pred_class)
    metrics = {m.name: m.result() for m in self.metrics}
    metrics['loss'] = loss
    metrics['obj_loss'] = loss

    return metrics
