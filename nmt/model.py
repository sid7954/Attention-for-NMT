# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

from tensorflow.python.layers import core as layers_core
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


'''
 inputs = decoder_state

  def _logit(self, inputs, char_index):
    """Creates logits for a character if required.

    Args:
      inputs: A tensor with shape [batch_size, ?] (depth is implementation
        dependent).
    Returns:
      A tensor with shape [batch_size, num_classes]
    """
    if char_index not in self._char_logits:
      inputs = tf.nn.tanh(self.encoder_parts + tf.expand_dims( tf.matmul(inputs, self._dW) + self._db, dim=1)) #batch, items, emb
      _shape = self._net.shape.as_list()
      logits = tf.reshape(tf.nn.xw_plus_b ( tf.reshape(inputs, [_shape[0]*_shape[1], self._mparams.num_lstm_units]),  self._softmax_w, self._softmax_b), 
              [_shape[0], _shape[1], self._params.num_char_classes])
      #batch, items, vocab

      self._char_logits[char_index] = tf.reduce_logsumexp(logits, 1)
    return self._char_logits[char_index]

Might need to append multiple steps output together, use this if required

lstm_ouptuts = """list of decoder outputs"""
    with tf.variable_scope('logits'):
      logits_list = [
          tf.expand_dims(self.char_logit(logit, i), dim=1)
          for i, logit in enumerate(lstm_outputs)
      ]
'''


class OurDense(layers_core.Dense):
  """Densely-connected layer class.
  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).
  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.
  Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               encoder_states=None,
               embedding_encoder=None,
               **kwargs):
    super(layers_core.Dense, self).__init__(trainable=trainable, name=name, **kwargs)
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = base.InputSpec(min_ndim=2)
    self.encoder_states=encoder_states
    self.embedding_encoder=embedding_encoder

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    self.kernel = self.add_variable('kernel',
                                    shape=[input_shape[-1].value, self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    #constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    self.w_kernel = self.add_variable('w_kernel',
                                    shape=[self.encoder_states.shape[-1], self.embedding_encoder.shape[-1]],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    #constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    #constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    #ofr i in self.encoder_parts:

    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    encoder_parts= ops.convert_to_tensor(self.encoder_states, dtype=self.dtype)
    print (encoder_parts, type(encoder_parts), inputs, self.encoder_states)
    _encoder_shape = encoder_parts.shape
    _decoder_shape = inputs.shape 

    encoder_parts=tf.reshape(encoder_parts,[-1,_encoder_shape[2].value])
    encoder_parts = tf.matmul(encoder_parts,self.w_kernel)
    #print (_shape[0]*_shape[1],_shape[2])
    self.logits_calculated = []
    decoder_states = tf.unstack(inputs, axis=1)
    for decoder_state in decoder_states:
      ed_states = tf.nn.tanh(encoder_parts + tf.expand_dims( tf.matmul(decoder_state, self.kernel) + self._db, dim=1)) #batch, items, emb
      logits = tf.reshape(tf.nn.xw_plus_b ( tf.reshape(ed_states, [_encoder_shape[0]*_encoder_shape[1], self.embedding_encoder.shape[-1]]),  self.kernel, self.bias), 
            [_encoder_shape[0], _encoder_shape[1],units])
    #batch, items, vocab
    self.logits_calculated.append(tf.reduce_logsumexp(logits, 1))

    return tf.stack(self.logits_calculated,1)
    # inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    # shape = inputs.get_shape().as_list()
    # if len(shape) > 2:
    #   # Broadcasting is required for the inputs.
    #   outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
    #                                                          [0]])
    #   # Reshape the output back to the original ndim of the input.
    #   if context.in_graph_mode():
    #     output_shape = shape[:-1] + [self.units]
    #     outputs.set_shape(output_shape)
    # else:
    #   outputs = standard_ops.matmul(inputs, self.kernel)
    # if self.use_bias:
    #   outputs = nn.bias_add(outputs, self.bias)
    # if self.activation is not None:
    #   return self.activation(outputs)  # pylint: disable=not-callable
    # return outputs

class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               single_cell_fn=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      single_cell_fn: allow for adding customized cell. When not specified,
        we default to model_helper._single_cell
    """
    assert isinstance(iterator, iterator_utils.BatchedInput)
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.num_layers = hparams.num_layers
    self.num_gpus = hparams.num_gpus
    self.time_major = hparams.time_major

    # Initializer
    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    # TODO(ebrevdo): Only do this if the mode is TRAIN?
    self.init_embeddings(hparams, scope)
    self.batch_size = tf.size(self.iterator.source_sequence_length)

    # Projection
    # with tf.variable_scope(scope or "build_network"):
    #   with tf.variable_scope("decoder/output_projection"):
    #     self.output_layer = layers_core.Dense(
    #         hparams.tgt_vocab_size, use_bias=False, name="output_projection")

    # To make it flexible for external code to add other cell types
    # If not specified, we will later use model_helper._single_cell
    self.single_cell_fn = single_cell_fn

    ## Train graph
    res = self.build_graph(hparams, scope=scope)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.final_context_state, self.sample_id = res
      self.sample_words = reverse_target_vocab_table.lookup(
          tf.to_int64(self.sample_id))

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

    ## Learning rate
    print("  start_decay_step=%d, learning_rate=%g, decay_steps %d,"
          "decay_factor %g" % (hparams.start_decay_step, hparams.learning_rate,
                               hparams.decay_steps, hparams.decay_factor))
    self.global_step = tf.Variable(0, trainable=False)

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      if hparams.optimizer == "sgd":
        self.learning_rate = tf.cond(
            self.global_step < hparams.start_decay_step,
            lambda: tf.constant(hparams.learning_rate),
            lambda: tf.train.exponential_decay(
                hparams.learning_rate,
                (self.global_step - hparams.start_decay_step),
                hparams.decay_steps,
                hparams.decay_factor,
                staircase=True),
            name="learning_rate")
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        tf.summary.scalar("lr", self.learning_rate)
      elif hparams.optimizer == "adam":
        assert float(
            hparams.learning_rate
        ) <= 0.001, "! High Adam learning rate %g" % hparams.learning_rate
        self.learning_rate = tf.constant(hparams.learning_rate)
        opt = tf.train.AdamOptimizer(self.learning_rate)

      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)

      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.train_loss),
      ] + gradient_norm_summary)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Print trainable variables
    utils.print_out("# Trainable variables")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=hparams.num_units,
            tgt_embed_size=hparams.num_units,
            num_partitions=hparams.num_embeddings_partitions,
            scope=scope,))

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update,
                     self.train_loss,
                     self.predict_count,
                     self.train_summary,
                     self.global_step,
                     self.word_count,
                     self.batch_size])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
                     self.predict_count,
                     self.batch_size])

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss, final_context_state),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: the total loss / batch_size.
        final_context_state: The final state of decoder RNN.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32
    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
      # Encoder
      encoder_outputs, encoder_state = self._build_encoder(hparams)
      
      self.encoder_parts=encoder_outputs
      # Projection
      with tf.variable_scope(scope or "build_network"):
        with tf.variable_scope("decoder/output_projection"):
          self.output_layer = OurDense(
            hparams.tgt_vocab_size, 
            use_bias=False, 
            name="output_projection",
            encoder_states=self.encoder_parts,
            embedding_encoder=self.embedding_encoder)

      ## Decoder
      logits, sample_id, final_context_state = self._build_decoder(
          encoder_outputs, encoder_state, hparams)

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
          loss = self._compute_loss(logits)
      else:
        loss = None

      return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                          base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""

    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)

    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(iterator.source_sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
          logits = self.output_layer(outputs.rnn_output)

      ## Inference
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)
        else:
          # Helper
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder
        and the intial state of the decoder RNN.
    """
    pass

  def _compute_loss(self, logits):
    """Compute optimization loss."""
    target_output = self.iterator.target_output
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(self.batch_size)
    return loss

  def _get_infer_summary(self, hparams):
    return tf.no_op()

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    return sess.run([
        self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
    ])

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    _, infer_summary, _, sample_words = self.infer(sess)

    # make sure outputs is of shape [batch_size, time]
    if self.time_major:
      sample_words = sample_words.transpose()
    return sample_words, infer_summary


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder(self, hparams):
    """Build an encoder."""
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    iterator = self.iterator

    source = iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(
          self.embedding_encoder, source)

      # Encoder_outpus: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(
            hparams, num_layers, num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=self.time_major)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=iterator.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
    return encoder_outputs, encoder_state

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state

class Model2(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder(self, hparams):
    """Build an encoder."""
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    iterator = self.iterator

    source = iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(
          self.embedding_encoder, source)

      # Encoder_outpus: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(
            hparams, num_layers, num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=self.time_major)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=iterator.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
    return encoder_outputs, encoder_state

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)

    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(iterator.source_sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,
            output_layer=self.output_layer)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
          logits = self.output_layer(outputs.rnn_output)

      ## Inference
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)
        else:
          # Helper
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
