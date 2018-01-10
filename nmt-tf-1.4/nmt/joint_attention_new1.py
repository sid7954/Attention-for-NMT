# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import functools
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


__all__ = [
    "JointAttentionDecoder",
    "JointAttentionState",
]


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access



class JointAttentionState(
    collections.namedtuple("JointAttentionState",
                           ("cell_output", "cell_state", "attention", "time", "alignments",
                            "alignment_history", "alphas"))):
  """`namedtuple` storing the state of a `JointAttentionDecoder`.
  Contains:
    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `cell_output`: The output of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.
    Example:
    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```
    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.
    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(JointAttentionState, self)._replace(**kwargs)




class JointAttentionDecoder(rnn_cell_impl.RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cell, 
               embedding_decoder,
               vocabulary_size,
               memory,
               cell_input_fn,
               output_layer,
               attn_size,
               source_sequence_length,
               k=1,
               is_training=False,
               name=None):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(JointAttentionDecoder, self).__init__()
    if not cell:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")

    self._cell = cell
    self._embedding_decoder = embedding_decoder

    memory_ex = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=k)
    source_sequence_length_ex = tf.contrib.seq2seq.tile_batch(
          source_sequence_length, multiplier=k)


    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        attn_size, memory_ex , memory_sequence_length=source_sequence_length_ex)

    self._attention_mechanism = attention_mechanism
    self._attention_mechanisms = [attention_mechanism]
    self._memory = memory
    self._output_attention = False
    self._attention_layer_size = sum(
          [attention_mechanism.values.get_shape()[-1].value])
    self.k = k
    self._vocab_size = vocabulary_size
    self._initial_cell_state =  None


    self._attention_layer_fn = layers_core.Dense( self._attention_layer_size , name="attention_tranform_layer", use_bias=False)
    
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)
        
    self._cell_input_fn = cell_input_fn
    self._output_layer = output_layer
    self._is_training = is_training
    
    def _stack_beam_helper(t):
        shape_t = t.get_shape().as_list()
        t = tf.reshape( t,tf.concat(axis=0, values=[ [tf.shape(t)[0]*self.k], tf.shape(t)[2:]]))
        t.set_shape([None] + shape_t[2:])
        return t
    
    def _unstack_beam_helper(t):
        shape_t = t.get_shape().as_list()
        t = tf.reshape( t,tf.concat(axis=0, values=[ [-1], [self.k], tf.shape(t)[1:]]))
        t.set_shape([None, self.k] + shape_t[1:])
        return t
    
    
    self._stack_beams = lambda inp: nest.map_structure(lambda s: _stack_beam_helper(s), inp)
    self._unstack_beams = lambda inp: nest.map_structure(lambda s: _unstack_beam_helper(s), inp)
    
  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
                                   attention_mechanism.batch_size,
                                   message=error_message)
            for attention_mechanism in self._attention_mechanisms]

  def _item_or_tuple(self, seq):
    """Returns `seq` as tuple or the singular element.

    Which is returned is determined by how the AttentionMechanism(s) were passed
    to the constructor.

    Args:
      seq: A non-empty sequence of items or generator.

    Returns:
       Either the values in the sequence as a tuple if AttentionMechanism(s)
       were passed to the constructor as a sequence or the singular element.
    """
    t = tuple(seq)
    return t[0]
    

  @property
  def output_size(self):
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    return JointAttentionState(
        cell_state=nest.map_structure(lambda s: [self.k, s] , self._cell.state_size),
        cell_output=[self.k, self._output_layer.units],
        time=tensor_shape.TensorShape([]),
        attention=tensor_shape.TensorShape([self.k, self._attention_layer_size]),
        alignments=tensor_shape.TensorShape([self.k,None]),
        alignment_history=tensor_shape.TensorShape([self.k, None]),
        alphas=tensor_shape.TensorShape([]))  # sometimes a TensorArray
  
  def zero_state(self, batch_size, dtype):
    k = self.k
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size*k, dtype)
        print ("ZSSS", cell_state)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          self._batch_size_checks(batch_size*k, error_message)):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      #print ("JSZERO", self._cell.output_size);
      JS_ZERO  = JointAttentionState(
          cell_state= self._unstack_beams(cell_state),
          cell_output = tf.zeros([batch_size, k, self._output_layer.units], dtype=dtype),
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=tf.reshape(_zero_state_tensors(self._attention_layer_size, batch_size*k,
                                        dtype),[batch_size,k,-1]),
          alignments=tf.reshape(self._attention_mechanism.initial_alignments(batch_size*k, dtype),
                               [batch_size,k,-1]), 
          alphas=self._item_or_tuple([tf.zeros([batch_size,k,self._vocab_size])]),
          alignment_history=tf.reshape(self._attention_mechanism.initial_alignments(batch_size*k, dtype),[batch_size,k,-1]),
                                                
          )
      print ("JS_ZER)", JS_ZERO)
      return JS_ZERO
  
  def call (self , inputs, state):
    for i in range(1):
        print ("Basecall", i)
        _,state = self.call1(inputs, state)
    return _, state

  def call2 (self, inputs, state):
      input_embedding = tf.nn.embedding_lookup(self._embedding_decoder, inputs)
      cstate = self._stack_beams(state.cell_state)
      print ("BLABLA", state)
      input_embedding = tf.contrib.seq2seq.tile_batch(input_embedding,self.k)
      cell_output, next_cell_state = self._cell(input_embedding, cstate)

      output_logits = self._output_layer(cell_output) # [b*k, V]
      next_state = JointAttentionState(
        time=state.time + 1,
        cell_state=self._unstack_beams(next_cell_state),
        cell_output=self._unstack_beams(cell_output),
        attention=state.attention,
        alignments=state.alignments,
        alphas = state.alphas,
        alignment_history = state.alignment_history)
      return output_logits, next_state 
 

      
  def call1(self, inputs, state):
    """Perform a step of attention-wrapped RNN.
                                                                                                                  
    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the feature vectors from alignments and the attention_mechanism's values (memory).
    """
    # SIZES
    # state.attention -> feature vectors of segment/attented input/context 
    # [b,k, FVdim]
    # state.alignments -> masks/attention vectors of alignments till now
    # [b,k, inputs]
    # state.alphas -> probabilities of alignments till now
    # [b,k,V]
    


    # Step 1: Calculate the inputs to the cell based on input received and attention

    print ("Cell call begins", state)
    k = self.k
    batch_size = tf.shape(inputs)[0]
    previous_alignment = state.alignments
    print ("Cell outut is", state.cell_output, tf.shape(state.cell_output)[2:])
    cell_output = self._stack_beams(state.cell_output)
    history = self._stack_beams(state.alignment_history)
    print ("State cell is:", state.cell_state)
    cell_state = state.cell_state
    # This fn becomes complicated
    
    input_embedding = tf.nn.embedding_lookup(self._embedding_decoder, inputs)

    reshaped_attention = self._stack_beams(state.attention)
    reshaped_attention = tf.reshape(reshaped_attention, [-1,self._attention_layer_size])
    cell_input = self._cell_input_fn( tf.contrib.seq2seq.tile_batch(input_embedding,k) , reshaped_attention)


    print ("EMBEDLOOKUP", tf.contrib.seq2seq.tile_batch(input_embedding,k), self._stack_beams(state.attention))
    cell_state = self._stack_beams(cell_state)
    cell_output, next_cell_state =  self._cell(cell_input, cell_state)

    key = cell_output
    attention_probs = self._attention_mechanism(key, history)
    attention_probs = tf.check_numerics(attention_probs , message="Attn prob pain")

    memory = self._attention_mechanism.values

    cell_output_expanded = tf.tile(tf.expand_dims(cell_output, 1), tf.stack([1, tf.shape(self._attention_mechanism.values)[-2] , 1]))

    memory_output = tf.concat( values= [ self._attention_mechanism.values,
                                         cell_output_expanded ], axis=-1)
    memory_output = tf.reshape(memory_output, tf.stack( [-1, self._attention_mechanism.values.shape[-1] + cell_output.shape[-1]]) )
    memory_output = self._attention_layer_fn(memory_output)
    memory_output_reshaped = tf.reshape(memory_output, [batch_size, -1, self._attention_layer_fn.units])


    if False:
      output_logits = self._output_layer(memory_output)
      output_probs = tf.nn.softmax(output_logits)
      output_probs = tf.reshape(output_probs, [batch_size, -1, self._output_layer.units])
      output_probs = tf.reduce_sum(output_probs*tf.reshape(attention_probs,[batch_size, -1, 1]), 1)
      output_logits = tf.log(output_probs)
      attention =  tf.reduce_sum(memory_output_reshaped*tf.reshape(attention_probs,[batch_size, -1, 1]),1)
    else:

      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      expanded_alignments = array_ops.expand_dims(attention_probs, 1)
      # Context is the inner product of alignments and values along the
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, memory_size]
      #   [batch_size, 1, memory_size].
      # we then squeeze out the singleton dim.
      context = math_ops.matmul(expanded_alignments, self._attention_mechanism.values)
      context = array_ops.squeeze(context, [1])

      attention = self._attention_layer_fn( tf.concat( axis=1, values = [cell_output, context])) 
      output_logits = self._output_layer(attention)



    alignments = attention_probs
    cell_output = output_logits
    #print ("OL", output_logits, alphaV_with_alignment_score)
    #alpha_with_alignment_score should be [b*k,1]
    #output_logits should be [b*k,V]
    next_state = JointAttentionState(
        time=state.time + 1,
        cell_state= self._unstack_beams(next_cell_state),
        cell_output= self._unstack_beams(cell_output),
        attention= self._unstack_beams(attention),
        alignments = self._unstack_beams(alignments),
        alphas = state.alphas,
	#self._item_or_tuple([tf.log(tf.nn.softmax(output_logits)) + alpha_with_alignment_score]),
        alignment_history = state.alignment_history)
    
    print ("Cell call done", next_state, output_logits)
 
    return output_logits , next_state

