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
               k=5,
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


    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
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
        cell_output=[self.k, self._cell.output_size],
        time=tensor_shape.TensorShape([]),
        attention=tensor_shape.TensorShape([self.k, self._attention_layer_size]),
        #self._attention_layer_size,
        alignments=tensor_shape.TensorShape([self.k,None]),
        #self._item_or_tuple(
        #    a.alignments_size for a in self._attention_mechanisms),
        alignment_history=tensor_shape.TensorShape([self.k, None]),
        #self._item_or_tuple(
        #    a.alignments_size for a in self._attention_mechanisms),
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
          cell_output = tf.zeros([batch_size, k, self._cell.output_size], dtype=dtype),
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=tf.reshape(_zero_state_tensors(self._attention_layer_size, batch_size*k,
                                        dtype),[batch_size,k,-1]),
          alignments=tf.reshape(self._attention_mechanism.initial_alignments(batch_size*k, dtype),
                               [batch_size,k,-1]), 
          #self._item_or_tuple(
          #    attention_mechanism.initial_alignments(batch_size*k, dtype)
          #    for attention_mechanism in self._attention_mechanisms),
          alphas=self._item_or_tuple([tf.zeros([batch_size,k,self._vocab_size])]),
          alignment_history=tf.reshape(self._attention_mechanism.initial_alignments(batch_size*k, dtype),[batch_size,k,-1]),
                                                
          #self._item_or_tuple(
          #    attention_mechanism.initial_alignments(batch_size*k, dtype)
          #    for attention_mechanism in self._attention_mechanisms)
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
    input_as_indices = tf.stack([ tf.range(tf.shape(inputs)[0]), (tf.cast(inputs, tf.int32))], axis=1)
    temp = tf.reshape(state.alphas, [batch_size, k, self._vocab_size])
    posterior_alpha =  tf.gather_nd(tf.transpose(temp, [0,2,1]), input_as_indices) # [b,k] ->

    previous_alignment = state.alignments
    print ("Cell outut is", state.cell_output, tf.shape(state.cell_output)[2:])
    cell_output = self._stack_beams(state.cell_output)
    history = self._stack_beams(state.alignment_history)
    print ("State cell is:", state.cell_state)
    cell_state = state.cell_state
    # This fn becomes complicated
    attention, alignments, cum_alignments, alpha_with_alignment_score, selected_beams = self.compute_attention(self.attention_mechanism, cell_output, previous_alignment,
                                    history, posterior_alpha, batch_size) 
    #attention is batch,k,features
    #alignment is batch,k,encoder
    #betas is batch,k,1
    alignment_history = cum_alignments 
    #previous_alignment_history.write(state.time, alignments)
    
    input_embedding = tf.nn.embedding_lookup(self._embedding_decoder, inputs)

    print ("EMBEDLOOKUP", tf.contrib.seq2seq.tile_batch(input_embedding,k), self._stack_beams(state.attention))
    cell_state = nest.map_structure(lambda x:  tf.gather_nd(x, selected_beams), cell_state)
    reshaped_attention = self._stack_beams(state.attention)
    reshaped_attention = tf.reshape(reshaped_attention, [-1,self._attention_layer_size])
    cell_input = self._cell_input_fn( tf.contrib.seq2seq.tile_batch(input_embedding,k) , reshaped_attention)
    
    print ("CELLCALL", cell_state, self._stack_beams(cell_state))
    print("CELLCALLQ", cell_input, reshaped_attention)
    cell_output, next_cell_state =  self._cell(cell_input, cell_state)

    output_logits = self._output_layer(cell_output) # [b*k, V]
    import sys
    sys.stdout.flush()
    #print ("OL", output_logits, alphaV_with_alignment_score)
    #alpha_with_alignment_score should be [b*k,1]
    #output_logits should be [b*k,V]
    next_state = JointAttentionState(
        time=state.time + 1,
        cell_state= self._unstack_beams(next_cell_state),
        cell_output= self._unstack_beams(cell_output),
        attention= self._unstack_beams(attention),
        alignments = self._unstack_beams(alignments),
        alphas = self._unstack_beams((tf.nn.log_softmax(output_logits)) + alpha_with_alignment_score),
	#self._item_or_tuple([tf.log(tf.nn.softmax(output_logits)) + alpha_with_alignment_score]),
        alignment_history = self._unstack_beams(alignment_history))
    
    print ("Cell call done", next_state, output_logits)
 
    #return self._unstack_beams(tf.log(tf.nn.softmax(output_logits)) + alpha_with_alignment_score), next_state
    output_probs = tf.nn.softmax(output_logits)
    output_probs_aggregated = tf.reduce_sum(self._unstack_beams(output_probs), axis=1)/self.k
    return tf.log(output_probs_aggregated) , next_state


  def compute_attention(self, attention_mechanism, cell_output, previous_alignment, alignment_history, alphas, batch_size):

    """Computes the attention and alignments for a given attention_mechanism."""
    # SIZES
    # attention_mechanism.values = memory = encoder_state _tensor = [b, memorysize, encoder_dim]
    # cell_output -> [b,k, cellDim]
    # previous_alignments -> [b,k, memory_size]
    k = self.k
    
    context_indices, cumulative_alignments, alignment_scores, selected_beam_index = self.attention_mechanism(
      cell_output, previous_alignment=previous_alignment, history=alignment_history, alphas=alphas, batch_size=batch_size) # prev alignment stuff moved here
    #cumulative is [b*k,memsz]
    #prob_align is [b*k,1]
    #context_indices is [b*k,1]

    print ("CI 1", context_indices) # should be [b,k,1]
    tiled_batch_range = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size,dtype=tf.int32),1),[1,self.k]), [batch_size*self.k,1])
    context_indices = tf.reshape(context_indices, [batch_size*k,1]) 
    context_indices = tf.concat( values=[tiled_batch_range, context_indices] , axis=1)
    #print ("CI", context_indices)
    print ("MEMO", self._memory, context_indices)
    context = tf.gather_nd( self._memory, context_indices) #will be [b*k, embedding_dim]
    
    print ("SETTING SHAPE :", batch_size, self.k, self._attention_layer_size)
    #context.set_shape([None, self._attention_layer_size])
    #context = tf.reshape(context, [batch_size, self.k,self._attention_layer_size ])
    
    
    print ("AH", cumulative_alignments, alignment_history)
    return context, cumulative_alignments - alignment_history, cumulative_alignments, alignment_scores, selected_beam_index 




  def attention_mechanism(self, key, previous_alignment, history, alphas, batch_size):
    #key -> [b*k, celldim]
    #history -> [b*k, memsize]
    #alphas -> [b*k, 1]
    
    k = self.k
    print ("Key s", key , history);
    attention_probs = self._attention_mechanism(key, history)
    attention_probs = tf.check_numerics(attention_probs , message="Attn prob pain again")
    previous_alignment = self._stack_beams(previous_alignment)
    valid_mask = tf.ones_like(previous_alignment)  #filter_mask( previous_alignment, history)
   
    
    mem_size = self._attention_mechanism.alignments_size
    tiled_batch_range = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size,dtype=tf.int32),1),[1,self.k]), [batch_size*self.k,1])


    log_attn_probs = tf.cast((1-valid_mask),tf.float32)*-1e2 + tf.cast((valid_mask),tf.float32)*tf.log(attention_probs + 1e-8) +\
         tf.reshape((alphas),[batch_size*self.k, 1])

    log_attn_probs = tf.check_numerics(log_attn_probs , message="Log Attn prob pain again")
    #log_attn_probs = tf.Print(log_attn_probs, [tf.shape(valid_mask), tf.shape(attention_probs), tf.shape(log_attn_probs), tf.shape(alphas)], message="logattn shapes")
    log_attn_probs = tf.nn.log_softmax(log_attn_probs)
    log_attn_probs_reshaped = tf.reshape(log_attn_probs, [batch_size, k*mem_size])

    #log_attn_probs_reshaped = tf.Print(log_attn_probs_reshaped, [tf.shape(valid_mask), tf.shape(attention_probs), tf.shape(log_attn_probs), tf.shape(alphas)], message="logattn shapes")

    if self._is_training:
      samples = tf.multinomial(log_attn_probs_reshaped, k)
      samples = tf.cast(samples, tf.int32)
      beam_index = tf.reshape(  tf.cast(samples / mem_size, tf.int32), [batch_size*k, 1])
      attention_index = tf.reshape(samples % mem_size, [batch_size*k, 1])
      
      prob_lookup_indices = tf.concat(values=[ tiled_batch_range, tf.reshape(samples,[-1,1])], axis=1)    
      prob_align = tf.gather_nd(log_attn_probs_reshaped, prob_lookup_indices)
      prob_align = tf.reshape(prob_align, [batch_size*k, 1])

    else:
      prob_align, topk_indices = tf.nn.top_k(log_attn_probs_reshaped, k=k)
      beam_index = tf.reshape( tf.cast(topk_indices / mem_size, tf.int32), [batch_size*k, 1])
      #beam_index = tf.cond( tf.reduce_sum(tf.cast(tf.abs(beam_index) >  0, tf.int32)) > 0 , lambda : tf.Print(beam_index, [tf.shape(log_attn_probs_reshaped), log_attn_probs_reshaped, topk_indices, tf.shape(topk_indices)], message="This is problem", summarize=1e5),lambda :beam_index)
      attention_index = tf.reshape(topk_indices % mem_size, [batch_size*k, 1])    
      prob_align = tf.reshape(prob_align, [batch_size*k, 1])


    prob_align = tf.check_numerics(prob_align , message="prob align pain again")
    #Handle cumulative_alignment here
    history_lookup_index = tf.concat(values=[tiled_batch_range, beam_index], axis=1)
    selected_history = tf.gather_nd( self._unstack_beams(history), history_lookup_index)
    #selected_history = tf.Print(selected_history, [tf.shape(selected_history), mem_size, batch_size, k, tf.shape(history),
    #                                               tf.shape(history_lookup_index), tf.shape(tf.one_hot(tf.reshape(attention_index,[batch_size, k]), mem_size))   ], "SH shape is")
    cum_align = selected_history + tf.one_hot(tf.reshape(attention_index,[batch_size*k]), mem_size)
    #cum_align = self._stack_beams(cum_align) 
    # alignment_indices should be [b*k, 1[)
    # selected_history is [b*k, memsize]
    
    print ("SH AI", selected_history, attention_index)
    return attention_index, cum_align, prob_align, history_lookup_index
