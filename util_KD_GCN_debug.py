
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import json
import math
import shutil

import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import pyhocon
import sys

import scipy.sparse as sp
from tensorflow.keras import activations, regularizers, constraints, initializers
from scipy.sparse import csr_matrix
import random_np

from scipy import sparse 
import networkx as nx
import scipy.sparse as sp

np.set_printoptions(threshold=np.inf)
def initialize_from_env(experiment, logdir=None):
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))
    else:
        set_gpus()

    print("Running experiment: {}".format(experiment))

    # xperiments.conf
    config = pyhocon.ConfigFactory.parse_file("experiments_KD_GCN_d.conf")[experiment]
    #config = pyhocon.ConfigFactory.parse_file("experiments_KD_GCN_CRAFT.conf")[experiment]

    if logdir is None:
        logdir = experiment

    config["log_dir"] = mkdirs(os.path.join(config["log_root"], "mgf0208"))
    #config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def initialize_from_env0():
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))
    else:
        set_gpus()

    #name = "predict"
    #name = "att"
    #name = "finalatt"
    #name="final"
    #name = "finalave"
    #name  = "final"
    #name ="best"
    #name = "AVEKB"
    #name = "ATTKB"
    #name ="finalave"
    name ="finalatt"

    print("Running experiment: {}".format(name))

    config = pyhocon.ConfigFactory.parse_file("experiments_KD_GCN_d.conf")[name]
    #config = pyhocon.ConfigFactory.parse_file("experiments_KD_GCN_CRAFT.conf")[name]
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], "mgf0208"))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config
  
  
def get_args():
    parser = ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('-l', '--logdir')
    parser.add_argument('--latest-checkpoint', action='store_true')
    return parser.parse_args()
    
def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def flatten(l):
  return [item for sublist in l for item in sublist]

def set_gpus(*gpus):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
  tf.ConfigProto().gpu_options.per_process_gpu_memory_fraction = 0.5
  tf.ConfigProto().gpu_options.allow_growth = True
  print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def getduijiao():
  
  return random_np.getduijiao()

def get_deplabel():
    
  return random_np.getdeprep()


def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with codecs.open(char_vocab_path, encoding="utf-8") as f:
    vocab.extend(l.strip() for l in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i, c in enumerate(vocab)})
  return char_dict

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)
  
#def gcn_projection(inputs, input_size, output_size, initializer=None):
  """
  output_weights = tf.get_variable("output_weights", [input_size, output_size], initializer=None)  #numwords,20
  output_bias = tf.get_variable("output_bias", [output_size])  #20
  outputs = tf.nn.xw_plus_b(inputs, output_weights, output_bias)
  """
  
def gcn_projection(inputs, output_size, initializer=None):

  hidden_weights = tf.get_variable("hidden_weights", [shape(inputs, 1), 1])
  hidden_bias = tf.get_variable("hidden_bias", [1])
  qt = tf.nn.xw_plus_b(inputs, hidden_weights, hidden_bias)  #numwords
    
  output_weights = tf.get_variable("output_weights", [shape(inputs, 1), output_size])   #20，1
  output_bias = tf.get_variable("output_bias", [output_size])  #1
  outputs = tf.transpose(tf.nn.tanh(tf.nn.xw_plus_b(inputs, output_weights, output_bias) ),(1,0)) #numowrds,1
  
  finalout=tf.expand_dims(tf.reduce_mean(tf.matmul(qt,outputs),0),0)   #1,numwords
  
  
  return finalout

def highway(inputs, num_layers, dropout):
  for i in range(num_layers):
    with tf.variable_scope("highway_{}".format(i)):
      j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
      f = tf.sigmoid(f)
      j = tf.nn.relu(j)
      if dropout is not None:
        j = tf.nn.dropout(j, dropout)
      inputs = f * j + (1 - f) * inputs
  return inputs

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size])
  outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
  return outputs
  
def getvariabe(name,shape):
    v=tf.get_variable(name, shape)
    return v
    
  
def cnn(inputs, filter_sizes, num_filters):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def batch_gather(emb, indices):
  batch_size = shape(emb, 0)
  seqlen = shape(emb, 1)
  if len(emb.get_shape()) > 2:
    emb_size = shape(emb, 2)
  else:
    emb_size = 1
  flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
  offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
  gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]
  if len(emb.get_shape()) == 2:
    gathered = tf.squeeze(gathered, 2) # [batch_size, num_indices]
  return gathered

def sp_matrix_to_sp_tensor(x):
  """
  Converts a Scipy sparse matrix to a SparseTensor.
  :param x: a Scipy sparse matrix.
  :return: a SparseTensor.
  """
  if not hasattr(x, 'tocoo'):
    try:
      x = sp.coo_matrix(x)
    except:
      raise TypeError('x must be convertible to scipy.coo_matrix')
  x = x.tocoo()
  out = tf.SparseTensor(indices=np.array([x.row, x.col]).T,values=x.data,dense_shape=x.shape)
    
  return tf.sparse.reorder(out) 
  
def sparse_dropout(x, keep_prob, noise_shape):
  """
  Dropout for sparse tensors.

  Parameters
  ----------
  x:		Input data
  keep_prob:	Keep probability
  noise_shape:	Size of each entry of x

  Returns
  -------
  pre_out:	x after dropout

  """
  random_tensor  = keep_prob
  random_tensor += tf.random.uniform(noise_shape)
  dropout_mask   = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
  pre_out        = tf.sparse.retain(x, dropout_mask)
  return pre_out * (1./keep_prob)
  
def preprocess_graph(AA):
  num_sent=shape(AA,0)
  max_sent=shape(AA,2)
  _adj=tf.reshape(AA,[num_sent,num_sent,max_sent,max_sent])

  
  _half=tf.reshape(tf.norm(AA,ord=2,axis=3),[num_sent,num_sent,max_sent]) 
  
  _D_half=tf.linalg.diag(_half)
  
  adj_normalized = tf.matmul(tf.matmul(_D_half,_adj),_D_half)
  #print(adj_normalized)
  return adj_normalized
  
def preprocess_features(features):
  """Row-normalize feature matrix and convert to tuple representation"""
  print("preprocess_features")
  
  rowsum =tf.reduce_sum(features,1)
  r_inv = np.power(rowsum, -1)
  #r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = tf.diag(r_inv)
  features =tf.matmul(r_mat_inv,features)
  
  #return features.todense()
  return features
    
    
class GraphAttention(object):
  def __init__(self,
          inputs,
          outsize,
          attn_heads=1,
          attn_heads_reduction='concat',  # {'concat', 'average'}
          dropout_rate=0.5,
          activation='relu',
          use_bias=True,
          w_initializer=tf.glorot_uniform_initializer(),
          bias_initializer=tf.zeros_initializer(),
          aw_initializer=tf.glorot_uniform_initializer(),
          w_regularizer=None,
          bias_regularizer=None,
          aw_regularizer=None):
    if attn_heads_reduction not in {'concat', 'average'}:
      raise ValueError('Possbile reduction methods: concat, average')

    self.inputs=inputs
    self.outsize = outsize  # Number of output features (F' in the paper)
    self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
    self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
    self.dropout_rate = dropout_rate  # Internal dropout rate
    self.activate = activation  # Eq. 4 in the paper

    self.use_bias = use_bias
    

    self.w_initializer = w_initializer
    self.bias_initializer = bias_initializer
    self.aw_initializer = aw_initializer

    self.w_regularizer = w_regularizer
    self.bias_regularizer = bias_regularizer
    self.aw_regularizer = aw_regularizer

    
 
 
  def call(self, inputs):
    X = self.inputs[0]  # Node features (2*4, 3)
    A = self.inputs[1]  # Adjacency matrix (2, 4 ,4)
    D = self.inputs[2]  # Adjacency matrix (2, 4 ,4)
    
    emb = shape(X, -1)  #embsize

    outputs = []
    for head in range(self.attn_heads):
      with tf.variable_scope("head_{}".format(head)):       
      
        
        W=tf.to_float(tf.get_variable("GAT_share_w", [emb, self.outsize],initializer=self.w_initializer, regularizer=self.w_regularizer) )
        b=tf.to_float(tf.get_variable("GAT_share_b", [self.outsize],initializer=self.bias_initializer))
        aW=tf.to_float(tf.get_variable("att_share_w", [self.outsize, 1],initializer=self.aw_initializer, regularizer=self.aw_regularizer)) #8,1
        aW2=tf.to_float(tf.get_variable("att_share_w2", [self.outsize, 1],initializer=self.aw_initializer, regularizer=self.aw_regularizer)) #8,1
        
        
        # Compute inputs to attention network
        features = tf.matmul(X,W)  #(numsent*maxseent, 8 )

        
        # Compute feature combinations
        # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
        attn_for_self = tf.matmul(features,aW)  #numsent*maxseent,1
        attn_for_neighs = tf.matmul(features,aW2)  #numsent*maxseent,1
  
  
  
        # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
        #dense = attn_for_self + tf.transpose(attn_for_neighs,(1,0))  # (numsent*maxseentx numsent*maxseent) via broadcasting  #2*4,2*4
        dense = attn_for_self + tf.transpose(attn_for_neighs,(1,0)) + D
  
        # Add nonlinearty
        dense = tf.nn.leaky_relu(dense) #numsent*maxseent, numsent*maxseent
  
        # Mask values before activation (Vaswani et al., 2017)
        #dense = tf.multiply(dense,D)
        
        mask = -10e9 * (1.0 - A)
        dense += mask
  
        # Apply softmax to get attention coefficients
        dense = tf.nn.softmax(dense)  # (numsent*maxseent x numsent*maxseent) #3,3
        
        
        #dense_print=tf.Print(dense, [ 'dense: ', dense[192:240,192:240]],summarize=3000)
        #dense_print=tf.Print(dense, [ 'dense: ', dense[35:70,35:70]],summarize=3000)
        
  
        # Apply dropout to features and attention coefficients
        dropout_attn = tf.nn.dropout(dense, self.dropout_rate)  # (numsent*maxseent x numsent*maxseent)
        dropout_feat = tf.nn.dropout(features, self.dropout_rate)   # (numsent*maxseent x 8)
        
        #tf.print("dense: ", dense[240:, 240:],output_stream=sys.stderr)
        #tf.print("dense: ", dense[35:70, 35:70],output_stream=sys.stderr)
  
  
        # Linear combination with neighbors' features
        node_features = tf.matmul(dropout_attn, dropout_feat)  # (numsent*maxseent x 8)
        
  
        if self.use_bias:
          node_features = tf.nn.bias_add(node_features,b )  #(numsent*maxseent x 8)
        
        # Add output of attention head to final output
        outputs.append(node_features)
    # Aggregate the heads' output according to the reduction method
    if self.attn_heads_reduction == 'concat':
      output = tf.concat(outputs,1)  # 3,64
      #print(output)
    else:
      output = tf.reduce_mean(tf.stack(outputs),0)   #3,8
       
    if self.activate == 'relu':
      output = tf.nn.relu(output)

    if self.activate == 'softmax':
      output = tf.nn.softmax(output)
    
    
    return output

class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

class EmbeddingDictionary(object):
  def __init__(self, info, normalize=True, maybe_cache=None):
    self._size = info["size"]
    self._normalize = normalize
    self._path = info["path"]
    if maybe_cache is not None and maybe_cache._path == self._path:
      assert self._size == maybe_cache._size
      self._embeddings = maybe_cache._embeddings
    else:
      self._embeddings = self.load_embedding_dict(self._path)

  @property
  def size(self):
    return self._size

  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = np.zeros(self.size)
    embedding_dict = collections.defaultdict(lambda:default_embedding)
    if len(path) > 0:
      vocab_size = None
      with open(path) as f:
        for i, line in enumerate(f.readlines()):
                  
          word_end = line.find("\t")
          word = line[:word_end]
          embedding = np.fromstring(line[word_end + 1:], np.float32, sep=",")
          #print(word)
          #print(embedding)
          
          assert len(embedding) == self.size
          embedding_dict[word] = embedding
      if vocab_size is not None:
        assert vocab_size == len(embedding_dict)
      print("Done loading word embeddings.")
    return embedding_dict

  def __getitem__(self, key):
    embedding = self._embeddings[key]
    if self._normalize:
      embedding = self.normalize(embedding)
    return embedding

  def normalize(self, v):
    norm = np.linalg.norm(v)
    if norm > 0:
      return v / norm
    else:
      return v

class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer
    

