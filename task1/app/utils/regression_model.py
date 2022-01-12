import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf

tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float32')
print(tf.__version__)

class MyModel(tf.keras.Model):

  def __init__(self,inputs,units,outputs,seed=None,name='dummy',**kwargs):
    super().__init__(name=name,**kwargs)
    if seed:
      tf.random.set_seed(seed=seed)

    # Inputs
    self._inputs = {}
    for key in inputs:
      self._inputs[key] = tf.keras.layers.Dense(units=units['input'],activation='linear',input_shape=(inputs[key],),name='input_layer_'+key,dtype=tf.float32)

    # Outputs
    n = []
    for key in outputs:
      n = n+[len(outputs[key])]
    self._outputs = dict(zip(list(outputs.keys()),n))

    # Layers
    self._layers = []
    for n in units['hidden']:
      self._layers = self._layers+[tf.keras.layers.Dense(units=n,activation='relu',name='hidden_layer_'+str(n),dtype=tf.float32)]

    n = sum(list(self._outputs.values()))
    self._layers = self._layers+[tf.keras.layers.Dense(units=n,activation='linear',name='output_layer',dtype=tf.float32)]

###
  def __call__(self,inputs):

    # Inputs
    x = []
    for key in self._inputs:
      x = x+[self._inputs[key](inputs[key])]
    print(x)
    x = tf.keras.layers.Concatenate()(x)

    # Network
    for layer in self._layers:
      x = layer(x)

    # Outputs
    n = list(self._outputs.values())
    x = tf.exp(x)
    y = dict(zip(list(self._outputs.keys()),tf.split(x,n,axis=1)))

    return y