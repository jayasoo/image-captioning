#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:20:35 2019

@author: jayasoo
"""

import tensorflow as tf

#CNN encoder
class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.inception = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
        self.inception.trainable = False
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')
    
    def call(self, x):
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        x = self.inception(x)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.fc(x)
        return x
    
class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
#RNN decoder
class Decoder(tf.keras.Model):
    def __init__(self, units, embedding_dim, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.CuDNNGRU(self.units, return_sequences=True, 
                                            return_state=True ,recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        
        self.attention = Attention(self.units)
        
    def call(self, x, hidden, encoder_outputs):
        context_vector, attention_weights = self.attention(hidden, encoder_outputs)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, state = self.gru(x)
        x = self.fc1(x)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights