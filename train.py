#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:18:12 2019

@author: jayasoo
"""

import os
import json
import pickle

import tensorflow as tf
from models import Encoder, Decoder
import config

tf.enable_eager_execution()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299,299))
    return img

def loss_fn(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss.dtype)
    masked_loss = loss * mask
    return tf.reduce_mean(masked_loss)

def train_step(inputs, targets, encoder, decoder, optimizer, tokenizer):
    dec_state = tf.zeros((targets.shape[0], config.units))
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * config.BATCH_SIZE, 1)
    loss = 0
    with tf.GradientTape() as tape:
        encoder_outputs = encoder(inputs)
        for i in range(1, targets.shape[1]):
            dec_output, dec_state, att_weights = decoder(dec_input, dec_state, encoder_outputs)
            loss += loss_fn(targets[:,i], dec_output)
            dec_input = tf.expand_dims(targets[:,i],1)
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

def train():
    with open('./annotations/captions_train2017.json') as f:
        annotations = json.load(f)
    
    captions = []
    image_paths = []
    for annot in annotations['annotations']:
        caption = "<start> " + annot['caption'] + " <end>"
        image_id = annot['image_id']
        image_path = "./train2017/" + '%012d.jpg' % (image_id)
        image_paths.append(image_path)
        captions.append(caption)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    train_sequences = tokenizer.texts_to_sequences(captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>' 

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)                                                 

    train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post')
    
    train_ds = tf.data.Dataset.from_tensor_slices((image_paths[:30000], train_sequences[:30000]))
    train_ds = train_ds.map(lambda x, y: (load_image(x), y))
    
    num_steps = len(captions) // config.BATCH_SIZE
    vocab_size = len(tokenizer.word_index) + 1
    
    train_ds = train_ds.shuffle(1000).batch(config.BATCH_SIZE)
    
    encoder = Encoder(config.embedding_dim)
    decoder = Decoder(config.units, config.embedding_dim, vocab_size)
    
    optimizer = tf.train.AdamOptimizer()
    
    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    
    for epoch in range(config.EPOCH):
        loss = 0
        for batch, (inputs, targets) in enumerate(train_ds):
            batch_loss = train_step(inputs, targets, encoder, decoder, optimizer, tokenizer)
            loss += batch_loss
            
            if batch % 50 == 0:
                print("Epoch {} batch {} loss {:.4f}".format(epoch+1, batch, batch_loss / int(targets.shape[1])))
        
        if epoch % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print("Epoch {} loss {:.6f}".format(epoch+1, loss/num_steps))
    
if __name__ == '__main__':
    train()