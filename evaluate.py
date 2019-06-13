#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:28:08 2019

@author: jayasoo
"""
import os
import pickle
import argparse

import tensorflow as tf
from models import Encoder, Decoder
from train import load_image
from beam_search import beam_search
import config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def evaluate(path):
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    encoder = Encoder(config.embedding_dim)
    decoder = Decoder(config.units, config.embedding_dim, vocab_size)
    
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
        
    image = load_image(path)
    encoder_outputs = encoder(tf.expand_dims(image, 0))
    dec_state = tf.zeros((1, config.units))
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    
    result = result = beam_search(config.beam_width, decoder, dec_input, dec_state, 
                         encoder_outputs, tokenizer.word_index['<end>'], vocab_size)
    
    result = tokenizer.sequences_to_texts([result])
    print(result)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", dest="image")
    args = parser.parse_args()
    evaluate(args.image)