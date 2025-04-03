import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_catagorical
import os
import pickle
from tqdm import tqdm
from PIL import Image
import nltk

def load_descriptions(filename):

    file = open(filename, 'r')
    text = file.read()
    file.close()

    descriptions = {}

    for line in text.split('\n'):
        tokens = line.split('\t')

        if(len(tokens) < 2): 
            continue

        image_id, caption = tokens[0].split('#')[0], tokens[1]
        caption = caption.lower().strip()

        if(image_id not in descriptions):
            descriptions[image_id] = []
        
        descriptions[image_id].append(f'<start> {caption} <end>')
    return descriptions

def build_tokenizer(descriptions, vocab_size = 5000):
    captions = list(caption for captions in descriptions.values() for caption in captions)
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

def extract_features(image_path, model):
    image = Image.open(image_path).resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis= 0)
    features = model.predict(image)
    return features

def transformer_data_generator(descriptions, photos, tokenizer, max_length, batch_size, vocab_size):
    x_img, x_seq, y = [], [], []
    n = 0
    
    while True:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]  # Extract CNN feature vector
            
            for description in desc_list:
                sequence = tokenizer.texts_to_sequences([description])[0]
                sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
                
                input_seq = sequence[:-1]  # All words except last one
                output_seq = sequence[1:]  # Shifted sequence

                output_seq = to_categorical(output_seq, num_classes=vocab_size)

                x_img.append(photo)
                x_seq.append(input_seq)
                y.append(output_seq)

                n += 1
                if n == batch_size:
                    yield [np.array(x_img), np.array(x_seq)], np.array(y)
                    x_img, x_seq, y = [], [], []
                    n = 0
