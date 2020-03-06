"""
Model definition for CNN sentiment training
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import boto3


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling
    """

    s3 = boto3.client('s3')
    s3_obj = s3.get_object(Bucket="ieor4577-hw6", Key="glove.twitter.27B.25d.txt")
    s3_data = s3_obj['Body'].read().decode('utf-8').split('\n')
    glove = s3_data[:config["embeddings_dictionary_size"]]

    embedding_matrix = np.zeros((config["embeddings_dictionary_size"], config["embeddings_vector_size"]))
    for i in range(config["embeddings_dictionary_size"]):
        if len(glove[i].split()[1:]) != config["embeddings_vector_size"]:
            continue
        embedding_matrix[i] = np.asarray(glove[i].split()[1:], dtype='float32')

    cnn_model = tf.keras.Sequential()
    cnn_model.add(layers.Embedding(weights=[embedding_matrix],
                            input_dim=config['embeddings_dictionary_size'],
                            output_dim=config['embeddings_vector_size'],
                            input_length=config['padding_size']))
    cnn_model.add(layers.Conv1D(filters=100,kernel_size=2,padding='valid',activation='relu',strides=1))
    cnn_model.add(layers.GlobalMaxPooling1D())
    cnn_model.add(layers.Dense(100, activation='relu'))
    cnn_model.add(layers.Dense(1, activation = 'sigmoid'))
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return cnn_model

def save_model(model):
    """
    Method to save models in SaveModel format with signature to allow for serving
    """

    # model.save(os.path.join(output))
    tf.saved_model.save(model, "output/sentiment_model.h5/1")

    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('output/sentiment_model.h5', 'ieor4577-hw6', 'sentiment_model.h5')

    # tf.saved_model.save(model, os.path.join(output, "1"))
    print("Model successfully saved at S3 bucket!")
