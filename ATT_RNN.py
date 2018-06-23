"""
Author: Xin-Xue Lin
e-mail: f05922134@ntu.edu.tw
"""

import json
import sys
import re
import numpy as np
import tensorflow as tf
import random as rn
import keras
import os
os.environ['PYTHONHASHSEED'] = '0'
from keras.layers import Embedding, LSTM, Conv1D, BatchNormalization, Multiply, Permute, Dot
from keras.layers import Dropout, MaxPooling1D, GlobalMaxPooling1D, Lambda, RepeatVector
from keras.layers import Input, Activation, Bidirectional, GRU, Dense, CuDNNGRU, CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, '')
MAX_SEQUENCE_LENGTH = 100  # max input sequence length
EMBEDDING_DIM = 300  # word embedding size
VALIDATION_SPLIT = 0.1  # validation ratio


def RNN_model(input_layer, num_class):  # RNN model
    def smoothing_attention(x):
        e = K.sigmoid(x)
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    reg = 0.0001
    dropout = 0.5
    hidden_dim = 1024
    vector = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=False, kernel_regularizer=keras.regularizers.l2(reg)))(input_layer)
    # vector = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=False, kernel_regularizer=keras.regularizers.l2(reg)))(vector)
    lstm = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg)))(input_layer)
    # lstm = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True, kernel_regularizer=keras.regularizers.l2(reg)))(lstm)
    ee = Dot(axes=-1, normalize=True)([vector, lstm])
    weights = Lambda(smoothing_attention)(ee)
    weights = RepeatVector(2*hidden_dim)(weights)
    weights = Permute((2, 1))(weights)
    output = Multiply()([weights, lstm])
    output = Lambda(lambda x: K.sum(x, axis=1))(output)
    output = Dense(512)(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Dense(256)(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Dropout(dropout)(output)
    output = Dense(num_class, activation='softmax')(output)
    model = Model(sequence_input, output)
    print(model.summary())
    return model


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use GPU 0
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt')) as f:  # read pre-trained word embedding
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    X_train = []
    Y_train = []
    label_to_y = dict()
    with open("TRAIN_FILE.txt") as f:
        for idx, l in enumerate(f):
            l = l.strip()
            if idx % 4 == 0:
                ID, sentence = l.split("\t")
                sentence = sentence[1:-1]
                sentence = sentence.replace('<e1>', 'xxxxxxxxxe1xxxxxxxxx ')
                sentence = sentence.replace('<e2>', 'xxxxxxxxxe2xxxxxxxxx ')
                sentence = sentence.replace('</e1>', ' ssssssssse1sssssssss')
                sentence = sentence.replace('</e2>', ' ssssssssse2sssssssss')
                X_train.append(sentence)
            elif idx % 4 == 1:
                label = l
                if label not in label_to_y:
                    label_to_y[label] = len(label_to_y)
                Y_train.append(label_to_y[label])
            else:
                pass
    y_to_label = {j:i for i, j in label_to_y.items()}
    Y_train = np.array(Y_train, dtype=int)
    num_class = max(Y_train) + 1
    Y_train = to_categorical(Y_train)
    tokenizer = Tokenizer(oov_token="UNK")
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    if VALIDATION_SPLIT > 0:  # generate validation set
        indices = np.arange(len(Y_train))
        np.random.shuffle(indices)
        val_index = int(VALIDATION_SPLIT * len(Y_train))
        X_val = X_train[indices[:val_index]]
        Y_val = Y_train[indices[:val_index]]
        X_train = X_train[indices[val_index:]]
        Y_train = Y_train[indices[val_index:]]
    X_test = []
    ID_test = []
    with open("TEST_FILE.txt") as f:
        for l in f:
            ID, sentence = l.strip().split("\t")
            sentence = sentence[1:-1]
            sentence = sentence.replace('<e1>', 'xxxxxxxxxe1xxxxxxxxx ')
            sentence = sentence.replace('<e2>', 'xxxxxxxxxe2xxxxxxxxx ')
            sentence = sentence.replace('</e1>', ' ssssssssse1sssssssss')
            sentence = sentence.replace('</e2>', ' ssssssssse2sssssssss')
            ID_test.append(ID)
            X_test.append(sentence)
    sequences = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    word_index = tokenizer.word_index  # word dictionary <word, index>
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  # create word embedding matrix
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(num_words,  # inital word embedding weights
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')  # input layer
    embedded_sequences = embedding_layer(sequence_input)  # word embedding
    model = RNN_model(embedded_sequences, num_class)
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.adam(lr= 0.001, amsgrad=True, clipvalue=15),
                metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss' if VALIDATION_SPLIT > 0 else "loss", patience=15, mode='min')
    model.fit(X_train, Y_train,
            batch_size=128,
            epochs=50,
            callbacks=[early_stop],
            validation_data=(X_val, Y_val) if VALIDATION_SPLIT > 0 else None)
    Y_pre = model.predict(X_test)
    Y_pre = np.argmax(Y_pre, axis=1)
    Y_pre = [y_to_label[i] for i in Y_pre]
    with open("predict.txt", 'w') as f:
        for ID, label in zip(ID_test, Y_pre):
            f.write(ID + "\t" + label + "\n")
