#encoding:utf-8
# @author April Wang
# @date 2019-02-15

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pdb
import sys
import json
import numpy
import time
import pickle
import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['KERAS_BACKEND'] = 'tensorflow'
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Input, Lambda,LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard, ProgbarLogger
import keras
from layers.attention2 import Attention
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils.generic_utils import CustomObjectScope
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

TRAIN_FILE_NAME='./data/train_data_demo'
TEST_FILE_NAME='./data/test_data_demo'
BASELINE_FILE_NAME='./data/test_data_demo' # here should use another validation sample set

OUTPUT_PREFIX = './result/demo'
EPOCH_NUM = 3
MAXLEN = 400 
BATCH_SIZE = 256
DUROS_LENGTH = 2
PAGE_ID_LENGTH=29
POINT_ID_LENGTH = 186
ORDER_TYPE_LENGTH = 4
def file_length(filename):
    file_length = 0
    for line in open(filename,'rU'):
        file_length += 1
    return file_length
TRAIN_FILE_LENGTH = file_length(TRAIN_FILE_NAME)
BATCH_NUM = TRAIN_FILE_LENGTH/BATCH_SIZE
LAST_BATCH_SIZE =  BATCH_SIZE + (TRAIN_FILE_LENGTH - BATCH_NUM*BATCH_SIZE)
INPUTS_NAME = ['input_sequence_dur_os', 'input_sequence_page', 'input_sequence_point', 'input_normal']
OUTPUTS_NAME = ['output']

def get_one_hot_vector(idx, length):
    x = [0]*length
    x[int(idx)] = 1
    return x

def sequence_preprocess(sequences_ori):
    sample_sequence = []
    for sequence_ubt in sequences_ori:
        inner_element = []
        event_diff = float(sequence_ubt[0])
        duration = float(sequence_ubt[1])+1
        os_type = float(sequence_ubt[2])+1
        page_sequence_cur = float(sequence_ubt[3])+1
        point_sequence_cur = float(sequence_ubt[4])+1
        inner_element = [event_diff, duration, os_type,  page_sequence_cur,  point_sequence_cur]
        sample_sequence.append(inner_element)
    sample_sequence_np = numpy.array(sample_sequence).T
    if len(sequences_ori) == 0:
        sample_sequence_np =  numpy.array([[0.0]*5]).T
    sequence_pad =  sequence.pad_sequences(sample_sequence_np, maxlen= MAXLEN, dtype='float32', value=0.0)
    dur_os_sequence = sequence_pad[:3,:].T
    for i in dur_os_sequence:
        if i[0] == 0:
            i[0] = 2592000.0
        else:
            break
    #ignore dur_os_sequence[:,0]
    dur_os_sequence = dur_os_sequence[:,1:]
    page_sequence  =  sequence_pad[3,:]
    point_sequence  =  sequence_pad[4,:]
    #pdb.set_trace()
    return dur_os_sequence, page_sequence, point_sequence

def line_preprocess(line, input_output_dict):
    line = line.strip().split('\t')
    sample = line[0]
    sequence_len = line[1]
    date =  line[2]
    label = int(line[3]) #1 overduedays>0, 0 else
    sample_type_id = line[4]
    sample_type_vector =  get_one_hot_vector(int(sample_type_id), ORDER_TYPE_LENGTH)
    sequences_ori = json.loads(line[5])
    dur_os_sequence, page_sequence, point_sequence = sequence_preprocess(sequences_ori)
    input_output_dict['input_sequence_dur_os'].append(dur_os_sequence)
    input_output_dict['input_sequence_page'].append(page_sequence)
    input_output_dict['input_sequence_point'].append(point_sequence)
    input_output_dict['input_normal'].append(sample_type_vector)
    input_output_dict['output'].append(label)
    return input_output_dict

def trans_np(input_output_dict):
    for k in input_output_dict:
        input_output_dict[k] = numpy.array(input_output_dict[k]).astype('float32')
    return ({k:input_output_dict[k] for k in INPUTS_NAME}, {k:input_output_dict[k] for k in OUTPUTS_NAME})

def initial_input_output():
    input_output_dict = {k:[] for k in INPUTS_NAME}
    for i in OUTPUTS_NAME:
        input_output_dict[i] = []
    return input_output_dict

def generate_validation_set(path):
    input_output_dict = initial_input_output()
    fin = open(path,'rb')
    for line in fin:
        input_output_dict = line_preprocess(line, input_output_dict)
    sample_tuple = trans_np(input_output_dict)
    fin.close()
    return sample_tuple

def generate_arrays_from_file(path, batch_size):
    input_output_dict = initial_input_output()
    batch_len = 0
    n_cur_batches = 1
    while True:
        fin = open(path,'rb')
        for line in fin:
            input_output_dict = line_preprocess(line, input_output_dict)
            batch_len += 1
            if n_cur_batches % BATCH_NUM == 0:
                batch_size_requirement = LAST_BATCH_SIZE
            else:
                batch_size_requirement = batch_size

            if batch_len == batch_size_requirement:
                n_cur_batches += 1
                sample_tuple = trans_np(input_output_dict)
                yield sample_tuple 
                batch_len = 0
                input_output_dict = initial_input_output()
        fin.close()

print('Loading testset ...')
t0 = time.time()
x_test, y_test = generate_validation_set(TEST_FILE_NAME)
t1 = time.time()
for name in INPUTS_NAME:
    print 'x_test %s shape: %s' %(name, x_test[name].shape)
for name in OUTPUTS_NAME:
    print 'y_test %s shape: %s' %(name, y_test[name].shape)
print 'Loading testset using %s s' %(int(t1 - t0))

print('Loading baselineset ...')
t0 = time.time()
x_baseline, y_baseline = generate_validation_set(BASELINE_FILE_NAME)
t1 = time.time()
for name in INPUTS_NAME:
    print 'x_baseline %s shape: %s' %(name, x_baseline[name].shape)
for name in OUTPUTS_NAME:
    print 'y_baseline %s shape: %s' %(name, y_baseline[name].shape)

#generator test
print('Generator testing  ...')
generator = generate_arrays_from_file(TRAIN_FILE_NAME, BATCH_SIZE)
x_train_demo, y_train_demo = generator.next()
for name in INPUTS_NAME:
    print 'x_train_demo %s shape: %s' %(name, x_train_demo[name].shape)
for name in OUTPUTS_NAME:
    print 'y_train_demo %s shape: %s' %(name, y_train_demo[name].shape)

def get_lstm_sequence_last_result(x):
    x_output = x[:,-1,:]
    return x_output

def lstm_weighted_attention(inputs):
    lstm_out, attention = inputs
    a = K.expand_dims(attention, axis=-1)
    h_after_a = lstm_out * a
    h_after_a = K.sum(h_after_a, axis=1)
    return h_after_a

print('Build model...')
input_sequence_dur_os = Input(shape=(MAXLEN, DUROS_LENGTH,), dtype='float32', name='input_sequence_dur_os')
input_sequence_page = Input(shape=(MAXLEN, ), dtype='float32', name='input_sequence_page')
input_sequence_point= Input(shape=(MAXLEN, ), dtype='float32', name='input_sequence_point')
page_embedding = Embedding(output_dim=16, input_dim = PAGE_ID_LENGTH+1, input_length=MAXLEN, name='page_embedding')(input_sequence_page)
point_embedding = Embedding(output_dim=128, input_dim = POINT_ID_LENGTH+1, input_length=MAXLEN, name='point_embedding')(input_sequence_point)
input_sequence = keras.layers.concatenate([input_sequence_dur_os, page_embedding, point_embedding]) #2+16+128
#lstm_out = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), merge_mode='sum')(input_sequence)
lstm_out = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(input_sequence)
attention = Attention(name='attention')(lstm_out)
lstm_after_attention = Lambda(lstm_weighted_attention, output_shape=(128,),name='lstm_weighted_a')([lstm_out, attention])
x = Dense(128, activation='relu')(lstm_after_attention)
main_output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[input_sequence_dur_os, input_sequence_page, input_sequence_point], outputs=[main_output])
print model.summary()
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[auc])

print('Train...')

class RocAucMetricCallback(keras.callbacks.Callback):
    #output the auc for y and model.predict(x)
    def __init__(self, x,y,name):
        super(RocAucMetricCallback, self).__init__()
        self.x = x
        self.y = y
        self.name=name
    def on_batch_begin(self, batch, logs={}):
        pass
 
    def on_batch_end(self, batch, logs={}):
        pass
    def on_train_begin(self, logs={}):
        if not (self.name+'_auc' in self.params['metrics']):
            self.params['metrics'].append(self.name+'_auc')
 
    def on_train_end(self, logs={}):
        pass
 
    def on_epoch_begin(self, epoch, logs={}):
        pass
 
    def on_epoch_end(self, epoch, logs={}):
        logs[self.name+'_auc']=float('-inf')
        y_pred = self.model.predict(self.x)[:,0]
        y_true = self.y['output']
        logs[self.name+'_auc']=roc_auc_score( y_true, y_pred)
 

checkpointer = ModelCheckpoint(filepath=OUTPUT_PREFIX+'_best_weights.h5',
                            monitor='val_auc',
                            verbose=1,
                            save_best_only='True',
                            mode='max',
                            period=1,
                            )
#tensorboard = TensorBoard(log_dir='./logs',histogram_freq=1, 
#                          #update_freq='batch',
#                        )
#auccb = RocAucMetricCallback(x_test, y_test, 'valcb')

print('Train...')
history = model.fit_generator(generate_arrays_from_file(TRAIN_FILE_NAME, BATCH_SIZE),
                    epochs=EPOCH_NUM,
                    steps_per_epoch = BATCH_NUM,
                    verbose = 1,
                    validation_data=(x_baseline, y_baseline),
                    #callbacks=[auccb,checkpointer, ProgbarLogger(count_mode='steps'),tensorboard],
                    callbacks=[checkpointer],
                    )

with CustomObjectScope({'auc':auc,'Attention':Attention}):
    model = load_model(OUTPUT_PREFIX+'_best_weights.h5')

print('Predict test...')
pred_test = model.predict(x_test,verbose=1)[:,0]
print 'pred_test shape: %s' %  str(pred_test.shape)
true_test = y_test['output']
print 'TEST_AUC: ', roc_auc_score(true_test, pred_test)

print('Predict baseline...')
pred_baseline = model.predict(x_baseline,verbose=1)[:,0]
print 'pred_baseline shape: %s' %  str(pred_baseline.shape)
true_baseline = y_baseline['output']
print 'BASELINE_AUC: ', roc_auc_score(true_baseline, pred_baseline)

print('Predict train...')
pred_train = model.predict_generator(generate_arrays_from_file(TRAIN_FILE_NAME, BATCH_SIZE),steps = BATCH_NUM,verbose=1)[:,0]
print 'pred_train shape: %s' % str(pred_train.shape)
true_train = numpy.array([])
generator = generate_arrays_from_file(TRAIN_FILE_NAME, BATCH_SIZE)
for i in range(BATCH_NUM):
    _, y_train_demo = generator.next()
    true_train  = numpy.concatenate((true_train, y_train_demo['output']))
print 'TRAIN_AUC: ', roc_auc_score(true_train, pred_train)

def print_third_file(y_true, y_pred, filename):
    assert y_true.shape == y_pred.shape
    fout = open(filename, 'wb')
    for i in range(len(y_true)):
        print >> fout, '%s\t%s\t%s' %(int(y_true[i]), '9990000001', y_pred[i])

print('Write train test baseline result...')
print_third_file(true_train, pred_train, OUTPUT_PREFIX+'_train_true_pred_result.txt')
print_third_file(true_test, pred_test, OUTPUT_PREFIX+'_test_true_pred_result.txt')
print_third_file(true_baseline, pred_baseline, OUTPUT_PREFIX+'_baseline_true_pred_result.txt')

#history_dict = history.history
#pickle.dump(history_dict, open(OUTPUT_PREFIX+'_history', 'wb'))
