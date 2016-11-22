from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import os
import nltk
import random

# Setting the initial values
batch_size = 100
start = 0
end = batch_size
start2 = 0
end2 = batch_size
start3 = 0
end3 = batch_size
num_classes = 8
# Paths to get files into memory
path1 = os.path.join(os.getcwd(),"aclImdb/train/pos")
path2 = os.path.join(os.getcwd(),"aclImdb/train/neg")
path3 = os.path.join(os.getcwd(),"aclImdb/test/pos")
path4 = os.path.join(os.getcwd(),"aclImdb/test/neg")
out_dir = os.getcwd()
embeddings = {}
time_steps = 300
embedding = 50
step = 1
_units = 384
new_cost = 0
new_accu = 0

# Function to get glove embeddings
def get_embedding():
    gfile_path = os.path.join(os.getcwd(),"glove.6B.50d.txt")
    f = open(gfile_path,'r')
    for line in f:
        sp_value = line.split()
        word = sp_value[0]
        embedding = [float(value) for value in sp_value[1:]]
        assert len(embedding) == 50
        embeddings[word] = embedding
    return embeddings

ebd = get_embedding()
# Function to get output data
def get_y(file_path):
    y_value = file_path.split('_')
    y_value = y_value[1].split('.')
    if y_value[0] == '1':
       return 0
    elif y_value[0] == '2':
         return 1
    elif y_value[0] == '3':
          return 2
    elif y_value[0] == '4':
          return 3
    elif y_value[0] == '7':
          return 4
    elif y_value[0] == '8':
          return 5
    elif y_value[0] == '9':
          return 6
    elif y_value[0] == '10':
          return 7 
# Function to get input data  
def get_x(file_path):
    x_value = open(file_path,'r')
    for line in x_value:
        x_value = line.replace("<br /><br />","") 
        x_value = x_value.lower()
    x_value = nltk.word_tokenize(x_value.decode('utf-8'))
    padding = 300 - len(x_value)
    if padding > 0:
       p_value = ['pad' for i in range(padding)]
       x_value = np.concatenate((x_value,p_value))
    if padding < 0:
       x_value = x_value[:300]
    for i in x_value:
        if ebd.get(i) == None:
           ebd[i] = [float(np.random.normal(0.0,1.0)) for j in range(50)]
    x_value = [ebd[value] for value in x_value]
    assert len(x_value) == 300
    return x_value

# Function to get complete dataset and then sliced to make train,validate and test sets.    
def get_total_files(path1,path2,path3,path4):
    directory1 = os.listdir(path1)
    file_path1 = [os.path.join(path1,file) for file in directory1]
    directory2 = os.listdir(path2)
    file_path2 = [os.path.join(path2,file) for file in directory2]
    directory3 = os.listdir(path3)
    file_path3 = [os.path.join(path3,file) for file in directory3]
    directory4 = os.listdir(path4)
    file_path4 = [os.path.join(path4,file) for file in directory4]
    total_files_train = np.concatenate((file_path1,file_path2))
    total_files_test = np.concatenate((file_path3,file_path4))
    random.shuffle(total_files_train)
    random.shuffle(total_files_test)    
    x1 = [get_x(file) for file in total_files_train]
    y1 = [get_y(file) for file in total_files_train]
    x2 = [get_x(file) for file in total_files_test]
    y2 = [get_y(file) for file in total_files_test]
    return x1 , y1 , x2 , y2

total_files_train_x, total_files_train_y, total_files_test_x, total_files_test_y = get_total_files(path1,path2,path3,path4)

# Train,Validate and Test datasets
train_set_x = total_files_train_x[:1000]
validate_set_x = total_files_train_x[1000:1100]
test_set_x = total_files_test_x[0:100]
train_set_y = total_files_train_y[:1000]
validate_set_y = total_files_train_y[1000:1100]
test_set_y = total_files_test_y[0:100]
    
# Tensorflow placeholders for input,output,dropout and normalization
X = tf.placeholder(tf.float32, [None,time_steps,embedding])
Y = tf.placeholder(tf.int32, [None])
A = tf.placeholder(tf.bool)
B = tf.placeholder(tf.float32)

x = tf.expand_dims(X,3)
# Convolution layer       
filter_shape = [1, embedding, 1, 1000]
conv_weights = tf.get_variable("conv_weights1" , filter_shape, tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
conv_biases = tf.Variable(tf.constant(0.1, shape=[1000]))
conv = tf.nn.conv2d(x, conv_weights, strides=[1,1,1,1], padding = "VALID")
normalize = tf.nn.elu(conv + conv_biases)
tf_normalize = tf.contrib.layers.batch_norm(inputs = normalize,is_training = A)
outputs_fed_lstm = tf_normalize

x = tf.squeeze(outputs_fed_lstm, [2])     
x = tf.transpose(x, [1, 0, 2])
x = tf.reshape(x, [-1, 1000])
x = tf.split(0, time_steps, x)
#LSTM
lstm = tf.nn.rnn_cell.LSTMCell(num_units = _units, state_is_tuple=True)
# Commented code shows the other methods that I have tried for this network.      
# multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * lstm_layers, state_is_tuple = True)
     
outputs , state = tf.nn.rnn(lstm,x, dtype = tf.float32)     

#convolution = tf.expand_dims(outputs[-1],2)
#convolution = tf.expand_dims(convolution,3)

#filter_shape4 = [64,1,1,8]
#conv_weights4 = tf.get_variable("conv_weights4" , filter_shape4, tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
#conv_biases4 = tf.Variable(tf.constant(0.1, shape=[8]))
#conv4 = tf.nn.conv2d(convolution, conv_weights4, strides=[1,1,1,1], padding = "VALID")
#logits = tf.nn.elu(conv4 + conv_biases4)
#logits = tf.squeeze(logits)

weights = tf.Variable(tf.random_normal([_units,num_classes]))
biases  = tf.Variable(tf.random_normal([num_classes]))

logits = tf.matmul(outputs[-1], weights) + biases
    
c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,Y)
loss = tf.reduce_mean(c_loss)
    

global_step = tf.Variable(0, name="global_step", trainable=False)
decayed_learning_rate = tf.train.exponential_decay(learning_rate = 0.0001,global_step = global_step,decay_steps = 30, decay_rate = 0.9, staircase = True)
optimizer= tf.train.AdamOptimizer(learning_rate = decayed_learning_rate)
# Code to check whether gradients are flowing properly.
#grads = optimizer.compute_gradients(loss,[conv_weights3])
minimize_loss = optimizer.minimize(loss, global_step=global_step)   
 
correct_predict = tf.nn.in_top_k(logits, Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    
init = tf.initialize_all_variables()

with tf.Session() as sess:
     sess.run(init)
     for i in range(1000):
         for j in range(10):
             training = True # Batch Normalization when training
             probability = 1 # Dropout when training
             x = train_set_x[start:end]
             y = train_set_y[start:end]
             start = end
             end += batch_size
             if start >= 1000:
                start = 0
                end = batch_size  
             sess.run(minimize_loss,feed_dict={X : x, Y : y, A : training, B : probability})  # Loss is minimized
             training = False # Batch Normalization when testing and validating
             probability = 1  # Dropout when testing and validating.
             cost = sess.run(loss,feed_dict = {X: x,Y: y, A : training, B : probability})
             new_cost += cost
             accu = sess.run(accuracy,feed_dict = {X: x, Y: y, A : training, B : probability})
             new_accu += accu
         real_cost = (new_cost/10)
         real_accu = (new_accu/10)
         new_cost = 0
         new_accu = 0
         print (step)
         step += 1
         print ("Epoch Finished")
         print ("Loss after one Epoch(Training) = " + "{:.3f}".format(real_cost) + ", Training Accuracy= " + "{:.3f}".format(real_accu))
         # Summary for accuracy, In Tensorboard
         accuracy_summary = tf.scalar_summary("accuracy", real_accu)
         train_summary_dir = os.path.join(out_dir, "summaries", "train")
         train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
         summary = sess.run(accuracy_summary)
         train_summary_writer.add_summary(summary, step)
         q = validate_set_x[start2:end2]
         w = validate_set_y[start2:end2]
         if start2 >= 100:
            start2 = 0
            end2 = batch_size
         cost = sess.run(loss,feed_dict = {X: q,Y: w, A : training, B : probability})
         accu = sess.run(accuracy,feed_dict = {X: q, Y: w, A : training, B : probability})
         print ("Loss after one Epoch(Validation) = " + "{:.3f}".format(cost) + ", Validation Accuracy= " + "{:.3f}".format(accu))
         # Summary, Tensorboard code
         accuracy_summary = tf.scalar_summary("accuracy", accu)
         dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
         dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
         summary = sess.run(accuracy_summary)
         dev_summary_writer.add_summary(summary, step)
         e = test_set_x[start3:end3]
         r = test_set_y[start3:end3]
         if start3 >= 100:
            start3 = 0
            end3 = batch_size
         cost = sess.run(loss,feed_dict = {X: e,Y: r, A : training, B : probability})
         accu = sess.run(accuracy,feed_dict = {X: e, Y: r, A : training, B : probability})
         print ("Loss after one Epoch(Test) = " + "{:.3f}".format(cost) + ", Test Accuracy= " + "{:.3f}".format(accu))
         # Summary, Tensorboard code
         accuracy_summary = tf.scalar_summary("accuracy", accu)
         test_summary_dir = os.path.join(out_dir, "summaries", "test")
         test_summary_writer = tf.train.SummaryWriter(test_summary_dir, sess.graph)
         summary = sess.run(accuracy_summary)
         test_summary_writer.add_summary(summary, step)
         
         
            


