import pandas as pd
import numpy as np
import script as sc
import tensorflow as tf

def run_CNN(features, labels, K, epochs, learning_rate, window1, stride1, window2, stride2):
   graph = tf.Graph()
   with graph.as_default():
      train_features = tf.placeholder(tf.float32, shape=(None, 240))
      x = tf.reshape(train_features, [-1, 80, 3])
      
   # with tf.Session(graph=graph) as sess:
   #    tf.global_variables_initializer().run()
   #    print(sess.run(x, feed_dict={train_features: features}))

   num_epochs = epochs

   best_epoch = 0
   best_acc = 0

   graph = tf.Graph()
   with graph.as_default():
      # Placeholders for the actual data.
      train_features = tf.placeholder(tf.float32, shape=(None, 240))
      train_labels = tf.placeholder(tf.int32, shape=(None, ))
      
      # Reshaping the data
      x = tf.reshape(train_features, [-1, 80, 3])
      y = tf.one_hot(
         train_labels,
         depth=K
      )
      
      # Convolution 1
      conv1 = tf.layers.conv1d(inputs=x, 
                              filters=8, 
                              kernel_size=8, 
                              padding="same", 
                              data_format="channels_last",
                              activation=tf.nn.tanh)
      # Max Pooling 1 (reduces samples from 80 --> 31)
      pool1 = tf.layers.max_pooling1d(inputs=conv1,
                                       pool_size=window1, 
                                       strides=stride1,
                                       data_format="channels_last")
      
      # Convolution 2
      conv2 = tf.layers.conv1d(inputs=pool1, 
                              filters=16, 
                              kernel_size=8, 
                              padding="same", 
                              data_format="channels_last",
                              activation=tf.nn.tanh)
      # Max Pooling 2 (reduces samples from 39 --> 17)
      pool2 = tf.layers.max_pooling1d(inputs=conv2,
                                       pool_size=window2, 
                                       strides=stride2,
                                       data_format="channels_last")
      
      # Flatten the Pooled Data
      outshape = int(((((80 - window1)/stride1) + 1 - window2)/stride2) + 1)
      pool2_flat = tf.reshape(pool2, [-1, outshape * 16])
      
      # Dense Layer (try adding dropout?)
      dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.tanh)
      
      # Logits Layer
      logits = tf.layers.dense(inputs=dense, units=K)
      
      # Define loss function and optimizer
      loss = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(
               logits=logits, 
               labels=y
         )
      )
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
      
      # Define accuracy (so that it can be calculated and printed)
      accuracy = tf.reduce_mean(
         tf.cast(
               tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)),
               tf.float32
         )
      )
      
   with tf.Session(graph=graph) as sess:
      tf.global_variables_initializer().run()
      i = 0
      prev = 0
      for _ in range(num_epochs):
         feed_dict = {
               train_features: features,
               train_labels: labels
         }
         _, _, acc = sess.run(
               [optimizer, loss, accuracy], 
               feed_dict=feed_dict)
         print("Epoch: ", i, " Acc: ", acc*100, " Change: ", (acc-prev)*100)
         if(acc > best_acc):
            best_acc = prev
            best_epoch = i
         i = i + 1
         prev = acc
   
   return best_epoch, best_acc

def main():
   # data = pd.read_csv("cnn_wide.csv")
   data = pd.read_csv(r"C:\Users\yashm\Google Drive\Data Capstone_\Project Folder\PreWideData\cnn_wide.csv")
   # data = pd.read_csv(sc.load_Data("AGG-Yash", "cnn_wide.csv"))

   labels = data['activity']
   K = len(data['activity'].unique())
   features = data[data.columns[1:-2]]

   print(features.shape, len(labels), K)

   epochs = 200
   learning_rate = 0.05
   window1 = 40
   stride1 = 2
   window2 = 3
   stride2 = 2

   with open("tanh.csv","a") as f:
      for i in range(5, 15):
         learning_rate = i/100
         print(learning_rate)
         best_epoch, best_acc = run_CNN(features, labels, K, epochs, learning_rate, window1, stride1, window2, stride2)
         out_string = str(learning_rate) + "," + str(best_epoch) + "," + str(best_acc) + "\n"
         f.write(out_string)
main()