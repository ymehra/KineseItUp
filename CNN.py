import pandas as pd
import numpy as np
import script as sc
import tensorflow as tf

# data = pd.read_csv(r"C:\Users\yashm\Google Drive\Data Capstone_\Project Folder\PreWideData\cnn_wide.csv")
data = pd.read_csv(sc.load_Data("AGG-Yash", "cnn_wide.csv"))
features = data[data.columns[0:240]]

labels = data['activity']
K = len(data['activity']. unique())


graph = tf.Graph()
with graph.as_default():
    train_features = tf.placeholder(tf.float32, shape=(None, 240))
    x = tf.reshape(train_features, [-1, 80, 3])
    
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(x, feed_dict={train_features: features}))

num_epochs = 100
learning_rate = 1e-2

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
                             activation=tf.nn.relu)
    # Max Pooling 1 (reduces samples from 80 --> 40)
    pool1 = tf.layers.max_pooling1d(inputs=conv1,
                                    pool_size=2, 
                                    strides=2,
                                    data_format="channels_last")
    
    # Convolution 2
    conv2 = tf.layers.conv1d(inputs=pool1, 
                             filters=16, 
                             kernel_size=8, 
                             padding="same", 
                             data_format="channels_last",
                             activation=tf.nn.relu)
    # Max Pooling 2 (reduces samples from 40 --> 20)
    pool2 = tf.layers.max_pooling1d(inputs=conv2,
                                    pool_size=2, 
                                    strides=2,
                                    data_format="channels_last")
    
    # Flatten the Pooled Data
    pool2_flat = tf.reshape(pool2, [-1, 20 * 16])
    
    # Dense Layer (try adding dropout?)
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    
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
    for _ in range(num_epochs):
        feed_dict = {
            train_features: features,
            train_labels: labels
        }
        _, _, acc = sess.run(
            [optimizer, loss, accuracy], 
            feed_dict=feed_dict)
        print(acc)
