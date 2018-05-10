import pandas as pd
import numpy as np
import script as sc
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([240, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                epoch_x = np.array(train_x[start:end])
                epoch_y = np.array(train_y[start:end])
            
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

# data = pd.read_csv(r"C:\Users\yashm\Google Drive\Data Capstone_\Project Folder\PreWideData\updated_wide.csv")
data = pd.read_csv(sc.load_Data('AGG-Yash', 'updated_wide.csv'))

data = data.drop(['Unnamed: 0'], axis = 1)
data['codes'] = data.activity.astype('category').cat.codes
classes = len(data.activity.unique())

x = []
y = []
for i in range(0, len(data), 80):
    if (data['activity'][i] != 'Private / Not Coded'):
        temp = np.insert(np.asarray(data['Accelerometer X'][i:i+80]), 1, data['Accelerometer Y'][i:i+80])
        temp = np.insert(temp, 1, np.asarray(data['Accelerometer Z'][i:i+80]))
        x.append(temp)
    #     x.append(np.asarray([np.asarray(data['Accelerometer X'][i:i+80]), np.asarray(data['Accelerometer Y'][i:i+80]), np.asarray(data['Accelerometer Z'][i:i+80])]))
        y.append(data.codes[i])
x = np.asarray(x) ## formated data shape is n [x, y, z] where n is each second and len(x) = 80

train_x = x[0:int(0.7*len(x))]
train_y = y[0:int(0.7*len(x))]
test_x = x[int(0.7*len(x)):len(x)]
test_y = y[int(0.7*len(y)):len(y)]

print(train_x.shape , len(train_y), test_x.shape, len(test_y))

## set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100

n_classes = classes

## trainX, trainY, testX, testY needed

x = tf.placeholder('float', [None, 240]) ## 80 + 80 + 80
y = tf.placeholder('float')

train_neural_network(x)

