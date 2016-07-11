# -*-coding:utf8-*-
import tensorflow as tf
import numpy as np
import cv2, random
from io import BytesIO
from captcha.image import ImageCaptcha

def gen_rand():
    buf = ""
    for i in range(1):
        buf += str(random.randint(0,9))
    return buf

def get_label(buf):
    a = [int(x) for x in buf]
    return np.array(a)

def label_one_hot(label_all):
    label_batch = []
    for label in label_all:
        one = []
        for i in label:
            one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            one_hot[i] = 1
            one += one_hot
        label_batch.append(one)
    label_batch = np.array(label_batch)
    return label_batch

def gen_sample(captcha, width, height):
    num = gen_rand()
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    #img = img.transpose(2, 0, 1)
    return (num, img)

class OCRIter(object):
    def __init__(self, count, batch_size, height, width):
        self.captcha = ImageCaptcha(fonts=['./times.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height)
                #img = np.transpose(img)
                data.append(img)
                label.append(get_label(num))

            data_all = np.array(data)
            label_all = np.array(label)
            label_batch = label_one_hot(label_all)
            data_all = data_all.astype("float32")
            label_batch = label_batch.astype("float64")
            data_names = ['data']
            label_names = ['softmax_label']
            # print type(data_all), np.shape(data_all)
            # print type(label_batch), np.shape(label_batch)
            # print type(data_all[0, 0, 0, 0])
            # print type(label_batch[0, 0])
            yield data_all, label_batch
            
    def reset(self):
        pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 28 ,28, 3])
y_ = tf.placeholder("float", [None, 10])
print '申请两个占位符'

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print '第一层卷积'

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print '第二层卷积'

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print 'flatten'

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print 'dropout'

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print 'softmax'

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
print 'cross_entropy'
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
print 'train_step'
print tf.shape(y_conv)
print tf.shape(y_)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
print 'correct_prediction'
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print 'accuracy'
sess.run(tf.initialize_all_variables())
print 'run'

batch_size = 50
print '生成数据'
i = 0
for (batch_data, label_batch) in OCRIter(10000000, batch_size, 28, 28):
    #print '训练模型'
    if i%1000 == 0:
        # print np.shape(batch_data), type(batch_data), type(batch_data[0, 0, 0, 0])
        # print np.shape(label_batch), type(label_batch), type(label_batch[0, 0])
        train_accuracy = accuracy.eval(feed_dict={x:batch_data, y_:label_batch, keep_prob: 1.0})
        print (i, train_accuracy)
    train_step.run(feed_dict={x: batch_data, y_: label_batch, keep_prob: 0.5})
    i = i + 1

for (test_data, test_label) in OCRIter(10000, 10000, 28, 28):
    print '测试模型'
    print(accuracy.eval(feed_dict={x: test_data, y_: test_label, keep_prob: 1.0}))