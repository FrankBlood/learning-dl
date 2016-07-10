# -*-coding:utf8-*-
import tensorflow as tf
# import sys
import numpy as np
import cv2, random
from io import BytesIO
from captcha.image import ImageCaptcha

# class OCRBatch(object):
#     def __init__(self, data_names, data, label_names, label):
#         self.data = data
#         self.label = label
#         self.data_names = data_names
#         self.label_names = label_names

    # @property
    # def provide_data(self):
    #     return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    # @property
    # def provide_label(self):
    #     return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    buf = ""
    for i in range(4):
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
    img = img.transpose(2, 0, 1)
    return (num, img)

class OCRIter(object):
    def __init__(self, count, batch_size, num_label, height, width):
        self.captcha = ImageCaptcha(fonts=['./times.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        # self.provide_data = [('data', (batch_size, 3, height, width))]
        # self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height)
                img = np.transpose(img)
                data.append(img)
                label.append(get_label(num))

            data_all = np.array(data)
            label_all = np.array(label)
            label_batch = label_one_hot(label_all)
            label_1 = label_batch[:, :10]
            label_2 = label_batch[:, 10:20]
            label_3 = label_batch[:, 20:30]
            label_4 = label_batch[:, 30:40]
            data_names = ['data']
            label_names = ['softmax_label']
            
            # data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_all, label_1, label_2, label_3, label_4

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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

# def Accuracy(label, pred):
#     label = label.T.reshape((-1, ))
#     hit = 0
#     total = 0
#     for i in range(pred.shape[0] / 4):
#         ok = True
#         for j in range(4):
#             k = i * 4 + j
#             if np.argmax(pred[k]) != int(label[k]):
#                 ok = False
#                 break
#         if ok:
#             hit += 1
#         total += 1
#     return 1.0 * hit / total

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 80 ,40, 3])
y_1 = tf.placeholder("float", [None, 10])
y_2 = tf.placeholder("float", [None, 10])
y_3 = tf.placeholder("float", [None, 10])
y_4 = tf.placeholder("float", [None, 10])
y_ = tf.placeholder("float", [None, 40])
y_ = tf.concat(1, [y_1, y_2, y_3, y_4])
print '申请两个占位符'
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = conv2d(x, W_conv1) + b_conv1
h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))
print '第一层卷积'
W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2	
h_pool2 = tf.nn.relu(max_pool_2x2(h_conv2))
print '第二层卷积'
W_conv3 = weight_variable([5, 5, 32, 32])
b_conv3 = bias_variable([32])
h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3	
h_pool3 = tf.nn.relu(max_pool_2x2(h_conv3))
print '第三层卷积'
W_fc1 = weight_variable([5*10*32, 256])
b_fc1 = bias_variable([256])
h_pool3_flat = tf.reshape(h_pool3, [-1, 5*10*32])
h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
print '全连接'
W_fc21 = weight_variable([256, 10])
b_fc21 = bias_variable([10])
h_fc21 = tf.nn.softmax(tf.matmul(h_fc1, W_fc21) + b_fc21)
print '第一个数字'
W_fc22 = weight_variable([256, 10])
b_fc22 = bias_variable([10])
h_fc22 = tf.nn.softmax(tf.matmul(h_fc1, W_fc22) + b_fc22)
print '第二个数字'
W_fc23 = weight_variable([256, 10])
b_fc23 = bias_variable([10])
h_fc23 = tf.nn.softmax(tf.matmul(h_fc1, W_fc23) + b_fc23)
print '第三个数字'
W_fc24 = weight_variable([256, 10])
b_fc24 = bias_variable([10])
h_fc24 = tf.nn.softmax(tf.matmul(h_fc1, W_fc24) + b_fc24)
print '第四个数字'
fc2 = tf.concat(1, [h_fc21, h_fc22, h_fc23, h_fc24])
print np.shape(h_fc21), np.shape(h_fc22), np.shape(h_fc23), np.shape(h_fc24)
print '组合四个数字'


cross_entropy = -tf.reduce_sum(y_*tf.log(fc2))
# cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(fc2, y_)
print '计算交叉熵'
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
print '梯度下降求解'

_1 = tf.argmax(h_fc21, 1)
_2 = tf.argmax(h_fc22, 1)
_3 = tf.argmax(h_fc23, 1)
_4 = tf.argmax(h_fc24, 1)

correct_prediction = tf.equal(_1, tf.argmax(y_1, 1)) and tf.equal(_2, tf.argmax(y_2, 1)) and tf.equal(_3, tf.argmax(y_3, 1)) and tf.equal(_4, tf.argmax(y_4, 1))
print '计算正确个数'


accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
print '运行sess'

batch_size = 50

# data_test, label_test = OCRIter(1000, batch_size, 4, 40, 80)
print '生成数据'
i = 0
for (batch_data, label_1, label_2, label_3, label_4) in OCRIter(100000, batch_size, 4, 40, 80):
    print '训练模型'
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_data, y_1:label_1, y_2:label_2, y_3:label_3, y_4:label_4})
        print (i, train_accuracy)
    train_step.run(feed_dict={x: batch_data, y_: batch_label})
    i = i + 1

# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))