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
    img = img.transpose(2, 0, 1)
    return (num, img)

class OCRIter(object):
    def __init__(self, count, batch_size, height, width):
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
            data_names = ['data']
            label_names = ['softmax_label']
            # data_batch = OCRBatch(data_names, data_all, label_names, label_all)
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
y_ = tf.placeholder("float", [None, 10])
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

cross_entropy = -tf.reduce_sum(y_*tf.log(h_fc21))
print '计算交叉熵'
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
print '梯度下降求解'
correct_prediction = tf.equal(tf.argmax(h_fc21,1), tf.argmax(y_,1))
print '计算正确个数'
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print '计算正确率'
sess.run(tf.initialize_all_variables())
print '运行sess'

batch_size = 50
print '生成数据'
i = 0
for (batch_data, label_batch) in OCRIter(100000, batch_size, 40, 80):
    print '训练模型'
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_data, y_:label_batch})
        print (i, train_accuracy)
    train_step.run(feed_dict={x: batch_data, y_: batch_label})
    i = i + 1
test_data, test_label = OCRIter(1000, 1000, 40, 80)
print(accuracy.eval(feed_dict={x: test_data, y_: test_label}))