# import sys
# sys.path.insert(0, "../../python")
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

#     @property
#     def provide_data(self):
#         return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

#     @property
#     def provide_label(self):
#         return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    buf = ""
    for i in range(4):
        buf += str(random.randint(0,9))
    return buf

def get_label(buf):
    a = [int(x) for x in buf]
    return np.array(a)

def gen_sample(captcha, width, height):
    num = gen_rand()
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    img = np.multiply(img, 1/255.0)
    img = img.transpose(2, 0, 1)
    return (num, img)

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

class OCRIter(object):
    def __init__(self, count, batch_size, num_label, height, width):
        self.captcha = ImageCaptcha(fonts=['./times.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num, img = gen_sample(self.captcha, self.width, self.height)
                data.append(img)
                label.append(get_label(num))

            data_all = np.array(data)
            # print data_all
            label_all = np.array(label)
            label_batch = label_one_hot(label_all)
            # print label_all
            data_names = ['data']
            label_names = ['softmax_label']
            
            # data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_all, label_batch

    def reset(self):
        pass

if __name__ == '__main__':
    batch_size = 8
    data_train = OCRIter(8, batch_size, 4, 30, 80)
    # data_test = OCRIter(8, batch_size, 4, 30, 80)
    i = 0
    j = 0
    for (x, y) in data_train:
        print np.shape(x), type(x)
        print np.shape(y), type(y)
    # for image in data_test:
    #     j = j + 1
    # print j