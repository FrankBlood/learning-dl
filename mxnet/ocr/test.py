# from io import BytesIO
# from captcha.image import ImageCaptcha

# image = ImageCaptcha(fonts=['times.ttf'])

# data = image.generate('1234')
# image.write('1234', 'out.png')
# label_all = [1, 2, 3, 4]
# one = []
# for label in label_all:
# 	one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 	one_hot[label] = 1
# 	print one_hot
# 	one += one_hot
# print one
import numpy as np
label_all = np.array([[1],
					  [2],
					  [3]])
label_batch = []
for label in label_all:
    one = []
    for i in label:
        one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        one_hot[i] = 1
        one += one_hot
    label_batch.append(one)
label_batch = np.array(label_batch)
print np.shape(label_batch)
print type(label_batch)
print label_batch



