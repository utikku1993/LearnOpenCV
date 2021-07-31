import numpy as np
import cv2

img = cv2.imread('./data/1.jpg')

all_rows = open('./data/googlenet/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('./data/googlenet/bvlc_googlenet.prototxt', './data/googlenet/bvlc_googlenet.caffemodel')

blob = cv2.dnn.blobFromImage(img, 1, (224, 224))

net.setInput(blob)

outp = net.forward()

idx = np.argsort(outp[0])[::-1][:5]

for i, id in enumerate(idx):
    print('{}. {} ({}): Probability {:.3}%'.format(i+1, classes[id], id, outp[0][id]*100))