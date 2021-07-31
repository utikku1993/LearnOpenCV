import numpy as np
import cv2

# Load video from file
# cap = cv2.VideoCapture('./data/Chantaje.mp4')

# Load video from webcam
cap = cv2.VideoCapture(0)

all_rows = open('./data/googlenet/synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('./data/googlenet/bvlc_googlenet.prototxt', './data/googlenet/bvlc_googlenet.caffemodel')

if not cap.isOpened():
    print('Can not open video stream')

while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))

    net.setInput(blob)

    outp = net.forward()

    idx = np.argsort(outp[0])[::-1][:5]

    r = 1
    for i, id in enumerate(idx):
        txt = '{}. {} ({}): Probability {:.3}%'.format(i+1, classes[id], id, outp[0][id]*100)
        cv2.putText(frame, txt, (0,25 + 40*r), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        r += 1

    if ret:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) and 0xFF == 27:
            break
    else:
        break