from flask import Flask, request
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

detector = HandDetector(detectionCon=0.8, maxHands=1)
classifier = Classifier("server/Model/SignToText.h5",
                        "server/Model/labels.txt")
offset = 20
imgSize = 300
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z', 'del', 'space']


def predict_image(handimage):
    # Load the image
    img = cv2.imread(f'{handimage}')

    # img = cv2.resize(img, (960, 540))

    # Find the hand and its landmarks
    hands, img = detector.findHands(img)

    imgOutput = img.copy()

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones(
            (imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h +
                      offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(
                imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(
                imgWhite, draw=False)
        # print(labels[index])

        return (labels[index])

    else:
        return "nothing"


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    filename = 'input.jpg'
    img.save(filename)

    result = predict_image(filename)

    # result = 'H'

    return result


app.run(host="0.0.0.0", port=5000)
