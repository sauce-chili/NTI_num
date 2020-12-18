# tf 2.0.1
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

'''
__init__() - принимает путь до файла .h5
predict() - принимает изображение(типа np.array),в разрешение 120*120;возвращает предсказанное число и точность
imageProcessing() - принимает целое изображение(640*480),типа np.array;возвращет коллекцию 
изображений для обработки нейронной сетью
'''


class Net:
    __classes = {
        0: 1,
        1: 2,
        2: 3,
        3: 0
    }

    __acc = 0  # точность с последнего расспознавания

    def __init__(self, modelPath: str):
        self.model = load_model(modelPath)
        np.set_printoptions(suppress=True)

    def imageProcessing(self, img: np.array) -> list:
        # преобразование изображение в формат 640*120
        h, w, _ = img.shape
        img = img[(h - w):h, 0:w]
        # маска
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 190), (255, 255, 255))
        # поиск контуров цифры внутри кубиков
        cntr = []
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            area = cv2.contourArea(c)
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if 1000 > area > 400:
                cntr.append(list(cv2.boundingRect(approx)))

        # Сортировка контуров слева на право
        for i in range(0, len(cntr)):
            for j in range(0, len(cntr) - i - 1):
                if cntr[j][0] > cntr[j + 1][0]:
                    cntr[j], cntr[j + 1] = cntr[j + 1], cntr[j]

        # формирование из контуров изображений 120*120 для модели нейронки
        imgCollection = []
        for c in cntr:
            picture = img.copy()
            [x, y, w, h] = c
            # координаты середины контура цифры
            xC = x + (w // 2)
            yC = y + (h // 2)
            imgCollection.append(picture[yC - 60:yC + 60, xC - 60:xC + 60])

        return imgCollection

    def predict(self, img: np.array) -> [int, float]:
        x = image.img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        accuriry = self.model.predict(x)
        pred = np.argmax(accuriry, axis=1)
        self.__acc = float(accuriry[0][pred[0]])

        return self.__classes[pred[0]]

    def getAccuriry(self):
        return self.__acc
