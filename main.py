import os
import re
import cv2
from easyocr import Reader
import helper.CharacterDetection as CharacterDetection
import helper.PlateDetection as PlateDetection

def checkPlate(filepath):
    blnKNNTrainingSuccessful = CharacterDetection.loadKNNDataAndTrainKNN()
    if not blnKNNTrainingSuccessful:
        print('Failed Load Data!')

    imgOriginalS: None = cv2.imread(filepath)

    scale_percent = 80
    width = int(imgOriginalS.shape[1] * scale_percent / 100)
    height = int(imgOriginalS.shape[0] * scale_percent / 100)
    dimensi = (width, height)
    imgOriginalScene = cv2.resize(imgOriginalS, dimensi, interpolation=cv2.INTER_AREA)

    if imgOriginalScene is None:
        print('Image Not Found!')

    listOfPossiblePlates = PlateDetection.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = CharacterDetection.detectCharsInPlates(listOfPossiblePlates)

    if len(listOfPossiblePlates) == 0:
        print('License Plate Not Found!')
    else:
        reader = Reader(['en'])
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        listPlate = []
        for plate in listOfPossiblePlates:
            licPlate = plate

            detection = reader.readtext(licPlate.imgPlate)
            licPlate.strChars = ''
            plateNumber = ''
            accuracy = 0
            for item in detection:
                accuracy = accuracy + item[2]
                plateNumber = plateNumber + item[1]

            if len(detection) > 0:
                accuracy = round((accuracy / len(detection)) * 100, 2)

            licPlate.strChars = plateNumber

            platNumber = plateNumber.replace(' ','')
            platNumber = re.search('^(?=.*[0-9])(?=.*[A-Z])([A-Z0-9]+)$',platNumber)
            if platNumber != None:
                licPlate.strChars = platNumber.group(0)
                listPlate.append([licPlate, accuracy])

        if len(listPlate) == 0:
            print('License Plate Not Detected!')
        else:
            listPlate.sort(key=lambda row: (row[1]), reverse=True)
            drawRedRectangleAroundPlate(imgOriginalScene,listPlate[0][0])
            cv2.imshow("Result",imgOriginalScene)
            print('License Plate:',listPlate[0][0].strChars)
            print('Accuracy:',listPlate[0][1])
            cv2.waitKey(0)

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    SCALAR_RED = (0.0, 0.0, 255.0)
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    p2fRectPoints2 = []
    for num in p2fRectPoints:
        subList = []
        for num1 in num:
            subList.append(round(num1))
        p2fRectPoints2.append(subList)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints2[0]), tuple(p2fRectPoints2[1]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints2[1]), tuple(p2fRectPoints2[2]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints2[2]), tuple(p2fRectPoints2[3]), SCALAR_RED, 5)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints2[3]), tuple(p2fRectPoints2[0]), SCALAR_RED, 5)

checkPlate("sample/N262.jpeg")