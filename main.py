from pathlib import Path

from src.Frameworks.Helper.h5FileManagement import h5FileManagement
from src.Frameworks.Models.ANALYSIS_TYPE_MODEL import ANALYSIS_TYPE_MODEL as type
from src.Frameworks.Models.MODEL_TYPE import MODEL_TYPE as ModelType
import cv2
import os


def readParameters():
    file = open(os.getenv('APPDATA') + "\\ScalpChecker\\TEMP\\Properties.txt", encoding='UTF8')
    parameters = file.read().split(";")
    return parameters


def getModels(type):
    file = open(os.getenv('APPDATA') + "\\ScalpChecker\\TEMP\\Models\\ModelDir_" + type + ".txt", encoding='UTF8')
    root = file.read()

    return Path(root.strip('\n'))


def createTxtFile(text, id):
    ROOT_DIR = os.getenv('APPDATA') + "\\ScalpChecker\\Results"

    if not os.path.exists(ROOT_DIR):
        PARENT_DIR = os.getenv('APPDATA') + "\\ScalpChecker"
        resultDirectory = "Results"
        realPath = os.path.join(PARENT_DIR, resultDirectory)

        os.mkdir(realPath)

    with open(ROOT_DIR + "\\Results_" + id + ".txt", "w") as f:
        f.write(text)


if __name__ == '__main__':
    params = readParameters()

    helper = h5FileManagement()
    id = params[0]
    types = params[1].split(", ")
    img_dir = params[2]
    modelType = params[3]
    usingModel = ModelType.Keras
    result = ""

    if modelType == "Keras":
        usingModel = ModelType.Keras

    elif modelType == "ViT":
        usingModel = ModelType.ViT

    for i in types:
        if i == "BIDUM":
            status_BIDUM = helper.analysis(img_dir, type.BIDUM, id, getModels("BIDUM"), usingModel)
            result += "BIDUM : %s\n" % status_BIDUM

        elif i == "FIJI":
            status_FIJI = helper.analysis(img_dir, type.FIJI, id, getModels("FIJI"), usingModel)
            result += "FIJI : %s\n" % status_FIJI

        elif i == "MISE":
            status_MISE = helper.analysis(img_dir, type.MISE, id, getModels("MISE"), usingModel)
            result += "MISE : %s\n" % status_MISE

        elif i == "TALMO":
            status_TALMO = helper.analysis(img_dir, type.TALMO, id, getModels("TALMO"), usingModel)
            result += "TALMO : %s\n" % status_TALMO

        elif i == "HONGBAN":
            status_HONGBAN = helper.analysis(img_dir, type.HONGBAN, id, getModels("HONGBAN"), usingModel)
            result += "HONGBAN : %s\n" % status_HONGBAN

        elif i == "NONGPO":
            status_NONGPO = helper.analysis(img_dir, type.NONGPO, id, getModels("NONGPO"), usingModel)
            result += "NONGPO : %s\n" % status_NONGPO

    print(result)
    os.remove(os.getenv('APPDATA') + "\\ScalpChecker\\TEMP\\Properties.txt")
    createTxtFile(result, id)
