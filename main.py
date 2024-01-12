import datetime
import glob
import os
import argparse
import string
import random

from pathlib import Path
from datetime import date
from src.Frameworks.Helper.h5FileManagement import h5FileManagement
from src.Frameworks.Models.AnalysisTypeModel import AnalysisTypeModel as type
from src.Frameworks.Models.ModelType import ModelType as ModelType



def readParameters():
    file = open(os.getenv('APPDATA') + "\\ScalpChecker\\TEMP\\Properties.txt", encoding='UTF8')
    parameters = file.read().split(";")
    return parameters


def getModels(type):
    file = open(os.getenv('APPDATA') + "\\ScalpChecker\\TEMP\\Models\\ModelDir_" + type + ".txt", encoding='UTF8')
    root = file.read()

    return Path(root.strip('\n'))


def getModel(type, dir, modelType):
    return dir + f"/{type}.h5" if modelType == ModelType.Keras else dir + f"/{type}_ViT.h5"


def createTxtFile(text, id):
    ROOT_DIR = os.getenv('APPDATA') + "\\ScalpChecker\\Results"

    if not os.path.exists(ROOT_DIR):
        PARENT_DIR = os.getenv('APPDATA') + "\\ScalpChecker"
        resultDirectory = "Results"
        realPath = os.path.join(PARENT_DIR, resultDirectory)

        os.mkdir(realPath)

    with open(ROOT_DIR + "\\Results_" + id + ".txt", "w") as f:
        f.write(text)


def createId():
    letters_set = string.ascii_letters + string.ascii_lowercase + string.digits
    random_list = random.sample(letters_set, 8)
    today = date.today().strftime("%m.%d.%y")
    id = ''.join(random_list)

    return f'{today}_{id}'


def predict_all(img, model_dir, types, id):
    model_types = [ModelType.Keras, ModelType.ViT]
    result = ""

    for model_type in model_types:
        for i in types:
            status = helper.analysis(img, get_type(i), f"{id}-EfficientNet" if model_type == ModelType.Keras else f"{id}-ViT", getModel(i, model_dir, model_type), model_type)
            result += f"{i} : {status}\n"

        print(result)
        createTxtFile(result, f"{id}-EfficientNet" if model_type == ModelType.Keras else f"{id}-ViT")
        result = ""


def get_type(type_as_str):
    if type_as_str == "BIDUM":
        return type.BIDUM

    elif type_as_str == "FIJI":
        return type.FIJI

    elif type_as_str == "MISE":
        return type.MISE

    elif type_as_str == "TALMO":
        return type.TALMO

    elif type_as_str == "HONGBAN":
        return type.HONGBAN

    elif type_as_str == "NONGPO":
        return type.NONGPO


if __name__ == '__main__':
    helper = h5FileManagement()

    parser = argparse.ArgumentParser(description='API Level of Scalp Checker')
    parser.add_argument('--d', default='', help='Define your image file or directory')

    parser.add_argument('--m', default='', help='Define your model root')
    parser.add_argument('--t', default='', help='Define your model type (EfficientNet / ViT / All)')
    args = parser.parse_args()

    if args.d == "" or args.m == "" or args.t == "":
        params = readParameters()
        index = 0

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
            status = helper.analysis(img_dir, get_type(i), id, getModels(i), usingModel)
            result += f"{i} : {status}\n"

        print(result)
        os.remove(os.getenv('APPDATA') + "\\ScalpChecker\\TEMP\\Properties.txt")
        createTxtFile(result, id)

    else:
        if args.d == "":
            raise Exception("Data directory not specified")

        elif args.m == "":
            raise Exception("Model directory not specified")

        elif args.t == "":
            raise Exception("Model type does not specified")

        else:
            data = args.d
            model_dir = args.m
            model_type = args.t

            if model_type != "All":
                usingModel = ModelType.Keras if model_type == "EfficientNet" else ModelType.ViT

                if not os.path.isdir(data):
                    id = createId()
                    types = ["BIDUM", "FIJI", "MISE", "TALMO", "HONGBAN", "NONGPO"]
                    img_dir = data
                    result = ""

                    for i in types:
                        status = helper.analysis(img_dir, get_type(i), id, getModel(i, model_dir, usingModel), usingModel)
                        result += f"{i} : {status}\n"

                    print(result)
                    createTxtFile(result, id)

                else:
                    files = glob.glob(data)

                    for file in files:
                        id = createId()
                        types = ["BIDUM", "FIJI", "MISE", "TALMO", "HONGBAN", "NONGPO"]
                        img_dir = data + f"\\{file}"
                        result = ""

                        for i in types:
                            status = helper.analysis(img_dir, get_type(i), id, getModel(i, model_dir, usingModel), usingModel)
                            result += f"{i} : {status}\n"

                        print(result)
                        createTxtFile(result, id)

            else:
                if not os.path.isdir(data):
                    id = createId()
                    types = ["BIDUM", "FIJI", "MISE", "TALMO", "HONGBAN", "NONGPO"]
                    img_dir = data
                    result = ""

                    predict_all(data, model_dir, types, id)

                else:
                    files = glob.glob(f"{data}/*")

                    for file in files:
                        id = createId()
                        types = ["BIDUM", "FIJI", "MISE", "TALMO", "HONGBAN", "NONGPO"]
                        result = ""

                        predict_all(file, model_dir, types, id)
