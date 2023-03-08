import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow_addons as tfa

from keras.utils import img_to_array, load_img
from keras_preprocessing.image import ImageDataGenerator

from include.vit_keras import _visualize
from include.vit_keras import vit, utils
from src.Frameworks.Models.ANALYSIS_TYPE_MODEL import ANALYSIS_TYPE_MODEL
from keras.models import Model, load_model
from datetime import date

from src.Frameworks.Models.MODEL_TYPE import MODEL_TYPE

today = date.today()


def __ImgReadUtf8__(path):
    stream = open(path.encode('utf-8'), 'rb')
    bytes = bytearray(stream.read())
    numpyArr = np.asarray(bytes, dtype=np.uint8)

    return cv2.imdecode(numpyArr, cv2.IMREAD_UNCHANGED)


def __loadImage__(imgPath):
    img = load_img(imgPath, target_size=(480, 480))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32')
    x /= 255

    return x


def __createGradCAM__(img, model, type, id, x):
    grad_model = Model(inputs=model.inputs, outputs=[model.layers[-3].output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(__loadImage__(img))
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img = __ImgReadUtf8__(img)
    # img = img_to_array(img)
    img = cv2.resize(img, (480, 480))
    img = np.array(img)[:, :, ::-1]
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (480, 480))
    cam = heatmap + np.float32(img)

    cam /= np.max(cam)
    # heatmap = np.uint8(255 * heatmap)
    # jet = cm.get_cmap("jet")
    # jet_colors = jet(np.arange(256))[:, :3]
    # jet_heatmap = jet_colors[heatmap]
    #
    # jet_heatmap = array_to_img(jet_heatmap)
    # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    # jet_heatmap = img_to_array(jet_heatmap)

    # superimposed_img = jet_heatmap * 0.4 + img

    __saveImage__(np.uint8(255 * cam), type, id)


def __createAttentionMap__(img, model, type, id):
    path = img
    img = load_img(img, target_size=(224, 224))
    img = img_to_array(img)

    outputs, weights, num_heads, attention_mask = _visualize.attention_map(model=model.layers[0], image=img)

    _img = __ImgReadUtf8__(path)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    _img = cv2.resize(_img, (224, 224))

    heatmap_img = cv2.applyColorMap((attention_mask * img).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(heatmap_img, 0.5, _img, 0.5, 0)

    __saveImage__(overlay_img, type, id)


def __saveImage__(img, type: ANALYSIS_TYPE_MODEL, id):
    date = today.strftime("%m.%d.%y")
    # ROOT_DIR = os.getenv('APPDATA') + "\\ScalpChecker"
    ROOT_DIR = os.getenv('APPDATA') + "\\ScalpChecker\\Results_Keras"
    IMG_DIR = ROOT_DIR + "\\" + str(id)

    if not os.path.exists(ROOT_DIR):
        print("ROOT DIR IS NOT EXISTS")
        PARENT_DIR = os.getenv('APPDATA')
        appDirectory = "ScalpChecker"
        appPath = os.path.join(PARENT_DIR, appDirectory)
        os.mkdir(appPath)

        imgDirectory = str(today)
        imgPath = os.path.join(appPath, imgDirectory)
        os.mkdir(imgPath)

    elif not os.path.exists(IMG_DIR):
        print("IMG DIR IS NOT EXISTS")
        imgDirectory = str(id)
        imgPath = os.path.join(ROOT_DIR, imgDirectory)
        os.mkdir(imgPath)

    cv2.imwrite(IMG_DIR + "\\" + str(date) + "_" + type.name + "_" + str(id) + ".png", img)


def __analysis_ViT__(img, type: ANALYSIS_TYPE_MODEL, id, modelRoot):
    if img is None:
        raise Exception("Cannot get image from directory")

    else:
        path = img
        model = load_model(modelRoot)
        # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
        #               optimizer=tfa.optimizers.RectifiedAdam(learning_rate=1e-4),
        #               metrics=['accuracy'],
        #               activation='softmax')

        x = utils.read(path, 224)
        x = vit.preprocess_inputs(x).reshape(1, 224, 224, 3)

        classes = {
            0 : "Severity : level0",
            1 : "Severity : level1",
            2 : "Severity : level2",
            3 : "Severity : level3"
        }

        #result = classes[y[0].argmax()]
        result = np.argmax(model.predict(x), axis=-1)
        names = [classes[i] for i in result]

        __createAttentionMap__(path, model, type, id)

        return names


def __analysis__(img, type: ANALYSIS_TYPE_MODEL, id, modelRoot):
    if img is None:
        print("Image directory is none.")
        return False

    else:
        path = img
        test_image = tf.keras.utils.load_img(img, target_size=(480, 480))
        x = tf.keras.utils.img_to_array(test_image)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32')
        x /= 255

        model = load_model(modelRoot)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        class_names = ["Severity : level0", "Severity : level1", "Severity : level2", "Severity : level3"]

        classes = np.argmax(model.predict(x), axis=-1)

        names = [class_names[i] for i in classes]

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        __createGradCAM__(path, model, type, id, img)

        return names

def __evaluate__(type : ANALYSIS_TYPE_MODEL, modelRoot):
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 200

    TEST_PATH_MISE = r'C:\Users\USER\Desktop\2023\ScalpChecker\assets\ScalpChecker\scalpdataset\trainvaltestdata\MISE\split_class4_MISE_480x480\test'
    TEST_PATH_FIJI = r'C:\Users\USER\Desktop\2023\ScalpChecker\assets\ScalpChecker\scalpdataset\trainvaltestdata\FIJI\split_class4_FIJI_480x480\test'
    TEST_PATH_NONGPO = r'C:\Users\USER\Desktop\2023\ScalpChecker\assets\ScalpChecker\scalpdataset\trainvaltestdata\NONGPO\split_class4_NONGPO_VER2_480x480\test'
    TEST_PATH_HONGBAN = r'C:\Users\USER\Desktop\2023\ScalpChecker\assets\ScalpChecker\scalpdataset\trainvaltestdata\HONGBAN\split_class4_HONGBAN_480x480\test'
    TEST_PATH_BIDUM = r'C:\Users\USER\Desktop\2023\ScalpChecker\assets\ScalpChecker\scalpdataset\trainvaltestdata\BIDUM\split_class4_BIDUM_480x480\test'
    TEST_PATH_TALMO = r'C:\Users\USER\Desktop\2023\ScalpChecker\assets\ScalpChecker\scalpdataset\trainvaltestdata\TALMO\split_class4_TALMO_480x480\test'


    datagen = ImageDataGenerator(rescale=1. / 255,
                                 samplewise_center=True,
                                 samplewise_std_normalization=True)

    if type == ANALYSIS_TYPE_MODEL.BIDUM_ViT:
        typeAsStr = "BIDUM"
        test_gen = datagen.flow_from_directory(
            directory=TEST_PATH_BIDUM,
            batch_size=BATCH_SIZE,
            seed=1,
            color_mode='rgb',
            shuffle=False,
            class_mode='categorical',
            target_size=(IMG_SIZE, IMG_SIZE))

    elif type == ANALYSIS_TYPE_MODEL.FIJI_ViT:
        typeAsStr = "FIJI"

        test_gen = datagen.flow_from_directory(
            directory=TEST_PATH_FIJI,
            batch_size=BATCH_SIZE,
            seed=1,
            color_mode='rgb',
            shuffle=False,
            class_mode='categorical',
            target_size=(IMG_SIZE, IMG_SIZE))

    elif type == ANALYSIS_TYPE_MODEL.MISE_ViT:
        typeAsStr = "MISE"

        test_gen = datagen.flow_from_directory(
            directory=TEST_PATH_MISE,
            batch_size=BATCH_SIZE,
            seed=1,
            color_mode='rgb',
            shuffle=False,
            class_mode='categorical',
            target_size=(IMG_SIZE, IMG_SIZE))

    elif type == ANALYSIS_TYPE_MODEL.TALMO_ViT:
        typeAsStr = "TALMO"

        test_gen = datagen.flow_from_directory(
            directory=TEST_PATH_TALMO,
            batch_size=BATCH_SIZE,
            seed=1,
            color_mode='rgb',
            shuffle=False,
            class_mode='categorical',
            target_size=(IMG_SIZE, IMG_SIZE))

    elif type == ANALYSIS_TYPE_MODEL.HONGBAN_ViT:
        typeAsStr = "HONGBAN"

        test_gen = datagen.flow_from_directory(
            directory=TEST_PATH_HONGBAN,
            batch_size=BATCH_SIZE,
            seed=1,
            color_mode='rgb',
            shuffle=False,
            class_mode='categorical',
            target_size=(IMG_SIZE, IMG_SIZE))

    elif type == ANALYSIS_TYPE_MODEL.NONGPO_ViT:
        typeAsStr = "NONGPO"

        test_gen = datagen.flow_from_directory(
            directory=TEST_PATH_NONGPO,
            batch_size=BATCH_SIZE,
            seed=1,
            color_mode='rgb',
            shuffle=False,
            class_mode='categorical',
            target_size=(IMG_SIZE, IMG_SIZE))

    else:
        return

    model = load_model(modelRoot)
    test_evaluate = model.evaluate(test_gen)
    print(typeAsStr + " : " + str(test_evaluate))

class h5FileManagement:
    def __init__(self):
        super().__init__()

        print("Tensorflow loaded with version %s" % (tf.__version__))

    def analysis(self, img, type: ANALYSIS_TYPE_MODEL, id, modelRoot, modelType):
        if self.detectGPU():
            print("GPU is available, tensorflow will work with GPU : 0")
            with tf.device('/device:GPU:0'):
                if modelType == MODEL_TYPE.ViT:
                    result = __analysis_ViT__(img, type, id, modelRoot)
                    return result

                else:
                    result = __analysis__(img, type, id, modelRoot)
                    return result

        else:
            print("GPU is not available, tensorflow will work with CPU")

            with tf.device('/device:cpu:0'):
                if modelType == MODEL_TYPE.ViT:
                    result = __analysis_ViT__(img, type, id, modelRoot)
                    return result

                else:
                    result = __analysis__(img, type, id, modelRoot)
                    return result

    def evaluate(self, type : ANALYSIS_TYPE_MODEL, modelRoot):
        with tf.device('/device:GPU:0'):
            __evaluate__(type, modelRoot)

    def detectGPU(self):
        return len(tf.config.list_physical_devices('GPU')) > 0
