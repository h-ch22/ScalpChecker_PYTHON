import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from keras.utils import img_to_array, load_img
from src.Frameworks.Models.ANALYSIS_TYPE_MODEL import ANALYSIS_TYPE_MODEL
from keras.models import Model, load_model
from datetime import date

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
    #img = img_to_array(img)
    img = cv2.resize(img, (480, 480))
    img = np.array(img)[:,:,::-1]
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


def __saveImage__(img, type: ANALYSIS_TYPE_MODEL, id):
    date = today.strftime("%m.%d.%y")
    ROOT_DIR = os.getenv('APPDATA') + "\\ScalpChecker"
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
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        class_names = ["Severity : level0", "Severity : level1", "Severity : level2", "Severity : level3"]

        classes = np.argmax(model.predict(x), axis=-1)

        names = [class_names[i] for i in classes]

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        img = __loadImage__(path)
        img = np.expand_dims(img, axis=0)
        cam = __createGradCAM__(path, model, type, id, img)

        #cv2.imshow(cam)

        return names


class h5FileManagement:
    def __init__(self):
        super().__init__()

        print("Tensorflow loaded with version %s" % (tf.__version__))

    def analysis(self, img, type: ANALYSIS_TYPE_MODEL, id, modelRoot):
        if self.detectGPU():
            print("GPU is available, tensorflow will work with GPU : 0")
            with tf.device('/device:GPU:0'):
                result = __analysis__(img, type, id, modelRoot)
                return result

        else:
            print("GPU is not available, tensorflow will work with CPU")

            with tf.device('/device:cpu:0'):
                result = __analysis__(img, type, id, modelRoot)
                return result

    def detectGPU(self):
        return len(tf.config.list_physical_devices('GPU')) > 0