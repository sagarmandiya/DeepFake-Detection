import cv2
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import os


def decode_predictions(preds, top=2):
    """
    Decode the prediction of the image and return the result with probablity of the classes.

    Args:
        preds (numpy.ndarray): [Preds is the encoded predicted value.]
        top (int, optional): [It is variable which shows us the top predicted class on the basis of probablity]. Defaults to 2.

    Returns:
        result [list]: [List of prediction with their respective probability]
    """
    global CLASS_INDEX, json_path
    with open(json_path) as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results


# Change the model path, with the name of the model with wich inference is required.
model_path, json_path = "../Reference_Data/Model/Xception/", "./class_index.json"
real_dir = "../Datasets/small-dataset/real/"
fake_dir = "../Datasets/small-dataset/fake/"
for filename in os.listdir(real_dir):
    file = real_dir + filename
    print(file)
    width, height, channel = 128, 128, 3
    size = (width, height)

    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[INFO]. Memory Growth on GPU Enabled.")
    except:
        print("[ERROR]. Could Not Enable Memory Growth on GPU.")

    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(file)
    img = cv2.resize(img, size)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    start_time = datetime.datetime.now()
    # Change from "GPU" to "CPU" if GPU is anavailable
    with tf.device('GPU:0'):
        preds = model.predict(img)
    end_time = datetime.datetime.now()

    prediction_time = end_time - start_time
    print('\n[INFO]. Prediction : ', decode_predictions(preds, top=2)[0][0])
    print('\n[INFO]. Prediction Time : {:4.1f} ms'.format(
        prediction_time.total_seconds()*1000))
