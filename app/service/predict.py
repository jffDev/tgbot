import numpy as np

import cv2
import torch
from torchvision import transforms as T
from app.features.utils import ModelUtils
from app.models.model import load_model, device


def predict(filename):
    print("Loading PyTorch model")
    model = load_model()

    data = {"success": False}
    classes = ['_', 'apple', 'orange', 'banana']

    # get image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # img = cv2.resize(img, (480, 480), cv2.INTER_AREA)
    img /= 255.0

    # Define a transform to convert the image to tensor
    transform = T.transforms.ToTensor()

    # Convert the image to PyTorch tensor
    img2 = transform(img)

    utils = ModelUtils()

    with torch.no_grad():
        prediction = model([img2.to(device())])[0]
        print(prediction)
        prediction = utils.nms(prediction, threshold=0.3)
    source_img = cv2.imread(filename)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    utils.plot_box(filename, source_img, prediction, classes)

    data["prediction"] = classes[prediction['labels'][0]]
    data["success"] = True
    data["filename"] = filename
    # os.remove(f.filename)
    return data
