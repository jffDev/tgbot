import os
import sys
import numpy as np
from pathlib import Path

import cv2
import flask
import torch
from torchvision import transforms as T
from src.features.utils import ModelUtils

import_path = os.getcwd()
sys.path.insert(0, import_path)

app = flask.Flask(__name__)
use_gpu = True
model = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    classes = ['_', 'apple', 'orange', 'banana']

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]
            # save file to disk
            f.save(f.filename)

            img_name = f.filename.title()
            print(img_name)

            # get image
            img = cv2.imread(f.filename.title())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # img = cv2.resize(img, (480, 480), cv2.INTER_AREA)
            img /= 255.0

            # Define a transform to convert the image to tensor
            transform = T.transforms.ToTensor()

            # Convert the image to PyTorch tensor
            img = transform(img)

            utils = ModelUtils()

            with torch.no_grad():
                prediction = model([img.to(device)])[0]
                print(prediction)
                prediction = utils.nms(prediction, threshold=0.3)
            source_img = cv2.imread(img_name)
            utils.plot_box(source_img, prediction, classes)

            print(prediction)

            data["prediction"] = classes[prediction['labels'][0]]
            data["success"] = True
            os.remove(f.filename)

            return flask.jsonify(data)




def load_model() -> None:
    global model
    # load model from disk
    PATH = Path("../../models") / "fast_rcnn_model.pt"
    PATH = PATH.resolve().absolute()

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    model = torch.load(str(PATH), map_location=device)
    model.eval()


if __name__ == "__main__":
    print("Loading PyTorch model")
    load_model()
    print("Starting Flask server")
    app.run(debug=True)
