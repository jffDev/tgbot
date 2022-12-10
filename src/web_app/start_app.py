import os
import sys
from pathlib import Path

import numpy as np
import cv2
import flask
import torch
import torchvision
from torchvision import transforms as T

import_path = os.getcwd()
sys.path.insert(0, import_path)

app = flask.Flask(__name__)
use_gpu = True
model = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    classes = (
        'apple',
        'orange',
        'banana'
    )

    if flask.request.method == "POST":
        print(flask.request.files)
        if flask.request.files.get("image"):
            # Read the image in PIL format
            f = flask.request.files["image"]
            print("title", f.filename.title())

            f.save(f.filename)  # save file to disk

            img = cv2.imread(f.filename.title())

            # Define a transform to convert the image to tensor
            transform = T.transforms.ToTensor()

            # Convert the image to PyTorch tensor
            img = transform(img)

            print(type(img))

            with torch.no_grad():
                prediction = model([img.to(device)])

            print(prediction)
            # keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], 0.2)
            # print(keep)
            # data["prediction"] = classes[prediction['labels'][keep]]

            data["prediction"] = 'apple'
            data["success"] = True
            # os.remove(f.filename)

            return flask.jsonify(data)


def tensorToPIL(img):
    return T.transforms.ToPILImage()(img).convert('RGB')


def load_model() -> None:
    global model
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
    print("Loading PyTorch model and Flask starting server.")
    print("Please wait until server has fully started...")
    load_model()
    app.run(debug=True)
