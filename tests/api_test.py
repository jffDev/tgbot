import requests
import os

PyTorch_REST_API_URL = "http://127.0.0.1:5000/predict"


def predict_result(image_path):

    image = open(image_path, "rb")
    payload = {"image": image}

    result = requests.post(PyTorch_REST_API_URL, files=payload).json()

    if result["success"]:
        print(f"Predicted class: {result['prediction']}")
    else:
        print("Request failed")


if __name__ == "__main__":
    # predict_result("data\\test\\apple_77.jpg")
    predict_result("data\\test\\banana_77.jpg")
