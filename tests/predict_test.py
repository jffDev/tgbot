from app.service.predict import predict
import shutil


def predict_result(image_path, image_name):

    shutil.copy(image_path + image_name, '.')
    result = predict('./'+image_name)

    if result["success"]:
        print(f"Predicted class: {result['prediction']}, url: {image_path}")
    else:
        print("Request failed")


if __name__ == "__main__":
    predict_result("../data/test/", 'mixed_23.jpg')
    predict_result("../data/test/", 'orange_80.jpg')
    predict_result("../data/test/", 'apple_77.jpg')
    predict_result("../data/test/", 'banana_77.jpg')
