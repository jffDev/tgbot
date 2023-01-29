from pathlib import Path
import torch

use_gpu = True


def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_model():
    # load model from disk
    PATH = Path("../models") / "fast_rcnn_model.pt"
    PATH = PATH.resolve().absolute()

    print('path = ' + str(PATH))

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    model = torch.load(str(PATH), map_location=device())
    model.eval()
    return model
