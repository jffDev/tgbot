# TG Bot with PyTorch model

Обученная модель fast_rcnn_model.pt занимат порядка 160мб и не помещается в репозитории github

## Disclaimer

Проект в стадии разработки и адаптации под модель FAST R-CNN

## Запуск

Для запуска необходимо перейти в web_deploy и запустить start_app.py

Для теста необходимо перейти в tests и запустить api_test.py

## Log:

Loading PyTorch model and Flask starting server.
Please wait until server has fully started...
Is CUDA supported by this system? True
CUDA version: 11.7
ID of current CUDA device:0
Name of current CUDA device:NVIDIA GeForce RTX 3090 Ti
* Serving Flask app 'api'
* Debug mode: on

***
<class 'torch.Tensor'>\
[{'boxes': tensor([[ 68.5390,  62.2780, 180.9433, 162.6742],\
[109.5106, 116.6273, 218.8277, 222.4048],\
[209.0050, 143.4062, 298.8925, 226.9113],\
[ 12.0190,  24.5841,  99.9162,  99.0302],\
[209.1016, 142.0529, 300.0000, 226.1681],\
[104.7043, 100.3560, 221.1093, 229.0000],\
[209.6621, 145.8296, 291.8750, 226.5049],\
[ 69.4276,  56.2068, 218.0771, 177.7239],\
[  6.3519,  21.3432, 104.0385, 102.8566],\
[ 65.6014,  60.9822, 177.3932, 161.7981],\
[ 11.9727,  25.5773,  93.6112,  99.4727]], device='cuda:0'),\

'labels': tensor([1, 1, 1, 1, 3, 3, 2, 3, 3, 2, 2], device='cuda:0'),\

'scores': tensor([0.9491, 0.8802, 0.4843, 0.2963, 0.1865, 0.1504, 0.1314, 0.1227, 0.0853,0.0722, 0.0501], device='cuda:0')\
}]
***

# Project description