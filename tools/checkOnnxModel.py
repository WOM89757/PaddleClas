import onnx
import numpy as np
import onnxruntime
import sys
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

onnx_file = sys.argv[1]
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# means = [0.5, 0.5, 0.5]
# stds = [0.5, 0.5, 0.5]
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=means, std=stds),
    ])


def cv2_transform(cv2_img):
    img = cv2_img.copy()
    img = cv2.resize(cv2_img, (224, 224), Image.BILINEAR)
    img = np.array(img[:, :, ::-1], dtype=np.float32)
    img = img / 255
    img = img - np.array(means, dtype=np.float32)
    img = img / np.array(stds, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :]
    img = torch.from_numpy(img)
    return img


image = Image.open(sys.argv[2])
img = get_test_transform()(image)
img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224
print("input img mean {} and std {}".format(img.mean(), img.std()))
# print(img.shape)

imageop = cv2.imread(sys.argv[2])
imageop = cv2_transform(imageop)
print("input imageop mean {} and std {}".format(imageop.mean(), imageop.std()))
# print(imageop.shape)

##onnx session
resnet_session = onnxruntime.InferenceSession(onnx_file)
#compute ONNX Runtime output prediction
inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
outs = resnet_session.run(None, inputs)[0]
np.set_printoptions(suppress=True)
print("onnx weights {},onnx prediction:{}".format(
    np.around(outs, 4), outs.argmax(axis=1)[0]))

inputs = {resnet_session.get_inputs()[0].name: to_numpy(imageop)}
outs = resnet_session.run(None, inputs)[0]
print("onnx weights {},onnx prediction:{}".format(
    np.around(outs, 4), outs.argmax(axis=1)[0]))

print("Exported model has been predicted by ONNXRuntime!")
