from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label

import os
from re import I
import numpy as np
import random
import time
import codecs
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.param_attr import ParamAttr
from PIL import Image, ImageEnhance

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sn

target_size = [3, 224, 224]
mean_rgb = [127.5, 127.5, 127.5]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_dir = sys.argv[1]
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
data_name = data_dir[data_dir.find('img'):]
save_freeze_dir = "inference_qc-30-" + data_name[:data_name.rfind('/')]

# save_freeze_dir = "inference_qc-30-img5.1.1-no-label-smoothing"
save_confusion_dir = "./figure_result/" + save_freeze_dir + '/'
if not os.path.exists(save_confusion_dir):
    os.makedirs(save_confusion_dir)
paddle.enable_static()
path_prefix = "/home/wangmao/code/PaddleClas/inference_qc/30-img6.1.1/inference"
[inference_program, feed_target_names, fetch_targets] = (
    paddle.static.load_inference_model(path_prefix, exe))


def crop_image(img, target_size):
    width, height = img.size
    w_start = (width - target_size[2]) / 2
    h_start = (height - target_size[1]) / 2
    w_end = w_start + target_size[2]
    h_end = h_start + target_size[1]
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def resize_img(img, target_size):
    ret = img.resize((target_size[1], target_size[2]),
                     Image.Resampling.BILINEAR)
    return ret


def read_image(img_path, data_root=None):
    if data_root is not None:
        img_path = data_root + img_path
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_img(img, target_size)
    img = np.array(img).astype('float32')
    img -= mean_rgb
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    img = img[np.newaxis, :]
    return img


def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program,
                    feed={feed_target_names[0]: tensor_img},
                    fetch_list=fetch_targets)
    return np.argmax(label), label[0][0]


def infer2(image_path):
    tensor_img = read_image(image_path, data_root=data_dir)
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: tensor_img},
                      fetch_list=fetch_targets)
    return np.argmax(results), results[0][0]


def auc(actual, pred):
    fpr, tpr, _ = metrics.roc_curve(actual, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def roc_plot(fpr, tpr, title_name, save_figure=False, save_dir='./'):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=lw,
        label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_name)
    plt.legend(loc="lower right")
    if save_figure:
        plt.savefig('{}{}.png'.format(save_dir, title_name))
    else:
        plt.show()


def eval_all(eval_file="eval.txt", label_file="label_list.txt", infer=infer):
    confusion_flag = True
    single_class = False
    class_detail_flag = True
    eval_file_path = os.path.join(data_dir, eval_file)
    label_file_path = os.path.join(data_dir, label_file)

    label_dict = {}
    label_count = {}
    label_right_count = {}
    right_count = 0
    total_count = 0
    for line in open(label_file_path):
        s = line.splitlines()[0].split('\t')
        label_dict[s[0]] = s[1]
        label_right_count[s[0]] = 0
    # print(label_dict)

    y_true = []
    y_probability = []
    y_predict = []
    with codecs.open(eval_file_path, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        t1 = time.time()

        for line in lines:
            total_count += 1
            parts = line.strip().split()
            result, probability = infer(parts[0])
            y_true.append(int(parts[1]))
            if single_class:
                y_probability.append(probability[np.argmax(probability)])
            else:
                y_probability.append(probability.tolist())
            y_predict.append(result)
            # print("infer result:{0} answer:{1}".format(result, parts[1]))
            if parts[1] in label_count:
                label_count[parts[1]] = label_count[parts[1]] + 1
            else:
                label_count[parts[1]] = 1
            # print(label_count)
            if str(result) == parts[1]:
                right_count += 1
                if parts[1] in label_right_count:
                    label_right_count[parts[1]] = label_right_count[parts[
                        1]] + 1
                else:
                    label_right_count[parts[1]] = 1

        period = time.time() - t1
        acc = right_count / total_count
        print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(
            total_count, "%2.2f sec" % period, acc))
        for iter in label_dict:
            print("class {3} {4}: eval count:{0}/{1} predict accuracy:{2}".
                  format(label_right_count[iter], label_count[
                      iter], label_right_count[iter] / label_count[iter], iter,
                         label_dict[iter]))

    if confusion_flag:
        matrixes = metrics.confusion_matrix(y_true, y_predict)
        print(matrixes)
        con_mat = matrixes
        con_mat_norm = con_mat.astype('float') / con_mat.sum(
            axis=1)[:, np.newaxis]
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        confusion_name = 'Confusion_Matrix—{}'.format(save_freeze_dir[
            save_freeze_dir.rfind('/') + 1:])
        plt.figure(confusion_name, figsize=(7, 7))
        sn.heatmap(con_mat_norm, annot=True, cmap='Blues')
        # sn.heatmap(con_mat_norm, annot=True, fmt='.20g', cmap='Blues')
        plt.ylim(0, len(label_dict) + 1)
        plt.xlabel('Predicted labels')
        plt.ylabel('True      labels')
        plt.title(confusion_name)
        plt.savefig('{}{}.png'.format(save_confusion_dir, confusion_name))

        target_names = []
        for iter in label_dict:
            target_names.append(label_dict[iter])
        print(
            metrics.classification_report(
                y_true, y_predict, target_names=target_names))

        if not single_class:
            label_types = np.unique(y_true)
            n_class = label_types.size
            y_one_hot = label_binarize(y_true, classes=np.arange(n_class))
            y_one_hot = np.array(y_one_hot)
            y_n_probability = np.array(y_probability)
            y_true = y_one_hot
            y_probability = y_n_probability
            if class_detail_flag:
                for i in range(len(label_dict)):
                    print('class:{} auc: {}'.format(
                        i, auc(y_true[:, i], y_probability[:, i])))
                    fpr, tpr, thresholds = metrics.roc_curve(
                        y_true[:, i], y_probability[:, i], pos_label=1)
                    Roc_name = 'Roc_Curve_class-{}—{}'.format(
                        i, save_freeze_dir[save_freeze_dir.rfind('/') + 1:])
                    roc_plot(
                        fpr,
                        tpr,
                        title_name=Roc_name,
                        save_figure=True,
                        save_dir=save_confusion_dir)
            y_true = y_one_hot.ravel()
            y_probability = y_n_probability.ravel()

        print('auc: {}'.format(auc(y_true, y_probability)))
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true, y_probability, pos_label=1)
        # print('fpr: {}'.format(fpr))
        # print('tpr: {}'.format(tpr))
        # print('thresholds: {}'.format(thresholds))
        Roc_name = 'Roc_Curve—{}'.format(save_freeze_dir[save_freeze_dir.rfind(
            '/') + 1:])
        roc_plot(
            fpr,
            tpr,
            title_name=Roc_name,
            save_figure=True,
            save_dir=save_confusion_dir)

    return acc


if __name__ == '__main__':
    if False:
        eval_all(infer=infer)
    else:
        eval_all(eval_file='val_list.txt', infer=infer2)
