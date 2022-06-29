import codecs
import os
import shutil
from xmlrpc.client import boolean
from PIL import Image
import sys
import argparse
from sklearn.preprocessing import label_binarize
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    'data_path', default='', type=str, help='Directory path of split dataset')
parser.add_argument(
    '-um', '--use_multi', default=False, type=bool, help='use multi label')
args = parser.parse_args()

dir_class = args.data_path
img_path = 'jpg'
image_dir = os.path.join(dir_class, img_path)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

use_multi_label = args.use_multi

train_file = codecs.open(os.path.join(dir_class, "train_list.txt"), 'w')
val_file = codecs.open(os.path.join(dir_class, "val_list.txt"), 'w')

if use_multi_label:
    multilabel_train_file = codecs.open(
        os.path.join(dir_class, "multilabel_train_list.txt"), 'w')
    multilabel_val_file = codecs.open(
        os.path.join(dir_class, "multilabel_val_list.txt"), 'w')

for name_dir in os.listdir(dir_class):
    if 'txt' in name_dir or 'jpg' in name_dir:
        continue
    all_file_dir = dir_class + name_dir
    print(all_file_dir)

    class_list = [
        c for c in os.listdir(all_file_dir)
        if os.path.isdir(os.path.join(all_file_dir, c))
    ]
    class_list.sort()
    class_num = len(class_list)
    # print(class_list)
    # sys.exit()
    with codecs.open(os.path.join(dir_class, "label_list.txt"),
                     "w") as label_list:
        label_id = 0
        for class_dir in class_list:
            if use_multi_label:
                y_one_hot = label_binarize(
                    [label_id], classes=np.arange(class_num))
                y_one_hot = y_one_hot[0][:].tolist()
                y_one_hot_str = ','.join(repr(e) for e in y_one_hot)
                # print(y_one_hot_str)

            # sys.exit()
            label_list.write("{0}\t{1}\n".format(label_id, class_dir))
            # print(str(class_dir) + '--' + str(label_id))
            image_path_pre = os.path.join(all_file_dir, class_dir)
            for file in os.listdir(image_path_pre):
                # print(file)
                try:
                    img = Image.open(os.path.join(image_path_pre, file))
                    # if random.uniform(0, 1) <= train_ratio:
                    if 'train' in name_dir:
                        shutil.copyfile(
                            os.path.join(image_path_pre, file),
                            os.path.join(image_dir, file))
                        train_file.write("{0} {1}\n".format(
                            os.path.join(img_path, file), label_id))
                        if use_multi_label:
                            multilabel_train_file.write("{0} {1}\n".format(
                                os.path.join(img_path, file), y_one_hot_str))
                    else:
                        shutil.copyfile(
                            os.path.join(image_path_pre, file),
                            os.path.join(image_dir, file))
                        val_file.write("{0} {1}\n".format(
                            os.path.join(img_path, file), label_id))
                        if use_multi_label:
                            multilabel_val_file.write("{0} {1}\n".format(
                                os.path.join(img_path, file), y_one_hot_str))
                except Exception as e:
                    pass
            label_id += 1

train_file.close()
val_file.close()
if use_multi_label:
    multilabel_train_file.close()
    multilabel_val_file.close()

print("finished data split!!!")
