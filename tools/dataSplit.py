import codecs
import os
import shutil
from PIL import Image
import sys

train_ratio = 0.5
dir_class = sys.argv[1]
data_path = 'jpg'
image_dir = os.path.join(dir_class, data_path)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

train_file = codecs.open(os.path.join(dir_class, "train_list.txt"), 'w')
val_file = codecs.open(os.path.join(dir_class, "val_list.txt"), 'w')

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
    # print(class_list)

    with codecs.open(os.path.join(dir_class, "label_list.txt"),
                     "w") as label_list:
        label_id = 0
        for class_dir in class_list:
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
                            os.path.join(data_path, file), label_id))
                    else:
                        shutil.copyfile(
                            os.path.join(image_path_pre, file),
                            os.path.join(image_dir, file))
                        val_file.write("{0} {1}\n".format(
                            os.path.join(data_path, file), label_id))
                except Exception as e:
                    pass
            label_id += 1

train_file.close()
val_file.close()

print("finished data split!!!")
