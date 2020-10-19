import os
from skimage import io
from tqdm import tqdm


def norm_bb(image_w, image_h, x1, y1, x2, y2):
    x_centre = (x1 + x2) / float(2)
    norm_x_centre = x_centre / float(image_w)
    y_centre = (y1 + y2) / float(2)
    norm_y_centre = y_centre / float(image_h)
    w = x2 - x1
    norm_w = w / float(image_w)
    h = y2 - y1
    norm_h = h / float(image_h)
    return (norm_x_centre, norm_y_centre, norm_w, norm_h)


def gen_yolov5_format(data_dir, image_list, bbox_list, label_list, output_file):

    with open(output_file, 'w') as fwrite:
        for index in tqdm(range(len(image_list))):
            image_path = image_list[index].strip()
            image_file = os.path.join(data_dir, image_path)
            fwrite.write(image_file + "\n")

            image = io.imread(image_file)
            image_h, image_w = image.shape[:2]

            bbox_xy = bbox_list[index].strip().split(' ')

            (x, y, w, h) = norm_bb(image_w, image_h, int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3]))
            class_label = int(label_list[index].strip()) - 1

            image_name = image_file.split('.')[0] + '.txt'
            with open(image_name, 'w') as fout:
                output_line = str(class_label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                fout.write(output_line + "\n")


data_dir = '/Users/mackim/datasets/deepfashion_c'

image_files = ['train.txt', 'val.txt', 'test.txt']
bbox_files = ['train_bbox.txt', 'val_bbox.txt', 'test_bbox.txt']
label_files = ['train_cate.txt', 'val_cate.txt', 'test_cate.txt']
output_files = ['train_dfc.txt', 'val_dfc.txt', 'test_dfc.txt']

for i in range(len(image_files)):
    print("Converting deepfashion-c {} file to Yolo format...".format(image_files[i]))
    with open(os.path.join(data_dir, 'anno_fine', image_files[i]), 'r') as fin:
        image_list = fin.readlines()

    with open(os.path.join(data_dir, 'anno_fine', bbox_files[i]), 'r') as fin:
        bbox_list = fin.readlines()

    with open(os.path.join(data_dir, 'anno_fine', label_files[i]), 'r') as fin:
        label_list = fin.readlines()

    assert len(image_list) == len(bbox_list)
    assert len(bbox_list) == len(label_list)

    output_file = os.path.join(data_dir, output_files[i])
    gen_yolov5_format(data_dir, image_list, bbox_list, label_list, output_file)

