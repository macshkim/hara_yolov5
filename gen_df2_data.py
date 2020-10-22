import os
from skimage import io
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import json


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


def gen_yolov5_format(data_dir, image_list, anno_dir, output_file):

    with open(output_file, 'w') as fwrite:
        for index in tqdm(range(len(image_list))):
            image_path = image_list[index].strip()
            image_file = os.path.join(data_dir, image_path)
            fwrite.write(image_file + "\n")

            image = io.imread(image_file)
            image_h, image_w = image.shape[:2]

            anno_file = os.path.join(anno_dir, image_path.split('.')[0] + '.json')
            with open(anno_file) as f:
                anno_dict = json.load(f)

            image_name = image_file.split('.')[0] + '.txt'
            with open(image_name, 'w') as fout:
                for key in anno_dict:
                    if key.startswith('item'):
                        bbox_xy = anno_dict[key]['bounding_box']
                        (x, y, w, h) = norm_bb(image_w, image_h, int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]),
                                               int(bbox_xy[3]))
                        class_label = int(anno_dict[key]['category_id']) - 1
                        output_line = str(class_label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                        fout.write(output_line + "\n")


data_dir = '/Users/mackim/datasets/deepfashion2'

for data_type in ['train', 'validation']:
    image_dir = os.path.join(data_dir, data_type, 'image')
    anno_dir = os.path.join(data_dir, data_type, 'annos')

    image_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    output_file = os.path.join(data_dir, data_type + '_df2.txt')
    gen_yolov5_format(image_dir, image_files, anno_dir, output_file)
