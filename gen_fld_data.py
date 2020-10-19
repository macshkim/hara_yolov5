import os
import pandas as pd
from shutil import copy
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


def gen_yolov5_format(data_df, images_file, label_dir):

    with open(images_file, 'w') as fwrite:
        for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
            image_file = os.path.join(data_dir, row['image_name'])
            fwrite.write(image_file + "\n")

            image = io.imread(image_file)
            image_h, image_w = image.shape[:2]

            (x, y, w, h) = norm_bb(image_w, image_h, row['x_1'], row['y_1'], row['x_2'], row['y_2'])
            class_label = int(row['clothes_type']) - 1
            image_name = row['image_name'].split('/')[1].split('.')[0]

            with open(os.path.join(label_dir, image_name + '.txt'), 'w') as fout:
                output_line = str(class_label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                fout.write(output_line + "\n")
            # copy(image_file, image_dir)


data_dir = '/Users/mackim/datasets/fld'
fld_data_file = os.path.join(data_dir, 'info/fld_info.csv')

fld_data_df = pd.read_csv(fld_data_file)

train_data_df = fld_data_df[fld_data_df['evaluation_status'] == 'train']
val_data_df = fld_data_df[fld_data_df['evaluation_status'] == 'val']
test_data_df = fld_data_df[fld_data_df['evaluation_status'] == 'test']

train_images_file = os.path.join(data_dir, 'train_fld.txt')
val_images_file = os.path.join(data_dir, 'val_fld.txt')
test_images_file = os.path.join(data_dir, 'test_fld.txt')
label_dir = os.path.join(data_dir, 'img')

print("Generating training data...")
gen_yolov5_format(train_data_df, train_images_file, label_dir)

print("\nGenerating validation data...")
gen_yolov5_format(val_data_df, val_images_file, label_dir)

print("\nGenerating testing data...")
gen_yolov5_format(val_data_df, test_images_file, label_dir)

