from PIL import Image
import os
import glob


data_dir = './Folder/Data/Raw/tiny-imagenet-200'


wnids_path = os.path.join(data_dir, 'wnids.txt')
with open(wnids_path, 'r') as f:
    wnids = [line.strip() for line in f.readlines()]
class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}


train_dir = os.path.join(data_dir, 'train')
train_list_path = os.path.join(data_dir, 'train_list.txt')
with open(train_list_path, 'w') as f:
    for wnid in wnids:
        class_dir = os.path.join(train_dir, wnid, 'images')
        for img_path in glob.glob(os.path.join(class_dir, '*.JPEG')):
            img_name = os.path.relpath(img_path, data_dir)  
            label = class_to_idx[wnid]
            f.write(f'{img_name} {label}\n')


val_dir = os.path.join(data_dir, 'val')
val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
val_list_path = os.path.join(data_dir, 'val_list.txt')
with open(val_annotations_path, 'r') as f_val, open(val_list_path, 'w') as f_out:
    for line in f_val.readlines():
        img_name, wnid, _, _, _, _ = line.strip().split('\t')
        label = class_to_idx[wnid]
        img_path = os.path.join(val_dir, 'images', img_name)  
        img_rel_path = os.path.relpath(img_path, data_dir)  
        f_out.write(f'{img_rel_path} {label}\n')