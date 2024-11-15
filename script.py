import os


ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'img')
ANN_DIR = os.path.join(DATA_DIR, 'ann')


for idx, filename in enumerate(os.listdir(IMG_DIR), start=1):
    old_img_path = os.path.join(IMG_DIR, filename)
    old_ann_path = os.path.join(ANN_DIR, f'{filename}.json')
    new_img_path = os.path.join(IMG_DIR, f'{idx}.jpg')
    new_ann_path = os.path.join(ANN_DIR, f'{idx}.jpg.json')
    os.rename(old_img_path, new_img_path)
    os.rename(old_ann_path, new_ann_path)
