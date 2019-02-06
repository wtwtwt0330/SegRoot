from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.morphology import dilation
import pickle
import numpy as np
import argparse
from dataloader import pad_pair_256

parser = argparse.ArgumentParser()  
parser.add_argument(
    "--dilate",
    default=0,
    type=int,
    help="dilation degree of masks")

args = parser.parse_args()
data_dir = Path('../data/data_raw')
mask_dir = Path('../data/masks')
mask_dir.mkdir(exist_ok=True, parents=True)

imgs = sorted(list(data_dir.glob('*Untitled.jpg')))
print('original images count : ', len(imgs))
masks = sorted(list(data_dir.glob('*Untitled-mask.jpg')))
print('original masks count : ', len(masks))

# generate binary masks for every annotated images
for m in masks:
    mask = io.imread(m.as_posix(), as_gray=True)
    # binarize
    mask[mask > 0.5 ] = 1.0
    mask[mask <= 0.5] = 0.0
    for i in range(args.dilate):
        mask = dilation(mask)
    print('binary masks dilated !!!')
    plt.imsave((mask_dir / m.parts[-1]).as_posix(), mask, cmap='gray')
print('binary masks generated !!!')

# save idx info in a dictionary
info_dict = {k: v.parts[-1] for k, v in enumerate(imgs)}
with open('../data/info.pkl', 'wb') as handle:
    pickle.dump(info_dict, handle)
print('index info saved!!!')

# crop the padded image to generate 256*256 subimages
new_masks = sorted(list(mask_dir.glob('*Untitled-mask.jpg')))
print('new_mask length : ',len(new_masks))

subimg_path = Path('../data/subimg')
subimg_path.mkdir(exist_ok=True, parents=True)
submask_path = Path('../data/submask')
submask_path.mkdir(exist_ok=True, parents=True)

for idx, (mask_path, img_path) in enumerate(zip(new_masks, imgs)):
    mask = Image.open(mask_path)
    img = Image.open(img_path)
    new_img, new_mask = pad_pair_256(img, mask)
    new_img, new_mask = np.array(new_img), np.array(new_mask)
    # padded shape (2560, 2304)
    w, h, _ = new_img.shape
    for i in range(int(w/256)):
        for j in range(int(h/256)):
            subimg = new_img[i*256:(i+1)*256, j*256:(j+1)*256, :]
            subimg_fn = '{}/{}-{}-{}.png'.format(
                Path('../data/subimg').as_posix(), idx, i, j)
            plt.imsave(subimg_fn, subimg)
            submask_fn = '{}/{}-{}-{}.png'.format(
                Path('../data/submask').as_posix(), idx, i, j)
            submask = new_mask[i*256:(i+1)*256, j*256:(j+1)*256]
            plt.imsave(submask_fn, submask, cmap='gray')
    print('No.{} image & mask cropped!!!'.format(idx))
