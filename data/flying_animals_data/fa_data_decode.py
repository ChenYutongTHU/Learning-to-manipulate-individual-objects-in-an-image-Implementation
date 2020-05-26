import numpy as np 
import imageio
import os
import argparse


rm_bg = [34,80,71,72,38,28,85,88,93,86,75,70,42,30,24,64,65,90,82,25,
29,21,40,4,51,79,33,68,83,91,59,6,87,45,94,99,
23,78,36,19,77,39,62,52,81,56,98,43] 
rm_animals = [2,3,7,8,11,20,4,6,9,19]


parser = argparse.ArgumentParser(description='decode raw images into .npz')
parser.add_argument('--data_dir', type=str, default='./data', help='directory of the data  data_dir/foregrounds,backgrounds,masks')

args = parser.parse_args()

data_dir = args.data_dir

data = {'background':[], 'foreground':[], 'mask':[]}
test_data = {'background':[], 'foreground':[], 'mask':[]}

for i in range(241):
    fg = imageio.imread(os.path.join(data_dir, 'foregrounds', '{}.png'.format(i))) 
    data['foreground'].append(fg)

    mask = imageio.imread(os.path.join(data_dir, 'masks', '{}.png'.format(i)))
    mask = mask.astype(np.bool) 
    data['mask'].append(mask)

    if not i in rm_animals:
        test_data['foreground'].append(fg) 
        test_data['mask'].append(mask)


for i in range(101):
    bg = imageio.imread(os.path.join(data_dir, 'backgrounds', '{}.png'.format(i)))
    data['background'].append(bg)

    if not i in rm_bg:
        test_data['background'].append(bg)



np.savez('img_data',background=data['background'], foreground=data['foreground'], mask=data['mask'])
np.savez('img_data_test',background=test_data['background'], foreground=test_data['foreground'], mask=test_data['mask'])