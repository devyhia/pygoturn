# necessary imports
import os
import time
import copy
from . import datasets
import argparse
from . import model
import torch
from torch.autograd import Variable
from torchvision import transforms
from .helper import ToTensor, Normalize, show_batch, scale_ratio, unscale_ratio
import torch.optim as optim
import numpy as np
from .helper import *
from .get_bbox import *
from torch.utils.data import Dataset
from matplotlib import animation, rc
from PIL import Image, ImageDraw
from glob import glob

use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='GOTURN Training')
parser.add_argument('-weights', '--model-weights', default='../saved_checkpoints/exp3/model_n_epoch_47_loss_2.696.pth', type=str, help='path to trained model')
parser.add_argument('-save', '--save-directory', default='', type=str, help='path to save directory')
parser.add_argument('-data', '--data-directory', default='../data/alov300/imagedata++/02-SurfaceCover/02-SurfaceCover_video00002', type=str, help='path to video frames')

class Tester(Dataset):
    """Test Dataset for Tester"""
    def __init__(self, root_dir, model_path, init_box=None, save_dir=None, ignore_gpu=False):
        self.root_dir = root_dir
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.model_path = model_path
        self.model = model.GoNet()

        if use_gpu:
            self.model = self.model.cuda()

        state = torch.load(model_path) if not ignore_gpu else torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state)

        frames = glob(root_dir)
        self.len = len(frames)-1
        frames = np.array(frames)
        frames.sort()
        self.x = []
        step = 5
        for i in range(0, self.len, step):
            self.x.append([frames[i], frames[i+step]])
        self.x = np.array(self.x)
        # code for previous rectange
        self.init_bbox = bbox_coordinates(self.x[0][0]) if init_box is None else init_box
        self.prev_rect = self.init_bbox


    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return self.transform(sample)

    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.prev_rect
        # Crop previous image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        crop_prev = CropPrev()
        # crop_curr = CropCurr()
        scale = Rescale((227,227))
        transform_prev = transforms.Compose([crop_prev, scale])
        # transform_curr = transforms.Compose([crop_curr, scale])
        prev_img = transform_prev({'image':prev, 'bb':prevbb})
        # Crop current image with height and width twice the prev bounding box height and width
        # Scale the cropped image to (227,227,3)
        curr_img = transform_prev({'image':curr, 'bb':prevbb})['image']
        sample = {'previmg': prev_img['image'], 'currimg': curr_img, 'currbb': prev_img['bb'] }
        return sample

    def get_rect(self, sample):
        x1, x2 = sample['previmg'], sample['currimg']
        x1 = x1[None,:,:,:]
        x2 = x2[None,:,:,:]
        x1,x2 = Variable(x1), Variable(x2)
        y = self.model(x1, x2)
        #print('y: ', y)
        bb = y.data.numpy().transpose((1,0))
        #print('Sample: ', sample)
        #print('Pred. BBox: ', bb)
        # Undo Normalize (from 0..10 to 0..227)
        bb = bb[:,0]
        bb = bb* unscale_ratio
        prevbb = self.prev_rect
        w = prevbb[2]-prevbb[0]
        h = prevbb[3]-prevbb[1]
        new_w = 2*w
        new_h = 2*h
        ww = 227
        hh = 227
        # BBox Width & Height Scaling
        bb = np.array([
            bb[0]*new_w/ww,
            bb[1]*new_h/hh,
            bb[2]*new_w/ww,
            bb[3]*new_h/hh
        ])

        left = prevbb[0]-w/2
        top = prevbb[1]-h/2
        #print('Unscaled BBox: ', bb)
        # uncrop
        bb = np.array([bb[0]+left, bb[1]+top, bb[2]+left, bb[3]+top])
        #print('Uncropped BBox: ', bb)
        return bb

    def animated_test(self):
        self.prev_rect = self.init_bbox

        fig, ax = plt.subplots()

        im = io.imread(self.x[0][1])
        self.anim_im = ax.imshow(im, animated=True)
        self.anim_idx = 1
        self.anim_fig = fig
        self.anim_ax = ax

        ani = animation.FuncAnimation(fig, self.test, interval=1, frames=10, blit=False)

        return ani

    def test(self, animated=True, *args):
        # show initial image with rectange
        i = self.anim_idx

        print('Testing frame # {}'.format(i))

        sample = self[i]
        bb = self.get_rect(sample)

        im = Image.open(self.x[i][1])
        draw = ImageDraw.Draw(im)

        cor = bb.tolist() # (x1,y1, x2,y2)
        line = (cor[0],cor[1],cor[0],cor[3])
        draw.line(line, fill="red", width=5)
        line = (cor[0],cor[1],cor[2],cor[1])
        draw.line(line, fill="red", width=5)
        line = (cor[0],cor[3],cor[2],cor[3])
        draw.line(line, fill="red", width=5)
        line = (cor[2],cor[1],cor[2],cor[3])
        draw.line(line, fill="red", width=5)
        # draw.rectangle(, outline='red')
        del draw

        if animated:
            self.anim_im.set_array(np.array(im))

        self.prev_rect = bb
        self.anim_idx += 1

        return im


def main():
    args = parser.parse_args()
    print(args)
    tester = Tester(args.data_directory, args.model_weights, args.save_directory)
    #tester.test()

if __name__ == "__main__":
    main()
