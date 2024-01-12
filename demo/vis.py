import sys
import cv2

from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
import shutil
from IPython import embed


sys.path.append(os.getcwd())
from model.GCN_conv import adj_mx_from_skeleton
from model.trans import HTNet
from common.camera import *
from common.h36m_dataset import Human36mDataset
from common.camera import camera_to_world
from common.opt import opts
opt = opts().parse()

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dataset_path = './dataset/data_3d_h36m.npz'
dataset = Human36mDataset(dataset_path, opt)
adj = adj_mx_from_skeleton(dataset.skeleton())

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img



def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)




def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(figure_path, output_dir, file_name):
    # Genarate 2D pose 
    keypoints, scores = hrnet_pose(figure_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    ## Reload
    previous_dir = './ckpt/cpn'
    model = HTNet(opt, adj).cuda()
    model_dict = model.state_dict()
    model_path = sorted(glob.glob(os.path.join(previous_dir, '*.pth')))[0]
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)
    model.eval()

    ## 3D
    img = cv2.imread(figure_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape
    input_2D_no = keypoints[:,0,:,:]

    joints_left =  [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[ :, :, 0] *= -1
    input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

    input_2D = input_2D[np.newaxis, :, :, :, :]

    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

    N = input_2D.size(0)

    ## estimation
    output_3D_non_flip = model(input_2D[:, 0])
    output_3D_flip     = model(input_2D[:, 1])

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    output_3D = output_3D[0:, opt.pad].unsqueeze(1)
    output_3D[:, :, 0, :] = 0
    post_out = output_3D[0, 0].cpu().detach().numpy()

    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)
    post_out[:, 2] -= np.min(post_out[:, 2])

    input_2D_no = input_2D_no[opt.pad]

    ## 2D
    image = show2Dpose(input_2D_no, copy.deepcopy(img))


    ## 3D
    fig = plt.figure( figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05)
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose( post_out, ax)


    output_dir_3D = output_dir +'pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    plt.savefig(output_dir_3D  + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')




    ## all
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    for i in range(len(image_3d_dir)):
        image_2d = image
        image_3d = plt.imread(image_3d_dir[i])
        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]
        ## show
        font_size = 12
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Pose", fontsize = font_size)

        ## save
        output_dir_pose = output_dir 
        plt.savefig(output_dir + file_name + '_pose.png', dpi=200, bbox_inches = 'tight')
    
    shutil.rmtree("./demo/output/pose3D")
    
    

if __name__ == "__main__":
    items = os.listdir('./demo/figure/')
    print(items)
    for i, file_name in enumerate(items): 
        print("Gnenerate Pose For " + file_name)
        figure_path = './demo/figure/' + file_name
        output_dir = './demo/output/'
        get_pose3D(figure_path, output_dir, file_name[:-4])



