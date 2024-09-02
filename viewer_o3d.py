import os 
import numpy as np
import open3d as o3d
from src.draw_utils import *
from src.camera import FanBeam
from src.utils import load_cameras

voxel_size = np.array([[800],[800],[800]])
cube1 = plot_cube(np.zeros((3,1)), voxel_size)

fanbeams = load_cameras()

# 시각화 위해 임의로 조정 ... 나중에 주석 처리
for i in range(len(fanbeams)):
    fanbeams[i].DSD = fanbeams[i].DSD + 1000 

image = draw_image(fanbeams[0])
for i in range(1, len(fanbeams)): 
    image += draw_image(fanbeams[i])

cam = draw_cam(fanbeams[0])
for i in range(1, len(fanbeams)):
    cam += draw_cam(fanbeams[i])

beads_path = './results/beads_optimized.npy'
beads = np.load(beads_path)

# beads y 좌표가 이미지 좌표니깐 -Cy 
beads[:, 1] = beads[:, 1] - fanbeams[0].Cy
beads = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(beads))
beads.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([cube1, image, beads, cam])