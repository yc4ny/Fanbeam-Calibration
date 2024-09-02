import os 
import numpy as np 
from src.camera import FanBeam
from src.constant import circles

def load_cameras_init():

    cameras = []
    DSD = 1100
    theta = 0
    T = np.array([0, 1000, 0])
    image_dir = 'data/images'
    image_ls = os.listdir(image_dir)
    for i in range(len(image_ls)):
        beads_2d = circles[i]
        image_path = os.path.join(image_dir, image_ls[i])
        cam = FanBeam(DSD, theta, T, image_path, beads_2d)
        cameras.append(cam)       

    return cameras

def load_cameras():
    # load cameras from calibration results 
    cameras = []
    try:
        calib_results = np.load('results/calibration_results.npy')
    except:
        raise FileNotFoundError('calibration results not found')
    image_dir = 'data/images'
    image_ls = os.listdir(image_dir)
    for i in range(calib_results.shape[0]):
        DSD = calib_results[i, 3]
        theta = calib_results[i, 2] * 0.01 # TODO 스케일링 따로 정리 
        T = np.array([calib_results[i, 0], 0, calib_results[i, 1]])
        image_path = os.path.join(image_dir, image_ls[i])
        beads_2d = circles[i]
        cam = FanBeam(DSD, theta, T, image_path, beads_2d)
        cameras.append(cam)

    return cameras

def c2w_transform(cam, u):
    # apply camera to world transformation
    # u: 이미지 좌표 -> world 좌표 (X, 0, Z)

    f = cam.DSD
    Cx = cam.Cx
    p3d = np.array([u - Cx, 0, f])
    w2c = np.zeros((4, 4))
    w2c[:3, :3] = cam.R
    w2c[:3, 3] = cam.T
    w2c[3, 3] = 1
    c2w = np.linalg.inv(w2c)
    p_world = np.dot(c2w, np.append(p3d, 1))
    p_world = p_world[:3]

    return p_world