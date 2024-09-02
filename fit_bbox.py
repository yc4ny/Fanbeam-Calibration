import os 
import numpy as np 
from src.visualhull import VisualHull2D
from src.camera import FanBeam
from src.constant import phantom 
from src.utils import load_cameras

"""
visual hull bbox fitting 데모 
"""

# beads optimize 안 한 결과
calib_results = np.array([[  86.8187,  705.5422, -182.7358,  690.7780],
    [  92.4040,  742.0136, -149.5497,  757.7974],
    [  83.3328,  708.5674, -114.4536,  760.5787],
    [  65.6873,  648.1901,  -78.9733,  726.6708],
    [  38.5659,  615.3066,  -43.8189,  717.4235],
    [   9.2079,  610.2416,   -8.5876,  726.1641],
    [ -25.2413,  617.2256,   26.0271,  729.0135],
    [ -51.8704,  628.0492,   61.2695,  723.6698],
    [ -77.4625,  658.5148,   96.0234,  721.1707]])

fanbeams = []
for i in range(len(calib_results)):
    T = np.array([calib_results[i, 0], 0, calib_results[i, 1]])
    theta = calib_results[i, 2] * 0.01
    DSD = calib_results[i, 3]
    fanbeam = FanBeam(DSD, theta, T, image_path=None, beads_2d=None)
    fanbeams.append(fanbeam)

# optimize 한 결과 load
fanbeams = load_cameras()
beads_path = './results/beads_optimized.npy'
phantom = np.load(beads_path)

bboxes = []
# 이미지 사이즈, 카메라 시각화 
for i in range(len(calib_results)):
    bbox = (0, 0, 640, 800) # (x1, y1, x2, y2)
    bboxes.append(bbox)

vh = VisualHull2D(fanbeams, bboxes, phantom) # full image ray
vh.run() # 카메라 visualization
# plt = vh.viz()

# 원통 실린더 
# TODO 모델 output 사용하고 RANSAC 추가! 
bboxes = [(298, 0, 512, 800), (303, 0, 526, 800), (293, 0, 526, 800), (272, 0, 515, 800), (236, 0, 492, 800), (200, 0, 459, 800), (161, 0, 418, 800),(133, 0, 383, 800),(112, 0, 354, 800)] # 직접 좌표 찍어서 확인 
fanbeams = [fanbeams[0], fanbeams[1], fanbeams[2], fanbeams[3], fanbeams[4], fanbeams[5], fanbeams[6], fanbeams[7], fanbeams[8]]
vh.bbox2ds = bboxes
vh.cameras = fanbeams    
vh.run()
plt = vh.viz()