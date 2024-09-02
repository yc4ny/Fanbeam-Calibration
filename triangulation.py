import os 
import numpy as np 
from src.visualhull import VisualHull2D
from src.camera import FanBeam
from src.constant import phantom 
from src.utils import load_cameras

"""
trianuglation 데모 
svd solution 
"""

# beads optimize 안 한 결과! 
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

# # optimize 한 결과 load
# fanbeams = load_cameras()
# beads_path = './results/beads_optimized.npy'
# phantom = np.load(beads_path)

bboxes = []
# 이미지 사이즈, 카메라 시각화 
for i in range(len(calib_results)):
    bbox = (0, 0, 640, 800) # (x1, y1, x2, y2)
    bboxes.append(bbox)

vh = VisualHull2D(fanbeams, bboxes, phantom) # full image ray
# vh.run()
# phantom_bead_top 
vh.triangulate_points(fanbeams, [(299 ,400), (300 ,400), (298 ,400), (302, 400), (306 ,400), (311 ,400), (317 ,400), (326 ,400), (330 ,400)], viz_rays=True) 
# tray
vh.triangulate_points(fanbeams, [(555,400), (412,400), (276,400), (128,400), (43,400), (19,400), (28,400), (75,400), (128,400)], viz_rays=False)
vh.triangulate_points([fanbeams[1], fanbeams[2], fanbeams[3]], [(446, 400), (478, 400), (541, 400)], viz_rays=False)
vh.triangulate_points([fanbeams[4], fanbeams[2]], [(92, 400), (508, 400)], viz_rays=False)
vh.triangulate_points([fanbeams[3], fanbeams[6], fanbeams[7]], [(631, 400), (605, 400), (466, 400)], viz_rays=False)

plt = vh.viz()

