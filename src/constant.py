import numpy as np

"""
    z               
  /
 /
/
------------->x (u)
|
|     이미지 평면 
|
y (v)
"""

# 구슬 디텍션 결과 (num_cameras, num_beads, 3)
circle_path = "./results/circles.npy" 
circles = np.load(circle_path)   

phantom = np.array([[6.9, 111.7, -75.0],
                    [61.8, 92.4, -62.1],
                    [98.5, 50.4, -49.8],
                    [110.5, -6.0, -38.5],
                    [92.6, -60.7, -26.7],
                    [50.2, -98.1, -14.3],
                    [-5.0, -109.9, -1.1],
                    [-59.4, -92.7, 12.2],
                    [-98.4, -50.2, 25.0],
                    [-110.8, 5.8, 38.2],
                    [-93.8, 59.7, 51.2],
                    [-51.7, 98.3, 63.9],
                    [3.8, 110.4, 76.9]])
phantom[:, 1], phantom[:, 2] = phantom[:, 2], phantom[:, 1].copy()
phantom[:, 0] = -phantom[:, 0]

# z 좌표는 이미지 bead y 좌표로 대체 # 시각화 편의상
# fanbeam 이므로 이미지 v 축으로는 독립 (Y축 방향)
# loss 값에 영향을 안줘요

phantom_FB = np.array([[6.9, 111.7, circles[0][0][1]],
                        [61.8, 92.4, circles[0][1][1]],
                        [98.5, 50.4, circles[0][2][1]],
                        [110.5, -6.0, circles[0][3][1]],
                        [92.6, -60.7, circles[0][4][1]],
                        [50.2, -98.1, circles[0][5][1]],
                        [-5.0, -109.9, circles[0][6][1]],
                        [-59.4, -92.7, circles[0][7][1]],
                        [-98.4, -50.2, circles[0][8][1]],
                        [-110.8, 5.8, circles[0][9][1]],
                        [-93.8, 59.7, circles[0][10][1]],
                        [-51.7, 98.3, circles[0][11][1]],
                        [3.8, 110.4, circles[0][12][1]]])
# y, z 축 바꾸기 
phantom_FB[:, 1], phantom_FB[:, 2] = phantom_FB[:, 2], phantom_FB[:, 1].copy()
# x 축 flip
phantom_FB[:, 0] = -phantom_FB[:, 0]


 