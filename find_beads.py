import os 
import cv2
import numpy as np

"""
캘리브레이션 팬텀 구슬 위치 찾기 
TODO: 잘 못 찾는 경우 수정 필요 (image processing...)
지금은 beads 중심 위치만 사용해 캘리브레이션 하는데, 타원 피팅으로  고도화 필요 
"""

h_min = 300 
h_max = 510
n_circles = 13

data_path = "./data/images/"
img_list = os.listdir(data_path)
img_list.sort()

ls = []
for img_name in img_list:
    img = cv2.imread(data_path + img_name, cv2.IMREAD_GRAYSCALE)
    img = img[h_min:h_max, :]

    # 파라미터 설정 중요 
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 5, param1=70, param2=7, minRadius=2, maxRadius=4)

    for i in circles[0]:
        cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 3)

    save_dir = './debug/HoughCircles/'
    os.makedirs(save_dir, exist_ok=True)
    img_name = img_name.split('/')[-1]
    cv2.imwrite(os.path.join(save_dir, img_name), img)

    if n_circles != len(circles[0]):
        print(f"Error: number of circles detected is not equal to {n_circles}")
        print(f"Number of circles detected: {len(circles[0])}")
        break
    
    # h가 낮은순으로 정렬
    circles = circles[0][circles[0][:, 1].argsort()]
    # 원본 이미지 좌표
    circles = circles[:, 0:2] + [0, h_min]
    ls.append(circles)

ls = np.array(ls) # (num_images, num_beads, 2) # (x,y) 
save_dir = './results/'
np.save(os.path.join(save_dir, 'circles.npy'), ls)
print('done')