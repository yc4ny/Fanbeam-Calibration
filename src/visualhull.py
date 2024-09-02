import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils import c2w_transform

class VisualHull2D:
    """
    Args:
    cameras (list): list of cameras [fanbeam1, fanbeam2 ...]
    bbox2s (list): list of 2D bounding boxes [(x1,y1,x2,y2), ...] # y값은 필요 없으나, 3d 만들 때 사용 
    phantom (np.array): 3D phantom [num_beads, 3] # 3d 구슬 위치, 시각화에 사용

    복셀 아니고 2d 그리드
    """
    def __init__(self, cameras, bbox2ds, phantom):

        self.voxel_size = (600, 600) # x, z 
        # self.voxel = self.get_grid()
        self.voxel = np.zeros(self.voxel_size)
        self.cameras = cameras
        self.bbox2ds = bbox2ds
        self.phantom = phantom
        self.lines = []
        self.points = []
        self.box_viz = []
        self.num_cam = len(cameras)
        assert len(cameras) == len(bbox2ds), "cameras and bbox2ds should have same length"

    def get_grid(self):
        voxel_size = self.voxel_size
        x = np.arange(-voxel_size[0]//2, voxel_size[0]//2)
        z = np.arange(-voxel_size[1]//2, voxel_size[1]//2)
        grid = np.meshgrid(x, z)
        return grid
        
    def run(self):
        for i in range(len(self.cameras)):
            cam = self.cameras[i]
            bbox = self.bbox2ds[i]
            cam_center = cam.cam_center
            R = cam.R
            DSD = cam.DSD
            Cx = cam.Cx
            p_world1 = c2w_transform(cam, bbox[0])
            p_world2 = c2w_transform(cam, bbox[2])
            px1, pz1 = p_world1[0], p_world1[2]
            px2, pz2 = p_world2[0], p_world2[2]
            self.lines.append([(px1, pz1), (cam_center[0], cam_center[2])])
            self.lines.append([(px2, pz2), (cam_center[0], cam_center[2])])

            # projection and update voxel
            x = np.arange(self.voxel_size[0]) - self.voxel_size[0]//2 # [:, None]
            z = np.arange(self.voxel_size[1]) - self.voxel_size[1]//2  
            xz = np.meshgrid(x, z)  
            K = np.array([[DSD, 0, Cx],
                            [0, DSD, cam.H/2],
                            [0, 0, 1]])
            RT = np.hstack((R, cam.T[:, None]))
            # (x, 0, z, 1) -> (u)
            p3d = np.array([xz[0], np.zeros_like(xz[0]), xz[1], np.ones_like(xz[0])]) # (4,600,600)
            p2d = np.dot(K, np.dot(RT, p3d.reshape(4, -1)))
            p2d = p2d / p2d[2]
            u = p2d[0].reshape(self.voxel_size)
            mask = (u < bbox[0]) | (u > bbox[2])
            self.voxel[mask] = 1
        try:
            self.fit_bbox()
        except:
            pass
        self.triangulate_box_center()

    def viz(self):
        for cam in self.cameras:
            cam_dir = cam.R.T @ np.array([0, 0, 1])
            plt.scatter(cam.cam_center[0], cam.cam_center[2], c='r')
            plt.arrow(cam.cam_center[0], cam.cam_center[2], cam_dir[0], cam_dir[2], head_width=5, head_length=200, fc='k', ec='k')
        # draw voxel box 
        b0 = (-self.voxel_size[0]//2, -self.voxel_size[1]//2)
        b1 = (-self.voxel_size[0]//2, self.voxel_size[1]//2)
        b2 = (self.voxel_size[0]//2, self.voxel_size[1]//2)
        b3 = (self.voxel_size[0]//2, -self.voxel_size[1]//2)
        plt.plot([b0[0], b1[0]], [b0[1], b1[1]], c='b')
        plt.plot([b1[0], b2[0]], [b1[1], b2[1]], c='b')
        plt.plot([b2[0], b3[0]], [b2[1], b3[1]], c='b')
        plt.plot([b3[0], b0[0]], [b3[1], b0[1]], c='b')

        color = 0
        for line in self.lines:
            if color < 2*self.num_cam:
                plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], c='r')
            else:
                plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], c='g')
            color += 1
        for point in self.points:
            plt.scatter(point[0], point[2], c='g')
        for p in self.phantom:
            plt.scatter(p[0], p[2], c='b')
        for box in self.box_viz:
            plt.plot([box[0][0], box[1][0]], [box[0][1], box[1][1]], c='y')
            plt.plot([box[1][0], box[2][0]], [box[1][1], box[2][1]], c='y')
            plt.plot([box[2][0], box[3][0]], [box[2][1], box[3][1]], c='y')
            plt.plot([box[3][0], box[0][0]], [box[3][1], box[0][1]], c='y')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('FanBeam CT')
        plt.xlim(-1200, 1200)
        plt.ylim(-1200, 1200)
        move = self.voxel_size[0]//2
        plt.imshow(self.voxel, cmap='gray', extent=[-move, self.voxel.shape[1]-move, self.voxel.shape[0]-move, -move])

        save_dir = "./debug"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'visual_hull.png'))
        return plt.show()

    def triangulate_points(self, cameras, points, viz_rays=False):
        """
        Args:
        cameras (list): list of cameras [fanbeam1, fanbeam2 ...]
        points (list): list of 2D points [(x1,y1), (x2,y2), ...]

        """
        # fan beam 이므로 y는 전부 H/2 로 .. 
        points = [(i[0], 400) for i in points]

        Pmat = []
        for i in range(len(cameras)):
            cam = cameras[i]
            f = cam.DSD
            Cx = cam.Cx
            Cy = cam.H/2
            K = np.array([[f, 0, Cx],
                            [0, f, Cy],
                            [0, 0, 1]])
            R = cam.R
            T = cam.T
            P = np.dot(K, np.hstack((R, T[:, None])))
            Pmat.append(P)

            if viz_rays:
                x, _ = points[i]
                p_world = c2w_transform(cam, x)
                cam_center = cam.cam_center
                self.lines.append([(p_world[0], p_world[2]), (cam_center[0], cam_center[2])])

        points = np.array(points)
        points = np.hstack((points, np.ones((points.shape[0], 1))))

        A = np.zeros((2*len(cameras), 4))
        for i in range(len(cameras)):
            A[2*i, :] = points[i, 0] * Pmat[i][2, :] - Pmat[i][0, :]
            A[2*i+1, :] = points[i, 1] * Pmat[i][2, :] - Pmat[i][1, :]

        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[3]
        self.points.append(X)
    
    def fit_bbox(self):
        """
        fit bbox to polygon 
        https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
        """
        polygon = self.voxel 
        polygon = 1 - polygon
        # find contour ... TODO visual hull 계산할 때 미리 저장해두고 사용해도 될 듯? ...굳이? 
        contours, _ = cv2.findContours(polygon.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = box - self.voxel_size[0]//2
        self.box_viz.append(box)
    
    def RANSAC(self):
        # TODO model output RANSAC 으로 걸러주기! 
        raise NotImplementedError
    
    def triangulate_box_center(self):
        # bbox 중심점 triangulation 
        bbox_centers = []
        for bbox in self.bbox2ds:
            center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
            bbox_centers.append(center)
        self.triangulate_points(self.cameras, bbox_centers, viz_rays=False)


class VisualHull_RANSAC(VisualHull2D):

    def __init__(self, cameras, bbox2ds, phantom, savedir='./debug/ransac'):
        super(VisualHull_RANSAC, self).__init__(cameras, bbox2ds, phantom)
        self.max_iter = 100 
        self.threshold = 0.1 # 적당히 수정 필요
        self.cameras_all = cameras
        self.bbox2ds_all = bbox2ds
        self.debug = False
        self.savedir = savedir

    def run_RANSAC(self):
        iter = 0
        best_inliers = []
        best_err = 1e10
        while iter < self.max_iter:
            print('iter:', iter, 'best_inliers', best_inliers)
            iter += 1
            # num_samples = np.random.randint(2, self.num_cam) # 수정
            num_samples = np.random.randint(np.max([len(best_inliers), 2]), self.num_cam)
            selected_index = np.random.choice(self.num_cam, num_samples, replace=False) # 샘플링 개선 가능 
            selected_index = np.sort(selected_index)
            cameras = [self.cameras_all[i] for i in selected_index]
            bbox2ds = [self.bbox2ds_all[i] for i in selected_index]
            # update selected cameras and bbox2ds
            self.cameras = cameras
            self.bbox2ds = bbox2ds
            # compute visual hull 
            self.voxel = np.zeros(self.voxel_size)
            self._run()
            if self.voxel.sum() == self.voxel_size[0]*self.voxel_size[1]:
                continue # no intersection
            err_ls = self.calc_error() # 전체 카메라에 대해 reprojection error 계산
            inliers = [i for i, err in enumerate(err_ls) if err < self.threshold] 
            if len(inliers) > len(best_inliers) and len(selected_index) >= len(best_inliers):
                best_inliers = selected_index
                best_err = np.mean(err_ls)
            elif len(inliers) == len(best_inliers) and np.mean(err_ls) < best_err and len(selected_index) >= len(best_inliers):
                best_inliers = selected_index
                best_err = np.mean(err_ls)

        print('inliers :', best_inliers)
        self.cameras = [self.cameras_all[i] for i in best_inliers]
        self.bbox2ds = [self.bbox2ds_all[i] for i in best_inliers]
        self.lines = []
        self.box_viz = [] 
        self.voxel = np.zeros(self.voxel_size)
        self.run()

    def calc_error(self):
        # porject reconstructed voxel to each camera and calculate length 
        points = self.voxel_to_points() 
        err_ls = [] 
        for i in range(len(self.cameras_all)):
            cam = self.cameras_all[i]
            f = cam.DSD
            Cx = cam.Cx
            Cy = cam.H/2
            K = np.array([[f, 0, Cx],
                            [0, f, Cy],
                            [0, 0, 1]])
            R = cam.R
            T = cam.T
            P = np.dot(K, np.hstack((R, T[:, None])))
            p2d = np.dot(P, points.T)
            p2d = p2d / p2d[2]
            p2d = p2d[:2]

            box_length = np.linalg.norm(self.bbox2ds_all[i][0] - self.bbox2ds_all[i][2])
            p2d_min = np.min(p2d[0]).clip(self.bbox2ds_all[i][0], self.bbox2ds_all[i][2])
            p2d_max = np.max(p2d[0]).clip(self.bbox2ds_all[i][0], self.bbox2ds_all[i][2])
            projected_length = np.linalg.norm(p2d_min - p2d_max) 
            err = np.abs(box_length - projected_length) / cam.W
            err_ls.append(err)
        return err_ls

    def voxel_to_points(self):
        # self.voxel -> points (x, 0, z, 1)
        i, j = np.where(self.voxel == 0)
        points = np.column_stack((i - self.voxel_size[0]//2, np.zeros_like(i), j - self.voxel_size[1]//2, np.ones_like(i)))
        return points
    
    def _run(self):
        for i in range(len(self.cameras)):
            cam = self.cameras[i]
            bbox = self.bbox2ds[i]
            R = cam.R
            DSD = cam.DSD
            Cx = cam.Cx

            x = np.arange(self.voxel_size[0]) - self.voxel_size[0]//2 # [:, None]
            z = np.arange(self.voxel_size[1]) - self.voxel_size[1]//2  
            xz = np.meshgrid(x, z)  
            K = np.array([[DSD, 0, Cx],
                            [0, DSD, cam.H/2],
                            [0, 0, 1]])
            RT = np.hstack((R, cam.T[:, None]))
            # (x, 0, z, 1) -> (u)
            p3d = np.array([xz[0], np.zeros_like(xz[0]), xz[1], np.ones_like(xz[0])]) # (4,600,600)
            p2d = np.dot(K, np.dot(RT, p3d.reshape(4, -1)))
            p2d = p2d / p2d[2]
            u = p2d[0].reshape(self.voxel_size)
            mask = (u < bbox[0]) | (u > bbox[2])
            self.voxel[mask] = 1
        if self.voxel.sum() == self.voxel_size[0]*self.voxel_size[1]:
            pass
        else:
            self.fit_bbox()
