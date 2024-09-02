import os 
import numpy as np
import torch 
from matplotlib import pyplot as plt
from src.camera import FanBeam
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
TODO: elipse fitting, 
팬텀 이미지가 N set 있을 경우, 각 set에 6dof로 파라미터 추가 필요 
MBIR 구현해보고 나온 calibration 결과 사용 
SSTLab 에서 recon해 준 voxel과 alignment 필요 할 듯 
"""

class FanbeamCalibrator: 
    def __init__(self, beads_3d, fanbeams, optim_f=False, optim_beads=False, n_iter=5000, save_dir="./debug"):
        self.beads_3d = beads_3d
        self.fanbeams = fanbeams
        self.optim_f = optim_f
        self.optim_beads = optim_beads
        self.num_beads = beads_3d.shape[0]
        self.num_cameras = len(fanbeams) 
        self.save_dir = save_dir
        self.n_iter = n_iter
        self.f_init = 1100.
        self.lr = 0.1
        self.params = torch.tensor([0., 1000., 0.]).repeat(self.num_cameras, 1) # Tx, Tz, theta 3자유도 
        self.params.requires_grad = True
        self.params_f = torch.tensor([self.f_init]*self.num_cameras, requires_grad=False) 
        self.params_beads = torch.from_numpy(beads_3d)
        self.params_beads.requires_grad = False
        if optim_f:
            self.params_f.requires_grad = True
        if optim_beads:
            self.params_beads.requires_grad = True
        self.optimizer = torch.optim.Adam([self.params, self.params_f, self.params_beads], lr=self.lr)
        self.save_fig = False
        self.losses = []

    def run(self):
        self.iter = 0
        for i in range(self.n_iter):
            self.optimizer.zero_grad()
            loss_all = 0
            for j in range(self.num_cameras):
                loss = self.cost(self.params[j], self.params_f[j], self.params_beads, self.fanbeams[j].beads_2d)
                loss_all += loss
            loss_all = loss_all / (self.num_cameras) #*self.num_beads 
            loss_all.backward()
            self.optimizer.step()
            self.losses.append(loss_all.item())
            print('iter:', i, 'loss:', loss_all.item())
            if self.save_fig:
                if i % 100 == 0:
                    self.update_cameras()
                    self.viz()
            self.iter += 1
        self.update_cameras()

    def update_cameras(self):
        for i in range(self.num_cameras):
            # self.fanbeams[i].T = np.array([self.params[i][0].item(), 0, self.params[i][1].item()])
            # self.fanbeams[i].theta = self.params[i][2].item() * 0.01
            # self.fanbeams[i].DSD = self.params_f[i].item()
            # self.fanbeams[i].R = self.fanbeams[i].get_rot_mat(self.fanbeams[i].theta)
            # self.fanbeams[i].cam_center = - np.dot(self.fanbeams[i].R.T, self.fanbeams[i].T)
            self.fanbeams[i] = FanBeam(self.params_f[i].item(), self.params[i][2].item() * 0.01, np.array([self.params[i][0].item(), 0, self.params[i][1].item()]), self.fanbeams[i].image_path, self.fanbeams[i].beads_2d)

    def cost(self, params, f, beads_3d, beads_2d):

        """
        프로젝션
        p2d = K @ R|T @ p3d  

        u    DSD 0 Cx      cos 0 sin | Tx     X
        v  = 0 DSD Cy  @   0  1  0   | 0   @  Y
        1    0  0  1      -sin 0 cos | Tz     Z
                                              1             
        """
        Tx, Tz, theta = params
        theta = theta * 0.01
        X = beads_3d[:, 0]
        Z = beads_3d[:, 2]
        # x = circles[index][:, 0]
        # x = torch.tensor(x)
        x = beads_2d[:, 0]
        x = torch.tensor(x)
        X_ = X*torch.cos(theta) + Z*torch.sin(theta) + Tx # X' : p3d_X in camera coordinate
        Z_ = Z*torch.cos(theta) - X*torch.sin(theta) + Tz # Z' : p3d_Z in camera coordinate
        # proj = (Cx + (f*u)/v) # for visualization
        loss = torch.nn.functional.mse_loss(x, (320 + (f*X_)/Z_))
        # TODO 행렬로 보기좋게 , LPB model
        # TODO camera.project_point 함수로 
        # TODO Cx 도 변수로 

        return loss
 
    def PNP_initialization(self):
        raise NotImplementedError
    
    def viz(self):
        beads = self.params_beads.detach().numpy()
        plt.clf()
        plt.scatter(beads[:, 0], beads[:, 2], c='r')
        for i in range(self.num_cameras):
            camera = self.fanbeams[i]
            cam_center = camera.cam_center
            R = camera.R
            cam_dir = np.dot(R.T, np.array([0, 0, 1]))
            plt.scatter(cam_center[0], cam_center[2], c='b')
            plt.arrow(cam_center[0], cam_center[2], cam_dir[0], cam_dir[2], head_width=5, head_length=200, fc='k', ec='k')
        plt.xlim(-1200, 1200)
        plt.ylim(-1200, 1200)
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Fanbeam Calibration Results')
        plt.text(-1000, 1000, 'iter: ' + str(self.iter))
        ax_ins = inset_axes(plt.gca(), width="30%", height="30%", loc=1)
        ax_ins.plot(range(len(self.losses)), self.losses, c='r')
        ax_ins.set_title('MSE Loss')
        save_dir = './debug/calibration'
        os.makedirs(save_dir, exist_ok=True)
        save_name = str(self.iter).zfill(5) + '.png'
        plt.savefig(os.path.join(save_dir, save_name))
        if self.iter == self.n_iter:
            return plt.show()
