import numpy as np

class FanBeam:
    def __init__(self, DSD, angle_y, T, image_path, beads_2d):
        self.H = 800
        self.W = 640
        self.Cx = self.W / 2 
        self.Cy = self.H / 2
        self.DSD = DSD
        self.near = 0
        self.far = 1000
        self.image_path = image_path 
        self.beads_2d = beads_2d

        # TODO distortion modeling 
        # self.detector_curvature = 0 
        # self.detector_rot_x = 0
        # self.detector_rot_y = 0
        # self.detector_rot_z = 0

        self.theta = angle_y
        self.R = self.get_rot_mat(self.theta) 
        self.T = T
        self.cam_center = - np.dot(self.R.T, self.T)
    
    def get_rot_mat(self, theta):
        """
        y축으로만 회전 
        Args:
        theta (float): camera rotation angle in radians.
        Returns:
        numpy.ndarray: A 3x3 rotation matrix.
        """

        theta_x = 0
        theta_y = theta
        theta_z = 0
        rot_x = np.array([[1, 0, 0],
                            [0, np.cos(theta_x), -np.sin(theta_x)],
                            [0, np.sin(theta_x), np.cos(theta_x)]])
        rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                            [0, 1, 0],
                            [-np.sin(theta_y), 0, np.cos(theta_y)]])
        rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                            [np.sin(theta_z), np.cos(theta_z), 0],
                            [0, 0, 1]])
        # rot = rot_x @ rot_y @ rot_z
        rot = rot_y
        return rot
    
    def get_ray(self):

        W, H = self.W, self.H
        rot = self.R
        cam_center = self.cam_center
        DSD = self.DSD
        i, j = np.meshgrid(np.arange(W), np.arange(H))
        uu = (i + 0.5 - W / 2) 
        vv = (j + 0.5 - H / 2) 
        dirs = np.stack([uu / DSD, vv / DSD, np.ones_like(uu)], -1)
        rays_d = np.sum(np.matmul(rot.T, dirs[..., None]), -1)
        rays_o = np.broadcast_to(cam_center, rays_d.shape)
        rays_d[:,:,1] = 0
        rays_o[:,:,1] = vv

        return rays_o, rays_d
    
    def project_point(self, p3d):
        raise NotImplementedError