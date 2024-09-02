import os 
import cv2
import numpy as np

"""
Linear Pushbroom model test
LPB 모델 synthetic 데이터로 테스트

Papers: 
Linear Pushbroom Cameras
https://users.cecs.anu.edu.au/~hartley/Papers/linear-pushbroom/PAMI/pami-paper.pdf
ROBUST PLANE-BASED CALIBRATION FOR LINEAR CAMERAS
https://donnessime.github.io/papers/lpb_calibration.pdf
Calibration for the stereo hyperspectral pushbroom camera HYPSOS
https://ieeexplore.ieee.org/document/9856258
"""

def estimate_homography(chessboard, uvs):
    """
    3.3 Homography estimation
    Args:
    chessboard: 3d points of chessboard (n_rows*n_cols, 3) # generate_chessboard(n_rows, n_cols, square_size) (a, b, 0)
    uvs: 2d points of chessboard (num_images, n_rows*n_cols, 2) # detected chessboard points
    Returns:
    H: homography matrix (num_images, 3, 6)
    """
    num_images = len(uvs)
    N = chessboard.shape[0]
    H = [None] * num_images
    a = chessboard[:, 0]
    b = chessboard[:, 1]
    for i in range(num_images):
        u = uvs[i][:, 0]
        v = uvs[i][:, 1]
        A = np.vstack([
            np.column_stack([a, b, np.ones(N), np.zeros((N, 6)), -u * a, -u * b, -u]),
            np.column_stack([np.zeros((N, 3)), a, b, np.ones(N), a**2, b**2, a * b, -a * v, -b * v, -v]),
            np.column_stack([-a * v, -b * v, -v, a * u, b * u, u, a**2 * u, b**2 * u, a * b * u, np.zeros((N, 3))])
        ])
        _, _, V = np.linalg.svd(A)
        h = V[-1, :] / V[-1, -1]
        H[i] = np.vstack([
            np.hstack([h[0:3], np.zeros(3)]),
            h[3:9],
            np.hstack([h[9:11], 1, np.zeros(3)])
        ])
    return H

def homography_projection(H, chessboard):
    """
    H : homography matrix (3, 6)
    H @ [a, b, 1, a^2, b^2, ab].T ~ [u, v, 1]
    """
    a = chessboard[:, 0]
    b = chessboard[:, 1]
    N = chessboard.shape[0]
    X = np.column_stack([a, b, np.ones(N), a**2, b**2, a * b])
    uv = np.dot(H, X.T)
    uv /= uv[2, :]
    return uv[:2, :].T

def extract_KRT(Hs):
    """
    extract intrinsic and extrinsic parameters from homography matrix
    Args:
    Hs: homography matrix (num_images, 3, 3)
    Returns:
    K: intrinsic parameter matrix (3, 3)
    R: extrinsic rotation matrix (num_images, 3, 3)
    T: extrinsic translation matrix (num_images, 3) 


    3.4 Intrinsic parameter estimation

    m11m12,     m11m32+m12m31,      m31m32,         m21m22 
    m11^2-m12^2, 2(m11m31-m12m32),  m31^2-m32^2, m21^2-m22^2
    @ (s^2, -s^2u0, s^2(u0^2+f^2), f^2/(ti,3)^2).T = 0

    (M_i^T @ X_i @ M_i)_12 = 0 
    (M_i^T @ X_i @ M_i)_11 = (M_i^T @ X_i @ M_i)_22

    """
    N = len(Hs)
    A = np.zeros((2*N, 3+N))
    for i in range(N):
        H = Hs[i]
        M = np.array([[H[0,0], H[0,1]],
                      [H[1,0]-H[2,0]*H[1,2], H[1,1]-H[2,1]*H[1,2]],
                      [H[2,0], H[2,1]]])
        A[2*i, :3] = [M[0, 0] * M[0, 1], M[0, 0] * M[2, 1] + M[0, 1] * M[2, 0], M[2, 0] * M[2, 1], ]  # M[1, 0] * M[1, 1]]
        A[2*i+1, :3] = [M[0, 0]**2 - M[0, 1]**2, 2 * (M[0, 0] * M[2, 0] - M[0, 1] * M[2, 1]), M[2, 0]**2 - M[2, 1]**2, ] # M[1, 0]**2 - M[1, 1]**2]
        A[2*i, 3+i] = M[1, 0] * M[1, 1]
        A[2*i+1, 3+i] = M[1, 0]**2 - M[1, 1]**2


    _, _, V = np.linalg.svd(A)
    x1, x2, x3 = V[-1, :3] 
    u0 = -x2 / x1
    f = np.sqrt(x3 / x1 - u0**2)

    xi = np.zeros(N)
    for i in range(N):
        xi[i] = V[-1, 3+i] 
    
    # ss =np.zeros(N)
    # for i in range(N):
    #     ss[i] = np.sqrt(Hs[i][0, 0]**2 + Hs[i][0, 1]**2) 
    # s = np.sum(ss) / N

    A = np.zeros((2*N, 1+N))
    b = np.tile([1, 1], N) # b = np.tile([1, 1, 0], N)

    for i in range(N):
        H = Hs[i]
        M = np.array([[H[0,0], H[0,1]],
                      [H[1,0]-H[2,0]*H[1,2], H[1,1]-H[2,1]*H[1,2]],
                      [H[2,0], H[2,1]]])

        A[2*i, 0] = M[1, 0]**2
        A[2*i, i+1] = ((u0**2 + f**2)*M[2, 0]**2 - 2*u0*M[0, 0]*M[2, 0] + M[0, 0]**2) / f**2
        A[2*i+1, 0] = M[1, 1]**2
        A[2*i+1, i+1] = ((u0**2 + f**2)*M[2, 1]**2 - 2*u0*M[0, 1]*M[2, 1] + M[0, 1]**2) / f**2

    x, _, _, _ = np.linalg.lstsq(A, b)
    s = np.sqrt(1 / x[0])
    t3 = np.sqrt(x[1:])
    K = np.array([[f, 0, u0], [0, s, 0], [0, 0, 1]])

    """
    3.5 extrinsic parameter estimation
    transformation matrix from equation (6) 
    enforce orthogonality constraint on R by SVD
    """

    R = [None] * N
    T = [None] * N

    for i in range(N):
        H = Hs[i]
        T[i] = np.array([(H[0, 2] - u0) * t3[i] / f, H[1, 2] / s, t3[i]])
        R[i] = np.array([(H[0, 0:2] - u0 * H[2, 0:2]) * T[i][2] / f, 
                        H[1, 0:2] / s - H[2, 0:2] * T[i][1], 
                        H[2, 0:2] * T[i][2]])
        R[i] = np.column_stack((R[i], np.cross(R[i][:, 0], R[i][:, 1])))
        U, _, V = np.linalg.svd(R[i])
        R[i] = np.dot(U, V)

    return K, R, T

def generate_chessboard(n_rows, n_cols, square_size):
    chessboard = np.zeros((n_rows * n_cols, 3))
    for i in range(n_rows):
        for j in range(n_cols):
            chessboard[i * n_cols + j, :] = [j * square_size, i * square_size, 0]
    return chessboard

def get_projection_mat(f, s, Cx):
    return np.array([
        [f, 0, Cx],
        [0, s, 0],
        [0, 0, 1]
    ])

def project_point(K, RT, p3d):
    """
    u   fX + CxZ     f  0  Cx    X
    v ~   sYZ     =  0  s  0  *  YZ 
    1      Z         0  0  1     Z
    
    """
    p3d = np.hstack([p3d, np.ones((p3d.shape[0], 1))])
    p3d_ = RT @ p3d.T
    # X, Y, Z -> X, YZ, Z 
    p3d_lpb = np.vstack([p3d_[0], p3d_[1] * p3d_[2], p3d_[2]])
    p2d = K @ p3d_lpb
    p2d /= p2d[2, :]

    return p2d[:2, :].T

def apply_object_transform(p3d, rx, ry, rz, tx, ty, tz):
    # apply rigid transform 

    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    rz = np.deg2rad(rz)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    T = np.array([tx, ty, tz])

    p3d = R @ p3d.T + T.reshape(3, 1)
    
    return p3d.T, R

def get_transform_mat(r1, r2, r3, t1, t2, t3):

    rx = np.deg2rad(r1)
    ry = np.deg2rad(r2)
    rz = np.deg2rad(r3)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    T = np.array([t1, t2, t3])
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = T

    return RT 

if __name__ == "__main__":
    
    #####################################################################
    # synthetic data generation
    #####################################################################
    H, W = 800, 640
    chessboard = generate_chessboard(9, 11, 20)
    f = np.random.randint(800, 1200)
    K = get_projection_mat(f, 1, W/2)
    print("GT K:", K)
    num_images = 5

    RT  = np.eye(4)
    RT[:3, :3] = np.eye(3)
    RT[:3, 3] = np.array([0, 400, 1000]) # w2c transformation
    RT = RT[:3, :]
    p2d_ls = [] # projected 2d points
    R_gt = [] # GT w2c rotation
    T_gt = [] # GT w2c translation
    for i in range(num_images):
        r1, r2, r3 = np.random.uniform(-50, 50, 3) # obj transform
        t1, t2, t3 = np.random.uniform(-100, 100, 3) # obj transform
        chessboard_transformed, R_ = apply_object_transform(chessboard, r1, r2, r3, t1, t2, t3)
        p2d = project_point(K, RT, chessboard_transformed)
        p2d_ls.append(p2d)

        R = RT[:3, :3] @ R_
        T = RT[:3, 3] + RT[:3, :3] @ np.array([t1, t2, t3])
        R_gt.append(R)
        T_gt.append(T)

    # draw chessboard 
    save_dir = 'debug/lpb/'
    os.makedirs(save_dir, exist_ok=True)
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_images)]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(num_images):
        for j in range(p2d_ls[i].shape[0]):
            cv2.circle(img, tuple(p2d_ls[i][j].astype(np.int32)), 3, colors[i], -1) 
    cv2.imwrite(os.path.join(save_dir, "img_all.png"), img)

    #####################################################################
    # LPB calibration example
    #####################################################################
    Hs = estimate_homography(chessboard, p2d_ls)
    reprojection = homography_projection(Hs[0], chessboard)
    mse = np.mean(np.linalg.norm(reprojection - p2d_ls[0], axis=1))
    print(mse)
    K, Rs, Ts = extract_KRT(Hs)

    print("estimated K :", K)
    print("estimated R :", Rs)
    print("GT R :", R_gt)
    print("estimated T :", Ts)
    print("GT T :", T_gt)

    # reprojection 
    for i in range(num_images):
        R = Rs[i]
        T = Ts[i]
        p2d_reproj = project_point(K, np.hstack([R, T.reshape(3, 1)]), chessboard)
        p2d = p2d_ls[i]
        mse = np.mean(np.linalg.norm(p2d_reproj - p2d, axis=1))
        print("MSE :", mse)
        save_dir = 'debug/lpb/'
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # for j in range(p2d.shape[0]):
        #     cv2.circle(img, tuple(p2d[j].astype(np.int32)), 3, (0, 255, 0), -1)
        #     cv2.circle(img, tuple(p2d_reproj[j].astype(np.int32)), 3, (0, 0, 255), -1)
        # cv2.imwrite(os.path.join(save_dir, f"img_{i}_reproj.png"), img)

    print("done")