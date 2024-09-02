import numpy as np
import open3d as o3d
import cv2

def plot_rays(ray_directions: np.array, ray_origins: np.array, ray_length: float):
    """
    Plot rays of a scanner (open3d).

    Args:
    ray_directions (np.array(W, H, 3)): ray directions.
    ray_origins (np.array(W, H, 3)): ray origins.
    ray_length (float): ray length.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    W, H, _ = ray_directions.shape
    ori1 = ray_origins[0, 0, :]
    ori2 = ray_origins[W - 1, 0, :]
    ori3 = ray_origins[W - 1, H - 1, :]
    ori4 = ray_origins[0, H - 1, :]
    end1 = ray_origins[0, 0, :] + ray_directions[0, 0, :] * ray_length
    end2 = ray_origins[W - 1, 0, :] + ray_directions[W - 1, 0, :] * ray_length
    end3 = ray_origins[W - 1, H - 1, :] + ray_directions[W - 1, H - 1, :] * ray_length
    end4 = ray_origins[0, H - 1, :] + ray_directions[0, H - 1, :] * ray_length
    lines = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
    pts = np.vstack([ori1, ori2, ori3, ori4, end1, end2, end3, end4])
    line_ray = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_ray

def plot_cube(cube_center: np.array, cube_size: np.array):
    """
    Plot a cube (open3d).

    Args:
    cube_center (np.array(3, 1)): cube center.
    cube_size (np.array(3, 1)): cube size.

    Returns:
    lines (o3d.geometry.LineSet): output lines.

    """

    # coordinate frame
    colorlines = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.array([[0], [0], [0], [1]])
    unit = 0.3
    axes = np.array([[unit, 0, 0],
                     [0, unit, 0],
                     [0, 0, unit],
                     [1, 1, 1]]) * np.vstack([np.hstack([cube_size, cube_size, cube_size]), np.ones((1, 3))])
    points = np.vstack([np.transpose(origin), np.transpose(axes)])[:, :-1]
    points += cube_center.squeeze()
    lines = [[0, 1], [0, 2], [0, 3]]
    worldframe = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    worldframe.colors = o3d.utility.Vector3dVector(colorlines)
    # bbox
    xyz_min = cube_center.squeeze() + np.array([-0.5, -0.5, -0.5]) * cube_size.squeeze()
    xyz_max = cube_center.squeeze() + np.array([0.5, 0.5, 0.5]) * cube_size.squeeze()
    points = [[xyz_min[0], xyz_min[1], xyz_min[2]],
              [xyz_max[0], xyz_min[1], xyz_min[2]],
              [xyz_min[0], xyz_max[1], xyz_min[2]],
              [xyz_max[0], xyz_max[1], xyz_min[2]],
              [xyz_min[0], xyz_min[1], xyz_max[2]],
              [xyz_max[0], xyz_min[1], xyz_max[2]],
              [xyz_min[0], xyz_max[1], xyz_max[2]],
              [xyz_max[0], xyz_max[1], xyz_max[2]]]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set_bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set_bbox.colors = o3d.utility.Vector3dVector(colors)
    return line_set_bbox + worldframe

def draw_line(pts1, pts2): 
    lines = [[0, 1]]
    pts = np.vstack([pts1, pts2])
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line


def draw_image(fanbeam): 
    # mesh 텍스처로 이미지 입혀서 시각화 
    
    image_path = fanbeam.image_path
    DSD = fanbeam.DSD
    cam_center = fanbeam.cam_center
    R = fanbeam.R
    Cx = fanbeam.Cx
    Cy = fanbeam.H / 2

    image_ = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) 

    T = -np.matmul(R, cam_center)
    T[1] = 0

    pts1 = np.array([-Cx, Cy, DSD])
    pts1 = R.T@(pts1 - T)
    pts1[1] = Cy
    pts2 = np.array([Cx, Cy, DSD])
    pts2 = R.T@(pts2 - T)
    pts2[1] = Cy
    pts3 = np.array([-Cx, -Cy, DSD])
    pts3 = R.T@(pts3 - T)
    pts3[1] = -Cy
    pts4 = np.array([Cx, -Cy, DSD])
    pts4 = R.T@(pts4 - T)
    pts4[1] = -Cy
    # TODO 함수로

    vertices = np.array([pts3, pts4, pts2, pts1])
    triangles = np.array([[0,1,2], [0,2,3]])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # texture 
    v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                    [0, 1], [1, 0], [0, 0]])
    mesh.textures = [o3d.geometry.Image(image_)]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
    
    vertices_ = np.array([pts4, pts3, pts1, pts2])
    triangles_ = np.array([[0,1,2], [0,2,3]])
    mesh_ = o3d.geometry.TriangleMesh()
    mesh_.vertices = o3d.utility.Vector3dVector(vertices_)
    mesh_.triangles = o3d.utility.Vector3iVector(triangles_)
    # texture_flip
    v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                    [0, 1], [1, 0], [0, 0]])
    image_flip = cv2.flip(image_, 1) 
    mesh_.textures = [o3d.geometry.Image(image_flip)]
    mesh_.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
    mesh_.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
    
    mesh += mesh_

    return mesh

def draw_cam(fanbeam):
    DSD = fanbeam.DSD
    cam_center = fanbeam.cam_center
    R = fanbeam.R
    T = -np.matmul(R, cam_center)
    Cx = fanbeam.Cx
    Cy = fanbeam.H / 2
    y = 0

    pts1 = np.array([-Cx,y, DSD])
    pts_w1 = R.T@(pts1 - T)
    pts_w1[1] = y
    pts2 = np.array([Cx,y, DSD])
    pts_w2 = R.T@(pts2 - T)
    pts_w2[1] = y

    line = draw_line(pts_w1, pts_w2)
    line1 = draw_line(pts_w1, cam_center)
    line2 = draw_line(pts_w2, cam_center)

    line_set = line + line1 + line2

    return line_set

def plot_ray(DSO, DSD, Cx, Cy, u, v, theta):

    cos = np.cos(theta)
    sin = np.sin(theta)
    Cw = np.array([DSO*sin , v-Cy , -DSO*cos])
    C_center = np.array([DSO*sin , 0 , -DSO*cos])

    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    T = -np.matmul(R, Cw)
    T[1] = v-Cy

    pts1 = np.array([Cx-u, v-Cy, DSD]) ### u-Cx
    pts1 = R.T@(pts1 - T)
    pts1[1] = v-Cy

    # line pts1 and Cw 
    # ray = draw_line(pts1, Cw)
    # ray += draw_line(C_center, Cw)
    ray = draw_line(pts1, C_center) ####### Cone beam
    
    
    return ray 

def plot_camcenter(cam_center):
    
    cam_center = np.array(cam_center)
    cam_center = np.reshape(cam_center, (1,3))
    print(cam_center.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_center)
    pcd.paint_uniform_color([0, 0, 1])

    return pcd