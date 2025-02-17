import os
import torch
import numpy as np
import pickle
from PIL import Image
from scripts.io_ply import read_ply
import json
import math

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def read_camera_mvhumannet(intri_name, extri_name, camera_scale ,cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    with open(intri_name, 'r') as f:
        camera_intrinsics = json.load(f)

    with open(extri_name, 'r') as f:
        camera_extrinsics = json.load(f)
        
    print("intri: ", camera_intrinsics)

    item = os.path.dirname(intri_name).split("/")[-1]
    print("item: ", item)
    # intri = FileStorage(intri_name)
    # extri = FileStorage(extri_name)
    cams, P = {}, {}

    # cam_names = intri.read('names', dt='list')

    cam_names = camera_extrinsics.keys()
    cam_infos = []

    for i, cam in enumerate(cam_names):
        
        updated_cam = cam.split('.')[0].split('_')
        # print("updated_cam_before: ", updated_cam)
        # updated_cam[1] = 'cache'   # for test
        updated_cam = updated_cam[-1]
        # print("updated_cam_after: ", updated_cam)

        cams[updated_cam] = {}
        # cams[updated_cam]['K'] = intri.read('K_{}'.format( cam))
        cams[updated_cam]['K'] = np.array(camera_intrinsics['intrinsics'])
        cams[updated_cam]['invK'] = np.linalg.inv(cams[updated_cam]['K'])

        # import IPython; IPython.embed(); exit()

        # Rvec = extri.read('R_{}'.format(cam))
        # Tvec = extri.read('T_{}'.format(cam))
        # assert Rvec is not None, cam
        # R = cv2.Rodrigues(Rvec)[0]
        
        R = np.array(camera_extrinsics[cam]['rotation']) 
        # opencv format -> opengl format
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ R @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # longgang
        # Tvec = np.array(camera_extrinsics[cam]['translation'])[:, None] / 1000 * 100 / 65
        # mm -> m
        Tvec = np.array(camera_extrinsics[cam]['translation'])[:, None] / 1000 * camera_scale

        RT = np.hstack((R, Tvec))

        cams[updated_cam]['RT'] = RT
        cams[updated_cam]['R'] = R
        # cams[updated_cam]['Rvec'] = Rvec
        cams[updated_cam]['T'] = Tvec
        # cams[updated_cam]['center'] = - Rvec.T @ Tvec
        P[updated_cam] = cams[updated_cam]['K'] @ cams[updated_cam]['RT']
        cams[updated_cam]['P'] = P[updated_cam]

        focal_length_x = camera_intrinsics['intrinsics'][0][0]
        focal_length_y = camera_intrinsics['intrinsics'][1][1]
        height = camera_intrinsics['intrinsics'][1][2] * 2
        width = camera_intrinsics['intrinsics'][0][2] * 2
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        depth_params = None
        image_name = "{}.png".format(i)
        
        # cams[updated_cam]['dist'] = np.array(camera_intrinsics['dist'])
        cams[updated_cam]['dist'] = None   # dist for cv2.undistortPoint

        cam_info = CameraInfo(
            uid=i,
            R=R,
            T=Tvec,
            FovY=FovY,
            FovX=FovX,
            depth_params=depth_params,
            image_path=image_name,
            image_name=image_name,
            depth_path="",
            width=width,
            height=height,
            is_test=False
        )
        cam_infos.append(cam_info)

    # cams['basenames'] = cam_names
    return cam_infos