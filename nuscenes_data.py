"""
code adapted from https://github.com/nv-tlabs/lift-splat-shoot
and also https://github.com/wayveai/fiery/blob/master/fiery/data.py
"""
import os
import random
from functools import reduce

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, PointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes

#for depth 
from nuscenes.utils.data_classes import LidarPointCloud
import torch.nn.functional as F


from PIL import Image
from pyquaternion import Quaternion
from scipy import ndimage
# distributed dataloader
from torch.utils.data.distributed import DistributedSampler



import utils.geom
import utils.py
import utils.vox

# custom eval split
from custom_nuscenes_splits import create_drn_eval_split_scenes
#from custom_nuscenes_splits import custom_create_splits_scenes

'''depth image process '''
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

from monodepth2_on_nuscenes_cityscapes import networks
from torchvision import transforms
'''depth image process '''



'''point cloud process start '''
import open3d as o3d
'''point cloud process end   '''



discard_invisible = False

class EgoPose_Class():
    
    def __init__(self):

        self.trans_nusce = None
        self.rote_nusce = None
        print('EgoPose_Class initialized')
        
        
    def update_egopose(self, new_trans, new_rote):
        self.trans_nusce = new_trans
        self.rote_nusce = new_rote

        data = new_trans
        print('update egopose')



def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def get_pseudolidar_data(nusc, sample_rec, nsweeps, min_distance, use_pseudolidar_filters, dataroot):
    """
    Returns at most nsweeps of pseudolidar in the ego frame.
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # import ipdb; ipdb.set_trace()

    # points = np.zeros((5, 0))
    points = np.zeros((19, 0))  # 18 plus one for time

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['RADAR_FRONT']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    if use_pseudolidar_filters:
        RadarPointCloud.default_filters()
    else:
        RadarPointCloud.disable_filters()

    # Aggregate current and previous sweeps.
    # from all pseudolidars
    pseudolidar_chan_list = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    #pseudolidar_chan_list = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for pseudolidar_name in pseudolidar_chan_list:
        sample_data_token = sample_rec['data'][pseudolidar_name]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = RadarPointCloud.from_file(os.path.join(dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # print('time_lag', time_lag)
            # print('new_points', new_points.shape)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

#def get_pseudolidar_data(nusc, sample_rec, nsweeps, min_distance, use_pseudolidar_filters, dataroot):
    """
    Returns at most nsweeps of pseudolidar in the ego frame.
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # import ipdb; ipdb.set_trace()

    # points = np.zeros((5, 0))
    points = np.zeros((19, 0))  # 18 plus one for time

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['RADAR_FRONT']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    if use_pseudolidar_filters:
        RadarPointCloud.default_filters()
    else:
        RadarPointCloud.disable_filters()

    # Aggregate current and previous sweeps.
    # from all pseudolidars
    pseudolidar_chan_list = ["CAM_BACK", "CAM_BACK_LEFT" "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    #pseudolidar_chan_list = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for pseudolidar_name in pseudolidar_chan_list:
        sample_data_token = sample_rec['data'][pseudolidar_name]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            #current_pc = RadarPointCloud.from_file(os.path.join(dataroot, current_sd_rec['filename']))
            current_pc = RadarPointCloud.from_file(os.path.join("/root/data/dataset/nuscenes_datasets/nuscenes_datasets/nuscenes/samplesdepth/", current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # print('time_lag', time_lag)
            # print('new_points', new_points.shape)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points



def ego_to_cam(points, rot, trans, intrins=None):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    if intrins is not None:
        points = intrins.matmul(points)
        points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) & \
        (pts[0] > 1) & (pts[0] < W - 1) & \
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img


def img_crop(img, resize_dims, crop):
    img = img.crop(crop)
    return img


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))

denormalize_img_torch = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
))

normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))
totorch_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))
normalize_img_torch = torchvision.transforms.Compose((
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084 / 2. + 0.5, W / 2.],
        [4.084 / 2. + 0.5, W / 2.],
        [4.084 / 2. + 0.5, -W / 2.],
        [-4.084 / 2. + 0.5, -W / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def add_ego2(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084 / 2. + 1, W / 2.],
        [4.084 / 2. + 1, W / 2.],
        [4.084 / 2. + 1, -W / 2.],
        [-4.084 / 2. + 1, -W / 2.],
    ])
    pts = (pts - bx) / dx
    # pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
                     "singapore-hollandvillage",
                     "singapore-queenstown",
                     "boston-seaport",
                     "singapore-onenorth",
                 ]}
    return nusc_maps


def fetch_nusc_map2(rec, nusc_maps, nusc, scene2map, car_from_current):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

    global_from_car = transform_matrix(egopose['translation'],
                                       Quaternion(egopose['rotation']), inverse=False)

    trans_matrix = reduce(np.dot, [global_from_car, car_from_current])

    rot = np.arctan2(trans_matrix[1, 0], trans_matrix[0, 0])
    center = np.array([trans_matrix[0, 3], trans_matrix[1, 3], np.cos(rot), np.sin(rot)])

    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    # Debug only:
    # print("Current map: ", map_name)

    poly_names = ['drivable_area', 'carpark_area', 'ped_crossing', 'walkway', 'stop_line']
    line_names = ['road_divider', 'lane_divider']

    # lmap = get_local_map(nusc_maps[map_name], center, 50.0, poly_names, line_names)

    # own labels handling
    # use build in nuscenes function to get binary map mask
    patch_box = (center[0], center[1], 100, 100)  # patch of the map (center_x, center_y, height, width)
    rot_angle = rot * 180 / np.pi  # rotation of the patch -> should match the model output
    # print for console debug
    # print("Car rot angle: ", rot_angle)
    patch_angle = rot_angle  # angle in degrees
    # patch_angle = 0.0  # for debugging only
    canvas_size = (200, 200)  # size of generated output mask -> 200x200
    layer_names = poly_names + line_names
    map_mask = nusc_maps[map_name].get_map_mask(patch_box=patch_box,
                                                patch_angle=patch_angle,
                                                layer_names=layer_names,
                                                canvas_size=canvas_size)

    """ Debug: test if mask is correct:
    figsize = (12, 4)
    fig, ax = nusc_maps[map_name].render_map_mask(patch_box=patch_box,
                                                  patch_angle=patch_angle,
                                                  layer_names=layer_names,
                                                  canvas_size=canvas_size,
                                                  figsize=figsize,
                                                  n_row=1)
    """

    # generate bev view of the ego car for the current patch
    ego_in_bev = np.zeros_like(map_mask[0])

    W = 1.85
    bx = -49.75
    dx = 0.5
    car_points = np.array([[-4.084 / 2. + 1, W / 2.],
                           [4.084 / 2. + 1, W / 2.],
                           [4.084 / 2. + 1, -W / 2.],
                           [-4.084 / 2. + 1, -W / 2.]])

    car_points_scale_shift = (car_points - bx) / dx

    round_car_approx = np.round(car_points_scale_shift)
    x = abs((round_car_approx[0][1] + 1) - round_car_approx[-1][1])
    y = abs(round_car_approx[0][0] - (round_car_approx[1][0] - 1))
    car_mask = np.ones((int(y), int(x)))
    ego_in_bev[int(round_car_approx[0][0]): int(round_car_approx[1][0] - 1),
    int(round_car_approx[-1][1]): int(round_car_approx[0][1] + 1)] = car_mask

    rot_angle = 0.0
    ego_angle = np.arctan2(car_from_current[1, 0], car_from_current[0, 0])
    ego_angle = ego_angle * 180 / np.pi

    bev_plane_rot = ndimage.rotate(ego_in_bev, angle=ego_angle + 90.0, reshape=False)
    egocar_bev = (bev_plane_rot > 0.1).astype(int)

    return map_mask, egocar_bev


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
            )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


# SegMam colors
masks_colors = np.zeros((7, 3))
masks_colors[0] = (125. / 255., 172. / 255., 103. / 255.)  # pink
masks_colors[1] = (155. / 255., 135. / 255., 0. / 255.)  # orange
masks_colors[2] = (0. / 255., 120. / 255., 0. / 255.)  # dimmer green
masks_colors[3] = (124. / 255., 130. / 255., 178. / 255.)  # teal
masks_colors[4] = (155. / 255., 0. / 255., 125. / 255.)  # red
masks_colors[4] = (125. / 255., 125. / 255., 0/ 255.)  # yellow
masks_colors[5] = (118. / 255., 0. / 255., 118. / 255.)  # purple. 


def get_rgba_map_from_mask2(map_masks, threshold=0.8, a=0.4):
    """
    class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
    class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
    class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
    class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
    class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
    class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
    class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n

    threshold: defines at which probability the pixel belongs to a certain class
    a: sets the opacity of the mask layers; if alpha=1.0 no overlapping visible; default: alpha=0.4
    """
    masks_shape = map_masks.shape
    rgba_image = np.ones((4, masks_shape[1], masks_shape[2]), dtype=np.float32)

    for layer, (mask, color) in enumerate(zip(map_masks, masks_colors)):
        alpha = a * (mask > threshold).astype(np.float32)  # a = 0.4
        # Assign RGB values based on the color
        rgb = np.full((3, 200, 200), color[..., np.newaxis, np.newaxis])

        # Combine the RGB and alpha channels
        rgba = np.concatenate([rgb, np.expand_dims(alpha, axis=0)], axis=0)

        # Update the RGBA image with the values of the current mask
        rgba_image = rgba_image * (1 - alpha) + rgba * alpha

    bev_map = rgba_image[:3]
    if bev_map.max() > 1.0:
        bev_map = np.clip(bev_map, 0.0, 1.0)
        
    return torch.Tensor(bev_map)


def get_rgba_map_from_mask2_on_batch(map_masks: torch.Tensor, threshold: float = 0.8, a: float = 0.4) -> torch.Tensor:
    """
    class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
    class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
    class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
    class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
    class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
    class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
    class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n

    threshold: defines at which probability the pixel belongs to a certain class
    a: sets the opacity of the mask layers; if alpha=1.0 no overlapping visible; default: alpha=0.4
    """
    B, classes, length_z, length_x = map_masks.shape  # (B,7,200,200)
    # (7, 200, 200)
    bev_map = np.ones((B, 3, 200, 200))  # (B, 3, 200, 200)
    rgba_image = np.ones((4, length_z, length_x), dtype=np.float32)  # (B, 4, 200, 200)

    for b in range(B):
        for layer, (mask, color) in enumerate(zip(map_masks[b], masks_colors)):
            alpha = a * (mask > threshold).astype(np.float32)  # a = 0.4   # (200, 200)
            # Assign RGB values based on the color
            rgb = np.full((3, 200, 200), color[..., np.newaxis, np.newaxis])  # (3, 200, 200)

            # Combine the RGB and alpha channels
            rgba = np.concatenate([rgb, np.expand_dims(alpha, axis=0)], axis=0)  # (4, 200, 200)

            # Update the RGBA image with the values of the current mask
            rgba_image = rgba_image * (1 - alpha) + rgba * alpha  # (4, 200, 200)

        bev_map[b] = rgba_image[:3]

    # clip values
    if bev_map.max() > 1.0:
        bev_map = np.clip(bev_map, 0.0, 1.0)
    return torch.Tensor(bev_map)


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc: NuScenes, nusc_maps: dict, is_train: bool, data_aug_conf: dict,
                 centroid: torch.Tensor = None, bounds: tuple = None, res_3d=None, nsweeps: int = 1, seqlen: int = 1,
                 refcam_id: int = 1, get_tids: bool = False, temporal_aug: bool = False,
                 use_pseudolidar_filters: bool = False, do_shuffle_cams: bool = True, pseudolidar_encoder_type: bool = None,
                 use_shallow_metadata: bool = True, use_pre_scaled_imgs: bool = True, custom_dataroot: str = None,
                 use_obj_layer_only_on_map: bool = False, vis_full_scenes: bool = False,
                 use_pseudolidar_occupancy_map: bool = False, do_drn_val_split: bool = False, get_val_day: bool = False,
                 get_val_rain: bool = False, get_val_night: bool = False, print_details: bool = False):

        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        # self.grid_conf = grid_conf
        self.nsweeps = nsweeps
        self.use_pseudolidar_filters = use_pseudolidar_filters
        self.do_shuffle_cams = do_shuffle_cams
        self.res_3d = res_3d
        self.bounds = bounds
        self.centroid = centroid

        self.seqlen = seqlen
        self.refcam_id = refcam_id
        self.vis_full_scenes = vis_full_scenes
        self.use_pseudolidar_occupancy_map = use_pseudolidar_occupancy_map

        self.do_drn_val_split = do_drn_val_split
        self.get_val_day = get_val_day
        self.get_val_rain = get_val_rain
        self.get_val_night = get_val_night

        self.print_details = print_details

        self.dataroot = self.nusc.dataroot

        self.scenes = self.get_scenes()
        #print(f'self.scenes{self.scenes}')
        
        
        if self.do_drn_val_split and not self.is_train:
            # returns samples in the correct "DAY/RAIN/NIGHT" order
            self.ixes = self.get_ordered_drn_samples()
            #print(f'lelf.ixes {len(self.ixes)} ')
            
        else:
            self.ixes = self.prepro()
            
        if temporal_aug:
            self.indices = self.get_indices_tempaug()
        else:
            self.indices = self.get_indices()
            #print('not temporal aug')

        self.get_tids = get_tids

        self.pseudolidar_encoder_type = pseudolidar_encoder_type 
        self.use_shallow_metadata = use_shallow_metadata

        self.use_pre_scaled_imgs = use_pre_scaled_imgs
        self.custom_dataroot = custom_dataroot

        self.use_obj_layer_only_on_map = use_obj_layer_only_on_map

        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.bounds
        Z, Y, X = self.res_3d
        self.Z = Z
        self.Y = Y
        self.X = X

        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=torch.from_numpy(self.centroid).float().cuda(),
            bounds=self.bounds,
            assert_cube=False)

        grid_conf = {  # note the downstream util uses a different XYZ ordering
            'xbound': [XMIN, XMAX, (XMAX - XMIN) / float(X)],
            'ybound': [ZMIN, ZMAX, (ZMAX - ZMIN) / float(Z)],
            'zbound': [YMIN, YMAX, (YMAX - YMIN) / float(Y)],
        }
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        if self.print_details:
            # print('ixes', self.ixes.shape)
            print('indices', self.indices.shape)

            
    def get_scenes(self):
        if self.do_drn_val_split and not self.is_train:  # only for eval...
            if self.get_val_day:
                split = 'val_day'
            elif self.get_val_rain:
                split = 'val_rain'
            elif self.get_val_night:
                split = 'val_night'
            else:  # get all val data, but in: day, rain, night order
                split = 'val_all'
            scenes = create_drn_eval_split_scenes()[split]
            
        else:
            # filter by scene split
            split = {
                'v1.0-trainval': {True: 'train', False: 'val'},
                'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
                'v1.0-test': {True: 'test', False: 'test'},
            }[self.nusc.version][self.is_train]
            scenes = create_splits_scenes()[split]
            
        return scenes
    
    #drn -> day, rain, night
    def get_drn_scene_tokens(self):
        # Create a mapping from scene names to scene tokens
        scene_name_to_token = {scene['name']: scene['token'] for scene in self.nusc.scene}
        #print(f'self scenes len {len(self.scenes)}')
        # Retrieve scene tokens in the order of scene names in self.scenes
        drn_scene_tokens = []
        for scene_name in self.scenes:
            if scene_name in scene_name_to_token:
                drn_scene_tokens.append(scene_name_to_token[scene_name])

        return drn_scene_tokens

    def get_ordered_drn_samples(self):
        """
        Returns:
            all samples in the dataset sorted for scene ids
        """
        # Create a mapping from scene names to scene tokens
        scene_name_to_token = {scene['name']: scene['token'] for scene in self.nusc.scene}
        drn_samples = []
        
        #print(f'self scenes len {len(self.scenes)} ')
        
        for scene_name in self.scenes:
            
            if scene_name in scene_name_to_token:
                scene_token = scene_name_to_token[scene_name]
                # Retrieve the first sample token from the scene
                scene = self.nusc.get('scene', scene_token)
                first_sample_token = scene['first_sample_token']
                #print(f"scenescenescenescene{scene}")

                # Start with the first sample
                current_sample_token = first_sample_token
                while current_sample_token:
                    # Retrieve the current sample
                    sample = self.nusc.get('sample', current_sample_token)
                    drn_samples.append(sample)

                    # Move to the next sample in the scene
                    current_sample_token = sample['next']
                    
        #print(f'drn_samples {len(drn_samples)} ')    
        return drn_samples

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
    
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        return samples

    def get_indices(self):
        """
        Note: if used for sequence collection -> e.g. get 40 samples of a scene for visualization
        the following will happen: scenes with under (<)40 samples will be skipped and scenes with
        over (>)40 samples  will occur essentially be stored multiple times
        this behaviour does not affect single sample sequences, used in training and eval.
        For scene - based visualization, please set self.vis_full_scenes to True.
        """
        indices = []
        if self.vis_full_scenes:
            # --> visualize the whole scene and avoid skipped / doubled scenes --> set variable
            valid = True
            index = 0
            while index < len(self.ixes):  # as long as we are not over every sample in the current split...
                # start with first index, go as long as "valid"
                current_indices = []
                while valid:
                    rec = self.ixes[index]
                    current_indices.append(index)
                    # check if next sample is in the current scene --> if not -> close scene here
                    if rec['next'] == '':
                        valid = False
                    index += 1

                # add length of valid indices
                current_indices.insert(0, len(current_indices))
                # check if we need to fill up to 42 values
                while len(current_indices) != 42:
                    current_indices.append(0)

                indices.append(current_indices)
                valid = True

        else:  # single sample or incomplete scenes
            for index in range(len(self.ixes)):
                is_valid_data = True
                previous_rec = None
                current_indices = []
                for t in range(self.seqlen):
                    index_t = index + t
                    # Going over the dataset size limit.
                    if index_t >= len(self.ixes):
                        is_valid_data = False
                        break
                    rec = self.ixes[index_t]
                    # Check if scene is the same
                    if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                        is_valid_data = False
                        break

                    current_indices.append(index_t)
                    previous_rec = rec

                if is_valid_data:
                    index += 38
                    indices.append(current_indices)

        return np.asarray(indices)

    def get_indices_tempaug(self):
        indices = []
        t_patterns = None
        if self.seqlen == 1:
            return self.get_indices()
        elif self.seqlen == 2:
            # seq options: (t, t+1), (t, t+2)
            t_patterns = [[0, 1], [0, 2]]
        elif self.seqlen == 3:
            # seq options: (t, t+1, t+2), (t, t+1, t+3), (t, t+2, t+3)
            t_patterns = [[0, 1, 2], [0, 1, 3], [0, 2, 3]]
        elif self.seqlen == 5:
            t_patterns = [
                [0, 1, 2, 3, 4],  # normal
                [0, 1, 2, 3, 5], [0, 1, 2, 4, 5], [0, 1, 3, 4, 5], [0, 2, 3, 4, 5],  # 1 skip
                # [1,0,2,3,4], [0,2,1,3,4], [0,1,3,2,4], [0,1,2,4,3], # 1 reverse
            ]
        else:
            raise NotImplementedError("timestep not implemented")

        for index in range(len(self.ixes)):
            for t_pattern in t_patterns:
                is_valid_data = True
                previous_rec = None
                current_indices = []
                for t in t_pattern:
                    index_t = index + t
                    # going over the dataset size limit
                    if index_t >= len(self.ixes):
                        is_valid_data = False
                        break
                    rec = self.ixes[index_t]
                    # check if scene is the same
                    if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                        is_valid_data = False
                        break

                    current_indices.append(index_t)
                    previous_rec = rec

                if is_valid_data:
                    indices.append(current_indices)
                    # indices.append(list(reversed(current_indices)))
                    # indices += list(itertools.permutations(current_indices))

        return np.asarray(indices)

    def sample_augmentation(self) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """
        Applies augmentation on inout image data by resizing and cropping
        Returns:
            tuple[tuple,tuple]: resize_dims, crop
        """
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']

            resize_dims = (int(fW * resize), int(fH * resize))

            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = self.data_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            resize_dims, crop = self.sample_augmentation()

            samp = self.nusc.get('sample_data', rec['data'][cam])

            if not self.use_pre_scaled_imgs:
                imgname = os.path.join(self.dataroot, samp['filename'])
                img = Image.open(imgname)
                W, H = img.size
            else:
                custom_path = os.path.join(self.custom_dataroot, '#CUSTOM_RES#')  # TODO: adapt to rescaled imgs folder
                imgname = os.path.join(custom_path, samp['filename'])
                img = Image.open(imgname)
                W, H = (1600, 900)  # img.size must be fixed to keep the rest of the code working

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # W and H: original image size to scale intrinsics correctly
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            intrin = utils.geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = utils.geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = utils.geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = img_transform(img, resize_dims, crop)

            imgs.append(totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

        return torch.stack(imgs), torch.stack(rots), torch.stack(trans), torch.stack(intrins)

    # def get_depth_image_data(self, rec, depth_cams):
    #     depth_imgs = []
    #     depth_rots = []
    #     depth_trans = []
    #     depth_intrins = []

    #     for depth_cam in depth_cams:
    #         depth_resize_dims, depth_crop = self.sample_augmentation()

    #         depth_samp = self.nusc.get('sample_data', rec['data'][depth_cam])

    #         if not self.use_pre_scaled_imgs:
    #             depth_imgname = os.path.join(self.dataroot, depth_samp['filename'])
    #             depth_img = Image.open(depth_imgname)
    #             depth_W, depth_H = depth_img.size
    #         else:
    #             custom_path = os.path.join(self.custom_dataroot, '#CUSTOM_RES#')  # TODO: adapt to rescaled imgs folder
    #             depth_imgname = os.path.join(custom_path, depth_samp['filename'])
    #             depth_img = Image.open(depth_imgname)
    #             depth_W, depth_H = (1600, 900)  # img.size must be fixed to keep the rest of the code working

    #         depth_sens = self.nusc.get('calibrated_sensor', depth_samp['calibrated_sensor_token'])
    #         depth_intrin = torch.Tensor(depth_sens['camera_intrinsic'])
    #         depth_rot = torch.Tensor(Quaternion(depth_sens['rotation']).rotation_matrix)
    #         depth_tran = torch.Tensor(depth_sens['translation'])

    #         # W and H: original image size to scale intrinsics correctly
    #         sx = depth_resize_dims[0] / float(depth_W)
    #         sy = depth_resize_dims[1] / float(depth_H)

    #         intrin = utils.geom.scale_intrinsics(depth_intrin.unsqueeze(0), sx, sy).squeeze(0)

    #         fx, fy, x0, y0 = utils.geom.split_intrinsics(depth_intrin.unsqueeze(0))

    #         depth_new_x0 = x0 - depth_crop[0]
    #         depth_new_y0 = y0 - depth_crop[1]

    #         depth_pix_T_cam = utils.geom.merge_intrinsics(fx, fy, depth_new_x0, depth_new_y0)
    #         depth_intrin = depth_pix_T_cam.squeeze(0)

    #         depth_img = img_transform(depth_img, depth_resize_dims, depth_crop)


    #         ''' rgb img to depth img transformation  start '''

    #         model_path = "/root/SegMam/monodepth2_on_nuscenes_cityscapes/models/weights_19"

    #         encoder_path = os.path.join(model_path, "encoder.pth")
    #         depth_decoder_path = os.path.join(model_path, "depth.pth")

    #         # LOADING PRETRAINED MODEL
    #         encoder = networks.ResnetEncoder(18, False)
    #         depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    #         loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    #         filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    #         encoder.load_state_dict(filtered_dict_enc)

    #         loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    #         depth_decoder.load_state_dict(loaded_dict)


    #         original_width, original_height = depth_img.size
    #         feed_height = loaded_dict_enc['height']
    #         feed_width = loaded_dict_enc['width']
    #         input_image_resized = depth_img.resize((feed_width, feed_height), pil.LANCZOS)

    #         input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    #         with torch.no_grad():
    #             features = encoder(input_image_pytorch)
    #             outputs = depth_decoder(features)

    #         disp = outputs[("disp", 0)]

    #         #print(f"depth image resize original_height {original_height}") 
    #         #print(f"depth image resize original_width {original_width}") 
            
    #         #depth image resize original_width 896
    #         #depth image resize original_height 448

    #         disp_resized = torch.nn.functional.interpolate(disp,
    #             (original_height, original_width), mode="bilinear", align_corners=False)
            
    #         #disp_resized_np = disp_resized.squeeze().cpu().numpy()
    #         disp_resized_np = disp.squeeze().cpu().numpy()

    #         vmax = np.percentile(disp_resized_np, 95)

    #         # # plt.subplot(212)
    #         # # plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    #         # # plt.title("Disparity prediction", fontsize=22)
    #         # # plt.axis('off')

    #         #save_dir = "/root/data/nuscenes_junhan/pdcms/ogm/camsix/scene_0103/rgb_img/CAM_FRONT"
    #         #os.makedirs(save_dir, exist_ok=True)

    #         # # 저장할 경로 + 파일명 설정
    #         #save_path = os.path.join(save_dir, f"cam.png")

    #         # # 이미지 저장
    #         #plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    #         #plt.axis('off')
    #         #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #         #plt.close()

    #         # disp_resized = torch.nn.functional.interpolate(disp,
    #         #     (original_height, original_width), mode="bilinear", align_corners=False)
            

    #         #depth_imgs.append(totorch_img(depth_img))

    #         #disp.append(totorch_img(disp))
    #         depth_imgs.append(disp_resized_np)
    #         # depth_intrins.append(depth_intrin)
    #         # depth_rots.append(depth_rot)
    #         # depth_trans.append(depth_tran)
            
    #         '''rgb img to depth img transformation  end '''


    #     #return torch.stack(depth_imgs)
    #     return torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in depth_imgs])


    # def get_depth_image_data(self, rec, depth_cams):

    #     depth_imgs = []
    #     depth_rots = []
    #     depth_trans = []
    #     depth_intrins = []

    #     for depth_cam in depth_cams:
    #         depth_resize_dims, depth_crop = self.sample_augmentation()

    #         depth_samp = self.nusc.get('sample_data', rec['data'][depth_cam])

    #         if not self.use_pre_scaled_imgs:
    #             depth_imgname = os.path.join(self.dataroot, depth_samp['filename'])
    #             depth_img = Image.open(depth_imgname)
    #             depth_W, depth_H = depth_img.size
    #         else:
    #             custom_path = os.path.join(self.custom_dataroot, '#CUSTOM_RES#')  # TODO: adapt to rescaled imgs folder
    #             depth_imgname = os.path.join(custom_path, depth_samp['filename'])
    #             depth_img = Image.open(depth_imgname)
    #             depth_W, depth_H = (1600, 900)  # img.size must be fixed to keep the rest of the code working

    #         depth_sens = self.nusc.get('calibrated_sensor', depth_samp['calibrated_sensor_token'])
    #         depth_intrin = torch.Tensor(depth_sens['camera_intrinsic'])
    #         depth_rot = torch.Tensor(Quaternion(depth_sens['rotation']).rotation_matrix)
    #         depth_tran = torch.Tensor(depth_sens['translation'])

    #         # W and H: original image size to scale intrinsics correctly
    #         sx = depth_resize_dims[0] / float(depth_W)
    #         sy = depth_resize_dims[1] / float(depth_H)

    #         intrin = utils.geom.scale_intrinsics(depth_intrin.unsqueeze(0), sx, sy).squeeze(0)

    #         fx, fy, x0, y0 = utils.geom.split_intrinsics(depth_intrin.unsqueeze(0))

    #         depth_new_x0 = x0 - depth_crop[0]
    #         depth_new_y0 = y0 - depth_crop[1]

    #         depth_pix_T_cam = utils.geom.merge_intrinsics(fx, fy, depth_new_x0, depth_new_y0)
    #         depth_intrin = depth_pix_T_cam.squeeze(0)

    #         depth_img = img_transform(depth_img, depth_resize_dims, depth_crop)


    #         ''' rgb img to depth img transformation  start '''

    #         model_path = "/root/SegMam/monodepth2_on_nuscenes_cityscapes/models/weights_19"

    #         encoder_path = os.path.join(model_path, "encoder.pth")
    #         depth_decoder_path = os.path.join(model_path, "depth.pth")

    #         # LOADING PRETRAINED MODEL
    #         encoder = networks.ResnetEncoder(18, False)
    #         depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    #         loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    #         filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    #         encoder.load_state_dict(filtered_dict_enc)

    #         loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    #         depth_decoder.load_state_dict(loaded_dict)


    #         original_width, original_height = depth_img.size
    #         feed_height = loaded_dict_enc['height']
    #         feed_width = loaded_dict_enc['width']
    #         input_image_resized = depth_img.resize((feed_width, feed_height), pil.LANCZOS)

    #         input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    #         with torch.no_grad():
    #             features = encoder(input_image_pytorch)
    #             outputs = depth_decoder(features)

    #         disp = outputs[("disp", 0)]

    #         #print(f"depth image resize original_height {original_height}") 
    #         #print(f"depth image resize original_width {original_width}") 
            
    #         #depth image resize original_width 896
    #         #depth image resize original_height 448

    #         disp_resized = torch.nn.functional.interpolate(disp,
    #             (original_height, original_width), mode="bilinear", align_corners=False)
            
    #         #disp_resized_np = disp_resized.squeeze().cpu().numpy()
    #         disp_resized_np = disp.squeeze().cpu().numpy()

    #         vmax = np.percentile(disp_resized_np, 95)

    #         # # plt.subplot(212)
    #         # # plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    #         # # plt.title("Disparity prediction", fontsize=22)
    #         # # plt.axis('off')

    #         #save_dir = "/root/data/nuscenes_junhan/pdcms/ogm/camsix/scene_0103/rgb_img/CAM_FRONT"
    #         #os.makedirs(save_dir, exist_ok=True)

    #         # # 저장할 경로 + 파일명 설정
    #         #save_path = os.path.join(save_dir, f"cam.png")

    #         # # 이미지 저장
    #         #plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    #         #plt.axis('off')
    #         #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #         #plt.close()

    #         # disp_resized = torch.nn.functional.interpolate(disp,
    #         #     (original_height, original_width), mode="bilinear", align_corners=False)
            

    #         #depth_imgs.append(totorch_img(depth_img))

    #         #disp.append(totorch_img(disp))
    #         depth_imgs.append(disp_resized_np)
    #         # depth_intrins.append(depth_intrin)
    #         # depth_rots.append(depth_rot)
    #         # depth_trans.append(depth_tran)
            
    #         '''rgb img to depth img transformation  end '''


    #     #return torch.stack(depth_imgs)
    #     return torch.stack([torch.from_numpy(img) if isinstance(img, np.ndarray) else img for img in depth_imgs])


    # def get_depth_image_data(self, rec, depth_cams):

    #     depth_imgs = []

    #     for depth_cam in depth_cams:
    #         resize_dims, crop = self.sample_augmentation()

    #         samp = self.nusc.get('sample_data', rec['data'][depth_cam])
    #         img_path = os.path.join(self.dataroot, samp['filename'])
    #         img = Image.open(img_path).convert('RGB')
    #         original_w, original_h = img.size

    #         # Resize to expected input size of MonoDepth2
    #         input_img_resized = img.resize((self.feed_width, self.feed_height), pil.LANCZOS)
    #         input_tensor = transforms.ToTensor()(input_img_resized).unsqueeze(0)

    #         with torch.no_grad():
    #             features = self.encoder(input_tensor)
    #             outputs = self.depth_decoder(features)
    #             disp = outputs[("disp", 0)]

    #         # Resize back to original
    #         disp_resized = torch.nn.functional.interpolate(
    #             disp, (original_h, original_w), mode="bilinear", align_corners=False
    #         ).squeeze(0).squeeze(0).cpu()  # shape: (H, W), move to CPU

    #         depth_imgs.append(disp_resized)
    #         #print(f"depth_imgs shape {depth_imgs.shape}")
    #         #print(f"depth_imgs shape {type(depth_imgs)}")
    #         print(f"depth_imgs shape {depth_imgs[0].shape}") # depth_imgs shape torch.Size([900, 1600])


    #     # for cam_name, file_name in cam_files.items():
    #     #     # Load depth and intrinsics
    #     #     depth = load_depth_image(os.path.join(img_dir, file_name))
    #     #     K = intrinsics_dict[cam_name]
    #     #     points_cam = backproject(depth, K)  # [N, 3]
    #     #     N = points_cam.shape[0]

    #     #     # Convert to homogeneous
    #     #     points_cam_h = np.hstack([points_cam, np.ones((N, 1))])  # [N, 4]

    #     #     # Extrinsic: inverse (camera → ego)
    #     #     T_lidar_to_cam = extrinsics_dict[cam_name]
    #     #     T_cam_to_lidar = np.linalg.inv(np.vstack([T_lidar_to_cam, [0, 0, 0, 1]]))[:3]  # 3x4

    #     #     points_ego = (T_cam_to_lidar @ points_cam_h.T).T  # [N, 3]
    #     #     all_points.append(points_ego)

    #     #     # Merge and visualize
    #     #     merged_points = np.vstack(all_points)
    #     #     print(f"merged_points.shape: {merged_points.shape}")

    #     #     # pcd = o3d.geometry.PointCloud()
    #     #     # pcd.points = o3d.utility.Vector3dVector(merged_points)
    #     #     # o3d.visualization.draw_geometries([pcd])

    #     return torch.stack(depth_imgs)


    def get_only_image_data(self, rec, cams):

        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            resize_dims, crop = self.sample_augmentation()

            samp = self.nusc.get('sample_data', rec['data'][cam])

            if not self.use_pre_scaled_imgs:
                imgname = os.path.join(self.dataroot, samp['filename'])
                img = Image.open(imgname)
                W, H = img.size
            else:
                custom_path = os.path.join(self.custom_dataroot, '#CUSTOM_RES#')  # TODO: adapt to rescaled imgs folder
                imgname = os.path.join(custom_path, samp['filename'])
                img = Image.open(imgname)
                W, H = (1600, 900)  # img.size must be fixed to keep the rest of the code working

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # W and H: original image size to scale intrinsics correctly
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            intrin = utils.geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = utils.geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = utils.geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = img_transform(img, resize_dims, crop)

            imgs.append(totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

        return torch.stack(imgs)


    # depth default 
    # def get_depth_image_data(self, rec, depth_cams):
    #     depth_imgs = []
    #     depth_rots = []
    #     depth_trans = []
    #     depth_intrins = []

    #     for depth_cam in depth_cams:
    #         depth_resize_dims, depth_crop = self.sample_augmentation()

    #         depth_samp = self.nusc.get('sample_data', rec['data'][depth_cam])

    #         if not self.use_pre_scaled_imgs:
    #             depth_imgname = os.path.join(self.dataroot, depth_samp['filename'])
    #             depth_img = Image.open(depth_imgname)
    #             depth_W, depth_H = depth_img.size
    #         else:
    #             custom_path = os.path.join(self.custom_dataroot, '#CUSTOM_RES#')  # TODO: adapt to rescaled imgs folder
    #             depth_imgname = os.path.join(custom_path, depth_samp['filename'])
    #             depth_img = Image.open(depth_imgname)
    #             depth_W, depth_H = (1600, 900)  # img.size must be fixed to keep the rest of the code working

    #         depth_sens = self.nusc.get('calibrated_sensor', depth_samp['calibrated_sensor_token'])
    #         depth_intrin = torch.Tensor(depth_sens['camera_intrinsic'])
    #         depth_rot = torch.Tensor(Quaternion(depth_sens['rotation']).rotation_matrix)
    #         depth_tran = torch.Tensor(depth_sens['translation'])

    #         # W and H: original image size to scale intrinsics correctly
    #         sx = depth_resize_dims[0] / float(depth_W)
    #         sy = depth_resize_dims[1] / float(depth_H)

    #         intrin = utils.geom.scale_intrinsics(depth_intrin.unsqueeze(0), sx, sy).squeeze(0)

    #         fx, fy, x0, y0 = utils.geom.split_intrinsics(depth_intrin.unsqueeze(0))

    #         depth_new_x0 = x0 - depth_crop[0]
    #         depth_new_y0 = y0 - depth_crop[1]

    #         depth_pix_T_cam = utils.geom.merge_intrinsics(fx, fy, depth_new_x0, depth_new_y0)
    #         depth_intrin = depth_pix_T_cam.squeeze(0)

    #         depth_img = img_transform(depth_img, depth_resize_dims, depth_crop)




    #         '''----rgb img to depth img transformation  start '''

    #         model_path = "/root/tmp/mdp/models/weights_9"

    #         encoder_path = os.path.join(model_path, "encoder.pth")
    #         depth_decoder_path = os.path.join(model_path, "depth.pth")

    #         # LOADING PRETRAINED MODEL
    #         encoder = networks.ResnetEncoder(18, False)
    #         depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    #         loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    #         #input_image = pil.open(depth_imgs).convert('RGB')
    #         #input_image = depth_imgs.convert('RGB')
    #         #input_image = [depth_img.convert('RGB') for depth_img in depth_imgs]

    #         original_width, original_height = depth_img.size
    #         feed_height = loaded_dict_enc['height']
    #         feed_width = loaded_dict_enc['width']
    #         input_image_resized = depth_img.resize((feed_width, feed_height), pil.LANCZOS)

    #         input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

    #         with torch.no_grad():
    #             features = encoder(input_image_pytorch)
    #             outputs = depth_decoder(features)

    #         disp_list = []
    #         disp = outputs[("disp", 0)]

    #         disp_resized = torch.nn.functional.interpolate(disp,
    #             (original_height, original_width), mode="bilinear", align_corners=False)
            

    #         #depth_imgs.append(totorch_img(depth_img))

    #         #disp.append(totorch_img(disp))
    #         disp_list.append(disp_resized)
    #         depth_intrins.append(depth_intrin)
    #         depth_rots.append(depth_rot)
    #         depth_trans.append(depth_tran)
            
    #         '''rgb img to depth img transformation  end '''


    #     return torch.stack(disp_list)

    ''' depth pcd default start'''
    # def get_depth_image_data(self, rec, depth_cams):
    #     """
    #     각 카메라에 대해 RGB 이미지를 기반으로 MonoDepth2로 depth map 생성,
    #     이를 point cloud로 backproject하여 최종 (N, 3) 형태의 numpy array로 반환.
    #     """

    #     all_points = []

    #     for depth_cam in depth_cams:
    #         # 1. 데이터 정보 가져오기
    #         #resize_dims, crop = self.sample_augmentation()
    #         samp = self.nusc.get('sample_data', rec['data'][depth_cam])
    #         img_path = os.path.join(self.dataroot, samp['filename'])
    #         img = Image.open(img_path).convert('RGB')
    #         original_w, original_h = img.size

    #         input_img_resized = img.resize((self.feed_width, self.feed_height), Image.LANCZOS)
    #         input_tensor = transforms.ToTensor()(input_img_resized).unsqueeze(0)

    #         with torch.no_grad():
    #             features = self.encoder(input_tensor)
    #             outputs = self.depth_decoder(features)
    #             disp = outputs[("disp", 0)]  # shape: (1, 1, H, W)

    #         disp_resized = torch.nn.functional.interpolate(
    #             disp, (original_h, original_w), mode="bilinear", align_corners=False
    #         ).squeeze().cpu().numpy()  # shape: (H, W)

    #         depth_map = 1.0 / np.maximum(disp_resized, 1e-6)  # shape: (H, W)

    #         K = self.intrinsics_dict[depth_cam]  # shape: (3, 3)
    #         fx, fy = K[0, 0], K[1, 1]
    #         cx, cy = K[0, 2], K[1, 2]

    #         h, w = depth_map.shape
    #         u, v = np.meshgrid(np.arange(w), np.arange(h))
    #         u = u.reshape(-1)
    #         v = v.reshape(-1)
    #         d = depth_map.reshape(-1)

    #         x = (u - cx) * d / fx
    #         y = (v - cy) * d / fy
    #         z = d
    #         points_cam = np.stack([x, y, z], axis=1)  # shape: (N, 3)

    #         points_cam_h = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)  # (N, 4)

    #         T_lidar_to_cam = self.extrinsics_dict[depth_cam]  # shape: (3, 4) or (4, 4)
    #         if T_lidar_to_cam.shape == (3, 4):
    #             T_lidar_to_cam = np.vstack([T_lidar_to_cam, [0, 0, 0, 1]])
    #         T_cam_to_lidar = np.linalg.inv(T_lidar_to_cam)  # 4x4

    #         points_ego = (T_cam_to_lidar @ points_cam_h.T).T[:, :3]  # shape: (N, 3)

    #         dists = np.linalg.norm(points_ego, axis=1)
    #         mask = (dists > 2.0) & (dists < 50.0)
    #         points_ego = points_ego[mask]

    #         all_points.append(points_ego)

    #     merged_points = np.vstack(all_points)  # shape: (N_total, 3)
    #     voxel_size = 0.5 # ex) 0.5m voxel로 downsampling
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(merged_points)
    #     pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    #     downsampled_points = np.asarray(pcd_down.points)
    #     downsampled_points = torch.from_numpy(downsampled_points)
    #     downsampled_points = downsampled_points.permute(1,0)

    #     #(19, 2061)   pseudolidar data 

    #     #downsampled_points shape: (30006, 3)

    #     print(f"downsampled_points shape: {downsampled_points.shape}")

    #     #return torch.from_numpy(downsampled_points).float()
    #     # print(f"merged_points shpae{ merged_points.shape}")
    #     # return torch.Tensor(merged_points)

    #     #RuntimeError: mat1 and mat2 shapes cannot be multiplied (35000x4 and 7x16)
    #     return downsampled_points

    ''' depth pcd default end'''

    def _scale_depth_with_lidar(self, rec, cam, depth_map, K, max_samples=80000):
        """
        depth_map(H,W)을 동일 프레임의 LIDAR_TOP으로 보정해 절대 스케일을 맞춘다.
        """
        sd_cam   = self.nusc.get('sample_data', rec['data'][cam])
        sd_lidar = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])

        cam_pose   = self.nusc.get('ego_pose', sd_cam['ego_pose_token'])
        cam_cs     = self.nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        lidar_pose = self.nusc.get('ego_pose', sd_lidar['ego_pose_token'])
        lidar_cs   = self.nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])

        # LiDAR 로드 & 샘플링
        pc = LidarPointCloud.from_file(os.path.join(self.dataroot, sd_lidar['filename']))
        if pc.points.shape[1] > max_samples:
            idx = np.random.choice(pc.points.shape[1], max_samples, replace=False)
            pc.points = pc.points[:, idx]

        # lidar -> ego(lidar) -> global -> ego(cam) -> cam
        T_lidar_to_ego   = transform_matrix(lidar_cs['translation'], Quaternion(lidar_cs['rotation']), inverse=False)
        T_ego_to_global  = transform_matrix(lidar_pose['translation'], Quaternion(lidar_pose['rotation']), inverse=False)
        T_global_to_egoC = transform_matrix(cam_pose['translation'],  Quaternion(cam_pose['rotation']),  inverse=True)
        T_egoC_to_cam    = transform_matrix(cam_cs['translation'],    Quaternion(cam_cs['rotation']),    inverse=True)
        T = T_egoC_to_cam @ T_global_to_egoC @ T_ego_to_global @ T_lidar_to_ego  # 4x4

        pts_cam = T @ np.vstack([pc.points[:3, :], np.ones((1, pc.points.shape[1]))])  # (4,N)
        x, y, z = pts_cam[0], pts_cam[1], pts_cam[2]
        valid = z > 0.1
        x, y, z = x[valid], y[valid], z[valid]

        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u = (fx * (x / z) + cx).astype(np.int32)
        v = (fy * (y / z) + cy).astype(np.int32)

        H, W = depth_map.shape
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if m.sum() < 200:
            return depth_map

        z_true = z[m]
        z_pred = depth_map[v[m], u[m]]

        r = z_true / np.clip(z_pred, 1e-6, None)
        r = r[np.isfinite(r)]
        r = np.clip(r, 0.2, 5.0)  # 이상치 컷
        if r.size == 0:
            return depth_map

        s = np.median(r)
        return depth_map * s


    '''start test '''
    def get_depth_image_data(self, rec, depth_cams, 
                         min_depth=0.5, max_depth=50.0,
                         voxel_size=0.5,  # downsample 간격(m)
                         stride=8):       # 픽셀 샘플링 간격(속도/메모리)
        """
        - MonoDepth2로 예측한 disparity를 원본 해상도로 리사이즈
        - NuScenes calibrated_sensor에서 카메라 intrinsics(K)와 cam->ego 외부 파라미터 사용
        - 6대 카메라 포인트를 전부 ego frame으로 합치고 voxel downsample
        - return: (3, N) torch.FloatTensor  [x,y,z] in ego
        """
        all_pts_ego = []

        for cam in depth_cams:
            sd = self.nusc.get('sample_data', rec['data'][cam])

            # 1) RGB 로드 + MonoDepth2 추론
            img_path = os.path.join(self.dataroot, sd['filename'])
            img = Image.open(img_path).convert('RGB')
            W, H = img.size

            with torch.no_grad():
                in_tensor = transforms.ToTensor()(
                    img.resize((self.feed_width, self.feed_height), Image.LANCZOS)
                ).unsqueeze(0)
                disp = self.depth_decoder(self.encoder(in_tensor))[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (H, W), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()

            # --- depth map 생성
            depth_map = 1.0 / np.maximum(disp_resized, 1e-6)
            depth_map = np.clip(depth_map, min_depth, max_depth)

            # 2) 카메라 intrinsics / cam->ego 외부 파라미터 from NuScenes
            cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            K = np.asarray(cs['camera_intrinsic'])      # 3x3
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            T_cam2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)

            # --- LiDAR로 스케일 정렬
            depth_map = self._scale_depth_with_lidar(rec, cam, depth_map, K)

            # 3) back-projection (stride 샘플링)
            u = np.arange(0, W, stride)
            v = np.arange(0, H, stride)
            uu, vv = np.meshgrid(u, v)
            d = depth_map[vv, uu]

            x = (uu - cx) * d / fx
            y = (vv - cy) * d / fy
            z = d
            pts_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # (N,3)

            # 4) cam -> ego
            pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
            pts_ego = (T_cam2ego @ pts_cam_h.T).T[:, :3]

            # 5) 필터링
            dist = np.linalg.norm(pts_ego[:, :2], axis=1)
            valid = (pts_ego[:, 2] > 0.1) & (dist < max_depth)
            if np.any(valid):
                all_pts_ego.append(pts_ego[valid])

        if len(all_pts_ego) == 0:
            return torch.zeros(3, 0, dtype=torch.float32)

        merged = np.concatenate(all_pts_ego, axis=0)

        # 6) voxel downsample
        if voxel_size and merged.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            merged = np.asarray(pcd.points)

        point = torch.from_numpy(merged).float()
        #print(f"point {point.shape}")

        return point  # (3, N)


    '''start end '''


    def get_pseudolidar_data(self, rec, nsweeps):
        pts = get_pseudolidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2,
                             use_pseudolidar_filters=self.use_pseudolidar_filters, dataroot=self.dataroot)
        #pts = get_dept

        # print(f"pseudolidar data {pts.shape}") # pseudolidar data (19, 2061)   pseudolidar data 
        # rada datar integration ehotj skdhsek.

        #print(f"pseudolidar type {pts.type}")


        return torch.Tensor(pts)

    def get_binimg(self, rec):
        
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        
        transformate = trans
        rotation = -np.array(egopose['rotation'])
        
        
        img = np.zeros((self.nx[0], self.nx[1]))
        for ii, tok in enumerate(rec['anns']):
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if discard_invisible and int(inst['visibility_token']) == 1:
                # filter invisible vehicles
                continue

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            # noinspection PyUnresolvedReferences
            cv2.fillPoly(img, [pts], ii + 1.0)

        return torch.Tensor(img).unsqueeze(0), torch.Tensor(convert_egopose_to_matrix_numpy(egopose)), transformate, rotation

    def get_seg_bev(self, lrtlist_cam, vislist):
        B, N, D = lrtlist_cam.shape
        assert (B == 1)

        seg = np.zeros((self.Z, self.X))
        val = np.ones((self.Z, self.X))

        corners_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist_cam)  # B, N, 8, 3
        y_cam = corners_cam[:, :, :, 1]  # y part; B, N, 8
        corners_mem = self.vox_util.Ref2Mem(corners_cam.reshape(B, N * 8, 3), self.Z, self.Y, self.X).reshape(B, N, 8,
                                                                                                              3)

        # take the xz part
        corners_mem = torch.stack([corners_mem[:, :, :, 0], corners_mem[:, :, :, 2]], dim=3)  # B, N, 8, 2
        # corners_mem = corners_mem[:,:,:4] # take the bottom four

        for n in range(N):
            _, inds = torch.topk(y_cam[0, n], 4, largest=False)  # returns indices of the 4 smallest values
            pts = corners_mem[0, n, inds].numpy().astype(np.int32)  # 4, 2

            # if this messes in some later conditions,
            # the solution is to draw all combos
            pts = np.stack([pts[0], pts[1], pts[3], pts[2]])

            # pts[:, [1, 0]] = pts[:, [0, 1]]
            # noinspection PyUnresolvedReferences
            cv2.fillPoly(seg, [pts], n + 1.0)

            if vislist[n] == 0:
                # draw a black rectangle if it's invisible
                # noinspection PyUnresolvedReferences
                cv2.fillPoly(val, [pts], 0.0)

        return torch.Tensor(seg).unsqueeze(0), torch.Tensor(val).unsqueeze(0)  # 1, Z, X

    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        #print(f'transtranstrans {trans}')
        
        global trans_nusce 
        global rote_nusce
        
        trans_nusce = trans
        rote_nusce = rot

        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if int(inst['visibility_token']) == 1:
                vislist.append(torch.tensor(0.0))  # invisible
            else:
                vislist.append(torch.tensor(1.0))  # visible

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            tidlist.append(inst['instance_token'])

            # print('rotation', inst['rotation'])
            r = box.rotation_matrix
            t = box.center
            l = box.wlh
            l = np.stack([l[1], l[0], l[2]])
            lrt = utils.py.merge_lrt(l, utils.py.merge_rt(r, t))
            lrt = torch.Tensor(lrt)
            lrtlist.append(lrt)
            ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
            rs = np.stack([ry * 0, ry, ry * 0])
            box_ = torch.from_numpy(np.stack([t, l, rs])).reshape(9)

            boxlist.append(box_)
        if len(lrtlist):
            lrtlist = torch.stack(lrtlist, dim=0)
            boxlist = torch.stack(boxlist, dim=0)
            vislist = torch.stack(vislist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros(0)
            tidlist = []

        return lrtlist, boxlist, vislist, tidlist

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.indices)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

         # DEPTH PARAM START

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #model_path = "/root/SegMam/monodepth2_on_nuscenes_cityscapes/models/weights_17"
        model_path = "/SegMam/monodepth2_on_nuscenes_cityscapes/models/mono_640x192"
        encoder_path = os.path.join(model_path, "encoder.pth")
        decoder_path = os.path.join(model_path, "depth.pth")

        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # Load weights
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()

        self.feed_width = loaded_dict_enc['width']
        self.feed_height = loaded_dict_enc['height']


        # DEPTH PARAM END

        # Depth PCD Transformation START

        # Intrinsic matrices from P2 line (CAM_X.txt)

        # self.intrinsics_dict = {
        #     'CAM_FRONT': np.array([[1266.4172, 0, 816.2670],
        #                         [0, 1266.4172, 491.5071],
        #                         [0, 0, 1]]),
        #     'CAM_FRONT_LEFT': np.array([[1272.598, 0, 826.615],
        #                                 [0, 1272.598, 479.752],
        #                                 [0, 0, 1]]),
        #     'CAM_FRONT_RIGHT': np.array([[1260.847, 0, 807.968],
        #                                 [0, 1260.847, 495.334],
        #                                 [0, 0, 1]]),
        #     'CAM_BACK': np.array([[809.2209, 0, 829.2196],
        #                         [0, 809.2209, 481.7784],
        #                         [0, 0, 1]]),
        #     'CAM_BACK_LEFT': np.array([[1256.741, 0, 792.1126],
        #                             [0, 1256.741, 492.7757],
        #                             [0, 0, 1]]),
        #     'CAM_BACK_RIGHT': np.array([[1259.514, 0, 807.253],
        #                                 [0, 1259.514, 501.196],
        #                                 [0, 0, 1]]),
        # }

        # # Extrinsics (Tr_velo_to_cam) from CAM_X.txt files: shape (3, 4)
        # self.extrinsics_dict = {
        #     'CAM_FRONT': np.array([[3.487968666398e-03, -9.999708566009e-01, 6.791172464157e-03, 1.190663537703e-02],
        #                         [1.859214393651e-02, -6.725192192724e-03, -9.998045328832e-01, -3.249862680961e-01],
        #                         [9.998210671207e-01, 3.613549339171e-03, 1.856814483859e-02, -7.590020378669e-01]]),
        #     'CAM_FRONT_LEFT': np.array([[8.193239137146e-01, -5.729418068352e-01, 2.111895820877e-02, -1.943077487789e-01],
        #                                 [2.389490014534e-02, -2.679485566266e-03, -9.997108852584e-01, -3.322944212408e-01],
        #                                 [5.728327488565e-01, 8.195916704906e-01, 1.149502066252e-02, -7.365951860513e-01]]),
        #     'CAM_FRONT_RIGHT': np.array([[-8.336674597099e-01, -5.516540059106e-01, -2.600815994197e-02, 2.295777549385e-01],
        #                                 [2.425179964555e-02, 1.047965861696e-02, -9.996509525676e-01, -3.363483595028e-01],
        #                                 [5.517340091337e-01, -8.340072149076e-01, 4.642052047181e-03, -7.515933625071e-01]]),
        #     'CAM_BACK': np.array([[4.693549526940e-03, 9.999410727305e-01, -9.788853838851e-03, -2.279094218707e-03],
        #                         [7.468503624160e-03, -9.823741143322e-03, -9.999238548827e-01, -2.764102127844e-01],
        #                         [-9.999610952664e-01, 4.620084045689e-03, -7.514171741229e-03, -9.108921364296e-01]]),
        #     'CAM_BACK_LEFT': np.array([[9.480828274767e-01, 3.170500237552e-01, 2.486432546145e-02, -2.408677462459e-01],
        #                             [3.285883117351e-02, -1.989249958769e-02, -9.992620205302e-01, -2.432475606792e-01],
        #                             [-3.163214337628e-01, 9.482001744869e-01, -2.927763045881e-02, -4.340852490002e-01]]),
        #     'CAM_BACK_RIGHT': np.array([[-9.334645808774e-01, 3.569045527253e-01, -3.553894330656e-02, 2.341589313129e-01],
        #                                 [4.012061602384e-02, 5.439469131878e-03, -9.991800380039e-01, -2.734470771671e-01],
        #                                 [-3.564185915708e-01, -9.341250196946e-01, -1.939678229044e-02, -4.284840865569e-01]]),
        # }


        # # Image paths
        # self.cam_files = {
        #     'CAM_FRONT': "cam0_rgb_000.png",
        #     'CAM_FRONT_LEFT': "cam1_rgb_000.png",
        #     'CAM_FRONT_RIGHT': "cam2_rgb_000.png",
        #     'CAM_BACK': "cam3_rgb_000.png",
        #     'CAM_BACK_LEFT': "cam4_rgb_000.png",
        #     'CAM_BACK_RIGHT': "cam5_rgb_000.png",
        # }

        # self.img_dir = "/home/junhan/Documents/mm/pcd_test"

        self.all_points = []


        # Depth PCD Transformation END


        Z, Y, X = self.res_3d
        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=torch.from_numpy(self.centroid).float().cuda(),
            bounds=self.bounds,
            assert_cube=False)
        self.Z, self.Y, self.X = Z, Y, X
        
        self.egocar = EgoPose_Class()

    
    # PCD TRANSFORM FUNCTION START

    def load_depth_image(path):
        depth_img = Image.open(path).convert('L')
        depth_arr = np.array(depth_img).astype(np.float32) / 255.0
        return depth_arr * 50.0  # 0~50m range

    def backproject(depth, K):
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        pixels_hom = np.stack((i, j, np.ones_like(i)), axis=-1).reshape(-1, 3).T
        depth_flat = depth.reshape(-1)
        cam_points = np.linalg.inv(K) @ pixels_hom * depth_flat
        return cam_points.T

    # PCD TRANSFORM FUNCTION END


       
        
    def get_single_item(self, index, cams, refcam_id: int = None):
        
        rec = self.ixes[index]
        imgs, rots, trans, intrins = self.get_image_data(rec, cams)
        binimg, egopose ,transformate, rotation = self.get_binimg(rec)

        '''depth image'''

        #depth_img = self.get_depth_image_data(rec, cams)
        depth_img = self.get_only_image_data(rec, cams)
        

  
        if refcam_id is None:
            if self.is_train:
                # randomly sample the ref cam
                refcam_id = np.random.randint(1, len(cams))
            else:
                refcam_id = self.refcam_id

        # move the target refcam_id to the zeroth slot
        img_ref = imgs[refcam_id].clone()
        img_0 = imgs[0].clone()
        imgs[0] = img_ref
        imgs[refcam_id] = img_0

        rot_ref = rots[refcam_id].clone()
        rot_0 = rots[0].clone()
        rots[0] = rot_ref
        rots[refcam_id] = rot_0

        tran_ref = trans[refcam_id].clone()
        tran_0 = trans[0].clone()
        trans[0] = tran_ref
        trans[refcam_id] = tran_0

        intrin_ref = intrins[refcam_id].clone()
        intrin_0 = intrins[0].clone()
        intrins[0] = intrin_ref
        intrins[refcam_id] = intrin_0

        # get map for current scene
        scene2map = {}

        scene_token = rec['scene_token']
        scene = self.nusc.get('scene', scene_token)
        log = self.nusc.get('log', scene['log_token'])
        scene2map[scene['name']] = log['location']

        car_from_current = np.eye(4)
        car_from_current[:3, :3] = rots[0].numpy()
        car_from_current[:3, 3] = np.transpose(trans[0].numpy())

        map_mask, egocar_bev = fetch_nusc_map2(rec, self.nusc_maps, self.nusc, scene2map, car_from_current)

        bev_map = get_rgba_map_from_mask2(map_masks=map_mask, threshold=0.8, a=1.0)  # a = 0.8
        bev_map_mask = torch.Tensor(map_mask)
        egocar_bev_tensor = torch.Tensor(egocar_bev).unsqueeze(0)

        # ---------------------------------------------------------------------------------

        # pseudolidar data handling
        #pseudolidar_data = self.get_pseudolidar_data(rec, nsweeps=self.nsweeps)
        '''get depth pcd '''
        pseudolidar_data = self.get_depth_image_data(rec, cams)

        lrtlist_, boxlist_, vislist_, tidlist_ = self.get_lrtlist(rec)
        N_ = lrtlist_.shape[0]

        # import ipdb; ipdb.set_trace()
        # go through all moving objects in the current sample (here: cars only)
        if N_ > 0:

            velo_T_cam = utils.geom.merge_rt(rots, trans)
            cam_T_velo = utils.geom.safe_inverse(velo_T_cam)

            # note we index 0:1, since we already put refcam into zeroth position
            lrtlist_cam = utils.geom.apply_4x4_to_lrt(cam_T_velo[0:1].repeat(N_, 1, 1), lrtlist_).unsqueeze(0)

            seg_bev, valid_bev = self.get_seg_bev(lrtlist_cam, vislist_)
        else:
            seg_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            valid_bev = torch.ones((1, self.Z, self.X), dtype=torch.float32)

        N = 150  # i've seen n as high as 103 before, so 150 is probably safe (max number of objects)
        lrtlist = torch.zeros((N, 19), dtype=torch.float32)
        vislist = torch.zeros(N, dtype=torch.float32)
        scorelist = torch.zeros(N, dtype=torch.float32)
        lrtlist[:N_] = lrtlist_
        vislist[:N_] = vislist_
        scorelist[:N_] = 1

        
        #pseudolidar has <700 points
        # pseudolidar_data = np.transpose(pseudolidar_data)

        
        # V = 10000 * self.nsweeps  # if nsweeps = 5 -> V=3500
        # if pseudolidar_data.shape[0] > V:
        #     print('pseudolidar_data', pseudolidar_data.shape)
        #     print('max pts', V)
        #     assert False, "Way more pseudolidar returns than expected"
        #     # pseudolidar_data = pseudolidar_data[:V]  # fix upper bound of number of pseudolidar readings
        # elif pseudolidar_data.shape[0] < V:
        #     pad = torch.zeros(3, V - N, dtype=pseudolidar_data.dtype)
        #     pseudolidar_data = torch.cat([pseudolidar_data, pad], dim=1)
        #     #pseudolidar_data = np.pad(pseudolidar_data, [(0, V - pseudolidar_data.shape[0]), (0, 0)], mode='constant')

        
        #pseudolidar_data = np.transpose(pseudolidar_data)
        #pseudolidar_data = torch.from_numpy(pseudolidar_data).float()


        # 1) 반드시 torch 텐서로 유지
        if isinstance(pseudolidar_data, np.ndarray):
            pseudolidar_data = torch.from_numpy(pseudolidar_data)

        # 2) 모양을 (3, P)로 통일
        if pseudolidar_data.dim() != 2 or min(pseudolidar_data.shape) == 0:
            raise RuntimeError(f"Unexpected pseudolidar_data shape: {pseudolidar_data.shape}")
        if pseudolidar_data.shape[1] == 3:           # (P,3) -> (3,P)
            pseudolidar_data = pseudolidar_data.t()
        elif pseudolidar_data.shape[0] == 3:         # 이미 (3,P)
            pass
        else:
            raise RuntimeError(f"Expected shape (?,3) or (3,?), got {pseudolidar_data.shape}")

        # 3) 길이 정규화: 잘라내거나 패딩해서 (3, V)
        V = 10000 * self.nsweeps   # 원하는 타겟 길이(포인트 수). 필요하면 하이퍼파라미터로 빼세요.
        P = pseudolidar_data.shape[1]

        if P > V:
            # 학습에선 랜덤 샘플링, 평가에선 앞에서 자르기 (원하는 정책으로)
            if self.is_train:
                idx = torch.randperm(P, device=pseudolidar_data.device)[:V]
                pseudolidar_data = pseudolidar_data[:, idx]
            else:
                pseudolidar_data = pseudolidar_data[:, :V]
        elif P < V:
            pad = torch.zeros(3, V - P, dtype=pseudolidar_data.dtype, device=pseudolidar_data.device)
            pseudolidar_data = torch.cat([pseudolidar_data, pad], dim=1)
        # 이제 pseudolidar_data는 (3, V)

        binimg = (binimg > 0).float()
        seg_bev = (seg_bev > 0).float()

        # if use_obj_layer_on_map --> if we want to use the obj seg as another additional layer in the map branch
        if self.use_obj_layer_only_on_map:
            seg_valid_bev = valid_bev * seg_bev
            bev_map_mask = torch.cat((bev_map_mask, seg_valid_bev), dim=0)

        # if pseudolidar data needs preprocessing
        if self.pseudolidar_encoder_type is not None:
            rad_data = pseudolidar_data.permute(1, 0)  # R, 19
            xyz_rad = rad_data[:, :3]
            meta_rad = rad_data[:, 3:]
            if self.use_shallow_metadata:
                shallow_meta_rad = rad_data[:, 5:8]
                meta_rad = shallow_meta_rad
            velo_T_cams = utils.geom.merge_rt(rots, trans)  # (6,4,4)
            cams_T_velo = utils.geom.safe_inverse(velo_T_cams)  # (6,4,4)
            cams_T_velo = cams_T_velo.unsqueeze(dim=0)
            xyz_rad = xyz_rad.unsqueeze(dim=0)
            meta_rad = meta_rad.unsqueeze(dim=0)
            rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_rad)

            if self.pseudolidar_encoder_type == "voxel_net":
                voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels = \
                    self.vox_util.voxelize_xyz_and_feats_voxelnet(rad_xyz_cam0, meta_rad, self.Z, self.Y, self.X,
                                                                  assert_cube=False,
                                                                  use_pseudolidar_occupancy_map=self.use_pseudolidar_occupancy_map)

                voxel_input_feature_buffer = voxel_input_feature_buffer.squeeze(dim=0)
                voxel_coordinate_buffer = voxel_coordinate_buffer.squeeze(dim=0)
                number_of_occupied_voxels = number_of_occupied_voxels.squeeze(dim=0)
                
                return imgs, rots, trans, intrins, seg_bev, valid_bev, \
                    pseudolidar_data, bev_map_mask, bev_map, \
                    egocar_bev_tensor, voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels ,transformate, rotation,\
                    depth_img
                
        else:
            return imgs, rots, trans, intrins, seg_bev, valid_bev, pseudolidar_data,\
                bev_map_mask, bev_map, egocar_bev_tensor, transformate, rotation,\
                depth_img
        
    

    def __getitem__(self, index):
        cams = self.choose_cams()

        if self.is_train and self.do_shuffle_cams:
            # randomly sample the ref cam
            if len(cams) > 1:
                refcam_id = np.random.randint(1, len(cams))
            else:
                refcam_id = 0
        else:
            refcam_id = self.refcam_id
        
        
        # trans_use = []
        # rots_use = []
        
        all_transformate = []
        all_rotation = []
        
        all_imgs = []
        all_rots = []
        all_trans = []
        all_intrins = []
        all_seg_bev = []
        all_valid_bev = []
        all_pseudolidar_data = []
        # added bev_map_gt
        all_bev_map_mask = []
        all_bev_map = []
        # added egocar in bev plane
        all_egocar_bev_tensors = []
        # Voxnet preprocessing
        all_voxel_input_feature_buffer = []
        all_voxel_coordinate_buffer = []
        all_number_of_occupied_voxels = []


        all_depth_img = [] 

        if self.vis_full_scenes:
            samples = self.indices[index]
            samples = samples[1:(samples[0] + 1)]
            print(f'samples{len(samples)}')
        else:
            samples = self.indices[index]
        
       
        for index_t in samples:
            if self.pseudolidar_encoder_type == "voxel_net":
                # voxelnet
                imgs, rots, trans, intrins, seg_bev, valid_bev, pseudolidar_data, \
                    bev_map_mask, bev_map, egocar_bev_tensor, voxel_input_feature_buffer, \
                    voxel_coordinate_buffer, number_of_occupied_voxels ,transformate, rotation, depth_img = \
                    self.get_single_item(index_t, cams, refcam_id=refcam_id)

                
                all_voxel_input_feature_buffer.append(voxel_input_feature_buffer)
                all_voxel_coordinate_buffer.append(voxel_coordinate_buffer)
                all_number_of_occupied_voxels.append(number_of_occupied_voxels)

            else:
                # default
                imgs, rots, trans, intrins, seg_bev, valid_bev, pseudolidar_data,\
                    bev_map_mask, bev_map, egocar_bev_tensor ,transformate, rotation, depth_img = \
                    self.get_single_item(index_t, cams, refcam_id=refcam_id)
                    
            transformate = torch.from_numpy(transformate)
            # np.array(rotation)
            rotation = torch.from_numpy(rotation)
            
            all_transformate.append(transformate)
            all_rotation.append(rotation)
            all_imgs.append(imgs)
            all_rots.append(rots)
            all_trans.append(trans)
            all_intrins.append(intrins)
            all_seg_bev.append(seg_bev)
            all_valid_bev.append(valid_bev)
            all_pseudolidar_data.append(pseudolidar_data)
            # added bev_map_gt
            all_bev_map_mask.append(bev_map_mask)
            all_bev_map.append(bev_map)
            # added egocar in bev plane
            all_egocar_bev_tensors.append(egocar_bev_tensor)

            '''dept'''
            all_depth_img.append(depth_img)
            
        

        all_transformate = torch.stack(all_transformate)
        all_rotation = torch.stack(all_rotation)
        
        all_imgs = torch.stack(all_imgs)
        all_rots = torch.stack(all_rots)
        all_trans = torch.stack(all_trans)
        all_intrins = torch.stack(all_intrins)
        all_seg_bev = torch.stack(all_seg_bev)
        all_valid_bev = torch.stack(all_valid_bev)
        
        '''pseudolidar data'''
        all_pseudolidar_data = torch.stack(all_pseudolidar_data)
        #all_pseudolidar_data = torch.cat(all_pseudolidar_data, dim=0)  # -> [sum_i N_i, 17500]


        # added bev_map_gt
        all_bev_map_mask = torch.stack(all_bev_map_mask)
        all_bev_map = torch.stack(all_bev_map)
        # added egocar in bev plane
        all_egocar_bev_tensors = torch.stack(all_egocar_bev_tensors)

        '''depth'''
        all_depth_img = torch.stack(all_depth_img)

        
        
        if self.pseudolidar_encoder_type == "voxel_net":
            # Voxnet preprocessing
            all_voxel_input_feature_buffer = torch.stack(all_voxel_input_feature_buffer)
            all_voxel_coordinate_buffer = torch.stack(all_voxel_coordinate_buffer)
            all_number_of_occupied_voxels = torch.stack(all_number_of_occupied_voxels)
            

            return all_imgs, all_rots, all_trans, all_intrins, \
                all_seg_bev, all_valid_bev, all_pseudolidar_data, \
                all_bev_map_mask, all_bev_map, all_egocar_bev_tensors, all_voxel_input_feature_buffer, \
                all_voxel_coordinate_buffer, all_number_of_occupied_voxels , all_transformate, all_rotation, all_depth_img
        else:
            return all_imgs, all_rots, all_trans, all_intrins, \
                all_seg_bev, all_valid_bev, all_pseudolidar_data, \
                all_bev_map_mask, all_bev_map, all_egocar_bev_tensors ,all_transformate, all_rotation, all_depth_img


def worker_rnd_init(x):
    np.random.seed(13 + x)


def my_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    # print("Worker seed: ", worker_seed, " Worker_id: ", worker_id)

    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


# def compile_data(version, dataroot, data_aug_conf, centroid, bounds, res_3d, bsz,
#                  nworkers, pseudolidar_encoder_type=None, shuffle=True, nsweeps=1, nworkers_val=1, seqlen=1, refcam_id=1,
#                  get_tids=False, temporal_aug=False, use_pseudolidar_filters=False, use_shallow_metadata=True,
#                  do_shuffle_cams=True, distributed_sampler=False, rank=None, use_pre_scaled_imgs=True,
#                  custom_dataroot=None, use_obj_layer_only_on_map=False, vis_full_scenes=False,
#                  use_pseudolidar_occupancy_map=False, do_drn_val_split=False, get_val_day=False, get_val_rain=False,
#                  get_val_night=False):

    
#     print_details = False
#     if rank == 0 or rank is None:
#         print_details = True
#         print('loading nuscenes...')
#     nusc = NuScenes(version='v1.0-{}'.format(version),
#                     dataroot=dataroot,
#                     verbose=print_details)
#     if print_details:
#         print('loading Maps...')
#     nusc_maps = get_nusc_maps(map_folder=dataroot)

#     traindata = VizData(
#         nusc,
#         nusc_maps=nusc_maps,
#         is_train=True,
#         data_aug_conf=data_aug_conf,
#         nsweeps=nsweeps,
#         centroid=centroid,
#         bounds=bounds,
#         res_3d=res_3d,
#         seqlen=seqlen,
#         refcam_id=refcam_id,
#         get_tids=get_tids,
#         temporal_aug=temporal_aug,
#         use_pseudolidar_filters=use_pseudolidar_filters,
#         do_shuffle_cams=do_shuffle_cams,
#         pseudolidar_encoder_type=pseudolidar_encoder_type,
#         use_shallow_metadata=use_shallow_metadata,
#         use_pre_scaled_imgs=use_pre_scaled_imgs,
#         custom_dataroot=custom_dataroot,
#         use_obj_layer_only_on_map=use_obj_layer_only_on_map,
#         vis_full_scenes=vis_full_scenes,
#         use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
#         print_details=print_details
#     )
#     valdata = VizData(
#         nusc,
#         nusc_maps=nusc_maps,
#         is_train=False,
#         data_aug_conf=data_aug_conf,
#         nsweeps=nsweeps,
#         centroid=centroid,
#         bounds=bounds,
#         res_3d=res_3d,
#         seqlen=seqlen,
#         refcam_id=refcam_id,
#         get_tids=get_tids,
#         temporal_aug=False,
#         use_pseudolidar_filters=use_pseudolidar_filters,
#         do_shuffle_cams=False,
#         pseudolidar_encoder_type=pseudolidar_encoder_type,
#         use_shallow_metadata=use_shallow_metadata,
#         use_pre_scaled_imgs=use_pre_scaled_imgs,
#         custom_dataroot=custom_dataroot,
#         use_obj_layer_only_on_map=use_obj_layer_only_on_map,
#         vis_full_scenes=vis_full_scenes,
#         use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
#         do_drn_val_split=do_drn_val_split,
#         get_val_day=get_val_day,
#         get_val_rain=get_val_rain,
#         get_val_night=get_val_night,
#         print_details=print_details
#     )

#     # for distributed data loading
#     if distributed_sampler:

#         g = torch.Generator()
#         g.manual_seed(125 + rank)

#         distributed_train_sampler = DistributedSampler(dataset=traindata, shuffle=shuffle)
#         distributed_val_sampler = DistributedSampler(dataset=valdata, shuffle=False)

#         trainloader = torch.utils.data.DataLoader(
#             traindata,
#             batch_size=bsz,
#             shuffle=False,  # formerly true  --> now handled by distributed sampler
#             sampler=distributed_train_sampler,
#             num_workers=nworkers,
#             drop_last=True,
#             worker_init_fn=my_seed_worker,
#             generator=g,
#             pin_memory=False)
#         valloader = torch.utils.data.DataLoader(
#             valdata,
#             batch_size=bsz,
#             shuffle=False,  # formerly true
#             sampler=distributed_val_sampler,
#             num_workers=nworkers_val,
#             drop_last=True,
#             pin_memory=False)
#     else:
#         trainloader = torch.utils.data.DataLoader(
#             traindata,
#             batch_size=bsz,
#             shuffle=shuffle,  # formerly true
#             num_workers=nworkers,
#             drop_last=True,
#             worker_init_fn=worker_rnd_init,
#             pin_memory=False)
#         valloader = torch.utils.data.DataLoader(
#             valdata,
#             batch_size=bsz,
#             shuffle=shuffle,  # formerly true
#             num_workers=nworkers_val,
#             drop_last=True,
#             pin_memory=False)
        
#     if print_details:
#         print(f'valloader valloader {len(valloader)}')

#     return trainloader, valloader


def compile_data(version, dataroot, data_aug_conf, centroid, bounds, res_3d, bsz,
                 nworkers, pseudolidar_encoder_type=None, shuffle=True, nsweeps=1, nworkers_val=1, seqlen=1, refcam_id=1,
                 get_tids=False, temporal_aug=False, use_pseudolidar_filters=False, use_shallow_metadata=True,
                 do_shuffle_cams=True, distributed_sampler=False, rank=None, use_pre_scaled_imgs=True,
                 custom_dataroot=None, use_obj_layer_only_on_map=False, vis_full_scenes=False,
                 use_pseudolidar_occupancy_map=False, do_drn_val_split=False, get_val_day=False, get_val_rain=False,
                 get_val_night=False):

    print_details = False
    if rank == 0 or rank is None:
        print_details = True
        print('loading nuscenes...')
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=print_details)
    if print_details:
        print('loading Maps...')
    nusc_maps = get_nusc_maps(map_folder=dataroot)

    traindata = VizData(
        nusc,
        nusc_maps=nusc_maps,
        is_train=True,
        data_aug_conf=data_aug_conf,
        nsweeps=nsweeps,
        centroid=centroid,
        bounds=bounds,
        res_3d=res_3d,
        seqlen=seqlen,
        refcam_id=refcam_id,
        get_tids=get_tids,
        temporal_aug=temporal_aug,
        use_pseudolidar_filters=use_pseudolidar_filters,
        do_shuffle_cams=do_shuffle_cams,
        pseudolidar_encoder_type=pseudolidar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        vis_full_scenes=vis_full_scenes,
        use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
        print_details=print_details
    )
    valdata = VizData(
        nusc,
        nusc_maps=nusc_maps,
        is_train=False,
        data_aug_conf=data_aug_conf,
        nsweeps=nsweeps,
        centroid=centroid,
        bounds=bounds,
        res_3d=res_3d,
        seqlen=seqlen,
        refcam_id=refcam_id,
        get_tids=get_tids,
        temporal_aug=False,
        use_pseudolidar_filters=use_pseudolidar_filters,
        do_shuffle_cams=False,
        pseudolidar_encoder_type=pseudolidar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        vis_full_scenes=vis_full_scenes,
        use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
        do_drn_val_split=do_drn_val_split,
        get_val_day=get_val_day,
        get_val_rain=get_val_rain,
        get_val_night=get_val_night,
        print_details=print_details
    )

    # for distributed data loading
    if distributed_sampler:
        print('dirstributed sampler')
        g = torch.Generator()
        g.manual_seed(125 + rank)

        distributed_train_sampler = DistributedSampler(dataset=traindata, shuffle=shuffle)
        distributed_val_sampler = DistributedSampler(dataset=valdata, shuffle=False)

        trainloader = torch.utils.data.DataLoader(
            traindata,
            batch_size=bsz,
            shuffle=False,  # formerly true  --> now handled by distributed sampler
            sampler=distributed_train_sampler,
            num_workers=nworkers,
            drop_last=True,
            worker_init_fn=my_seed_worker,
            generator=g,
            pin_memory=False)
        valloader = torch.utils.data.DataLoader(
            valdata,
            batch_size=bsz,
            shuffle=False,  # formerly true
            sampler=distributed_val_sampler,
            num_workers=nworkers_val,
            drop_last=True,
            pin_memory=False)
    else:
        print('not dirstributed sampler')
        trainloader = torch.utils.data.DataLoader(
            traindata,
            batch_size=bsz,
            shuffle=shuffle,  # formerly true
            num_workers=nworkers,
            drop_last=True,
            worker_init_fn=worker_rnd_init,
            pin_memory=False)
        valloader = torch.utils.data.DataLoader(
            valdata,
            batch_size=bsz,
            shuffle=shuffle,  # formerly true
            num_workers=nworkers_val,
            drop_last=True,
            pin_memory=False)

    if print_details:
        print('data ready')
    return trainloader, valloader