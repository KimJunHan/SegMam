# set seed in the beginning
import argparse
import os
import random
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # may help for debugging
# print("FIXED CUDA DEVICE: " + os.environ['CUDA_VISIBLE_DEVICES'])  # debug-only
import time
import warnings

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
import yaml
from shapely.errors import ShapelyDeprecationWarning
from tensorboardX import SummaryWriter

import nuscenes_data
import saverloader
import utils.basic
import utils.geom
import utils.improc
import utils.misc
import utils.vox
from nets.segnet_transformer_lift_fuse_new_decoders import (
    SegnetTransformerLiftFuse,
)

# Suppress deprecation warnings from shapely regarding the nuscenes map api
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning, module="nuscenes.map_expansion.map_api")

torch.multiprocessing.set_sharing_strategy('file_system')

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

# the scene centroid is defined wrt a reference camera,
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0  # down 1 meter
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200


def requires_grad(parameters: iter, flag: bool = True) -> None:
    """
    Sets the `requires_grad` attribute of the given parameters.
    Args:
        parameters (iterable): An iterable of parameter tensors whose `requires_grad` attribute will be set.
        flag (bool, optional): If True, sets `requires_grad` to True. If False, sets it to False.
            Default is True.

    Returns:
        None
    """
    for p in parameters:
        p.requires_grad = flag


def fetch_optimizer(lr: float, wdecay: float, epsilon: float, num_steps: int, params: iter) \
        -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Fetches an AdamW optimizer and a OneCycleLR scheduler.
    Args:
        lr (float): Learning rate for the optimizer.
        wdecay (float): Weight decay (L2 penalty) for the optimizer.
        epsilon (float): Term added to the denominator to improve numerical stability in the optimizer.
        num_steps (int): Number of steps for the learning rate scheduler.
        params (iter): Iterable of parameters to optimize or dictionaries defining parameter groups.

    Returns:
        tuple: A tuple containing the optimizer and the learning rate scheduler.
            - optimizer (torch.optim.AdamW): The AdamW optimizer.
            - scheduler (torch.optim.lr_scheduler.OneCycleLR): The OneCycleLR learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps + 100, pct_start=0.05,
                                                    cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class SimpleLoss(torch.nn.Module):
    """
    SimpleLoss module that computes the binary cross-entropy loss.

    Args:
        pos_weight (float): Positive class weight for the binary cross-entropy loss.

    Methods:
        forward(ypred: torch.Tensor, ytgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            Forward pass that computes the binary cross-entropy loss.
    """

    def __init__(self, pos_weight: float):
        """Initializes the SimpleLoss module with the specified positive class weight."""
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred: torch.Tensor, ytgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the binary cross-entropy loss.

        Args:
            ypred (torch.Tensor): Predicted logits.
            ytgt (torch.Tensor): Target tensor.
            valid (torch.Tensor): Mask indicating valid elements.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss


class SigmoidFocalLoss(torch.nn.Module):
    """
    Computes the sigmoid of the model output to get values between 0 and 1, then applies the Focal Loss.
    """

    def __init__(self, alpha: float = -1.0, gamma: int = 2, reduction: str = "mean"):
        """
        Args:
            alpha (float, optional): Balances the importance of positive/negative examples. Default is -1.0.
            gamma (int, optional): If >= 0, reduces the loss contribution from easy examples
                and extends the range in which an example receives low loss. Default is 2.
            reduction (str, optional): Specifies the reduction to apply to the output. Options are 'mean', 'sum',
                and 'sum_of_class_means'. Default is 'mean'.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, map_seg_e: torch.Tensor, map_seg_gt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes the sigmoid focal loss.

        Args:
            map_seg_e (torch.Tensor): Predicted logits.
            map_seg_gt (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        # get predictions between 0 and 1
        p = torch.sigmoid(map_seg_e)
        # BCE with logits
        ce_loss = F.binary_cross_entropy_with_logits(input=map_seg_e, target=map_seg_gt, reduction="none")
        p_t = p * map_seg_gt + (1 - p) * (1 - map_seg_gt)
        f_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * map_seg_gt + (1 - self.alpha) * (1 - map_seg_gt)
            f_loss = alpha_t * f_loss
        else:
            f_loss = f_loss

        if self.reduction == "mean":  # get mean over all classes
            f_loss = f_loss.mean()
        elif self.reduction == "sum":
            f_loss = f_loss.sum()
        elif self.reduction == "sum_of_class_means":
            # mean over B and bev grid -> then sum avg class error
            f_loss = f_loss.mean(dim=[0, 2, 3]).sum()
        return f_loss


def grad_acc_metrics(metrics_single_pass: dict, metrics_mean_grad_acc: dict, internal_step: int, grad_acc: int) \
        -> dict:
    """
    Accumulates metrics over gradient accumulation steps and computes mean values.
    Args:
        metrics_single_pass (dict): Dictionary containing metrics for a single pass.
        metrics_mean_grad_acc (dict): Dictionary containing accumulated metrics over gradient accumulation steps.
        internal_step (int): Current internal step within the gradient accumulation process.
        grad_acc (int): Number of gradient accumulation steps.

    Returns:
        dict: Dictionary containing mean values of accumulated metrics.
    """
    # Idea: loop over all keys -> if value is None -> do nothing; if value is not None -> accumulate
    for key in metrics_single_pass.keys():
        if metrics_single_pass[key] is not None and key != 'map_seg_thresholds':
            metrics_mean_grad_acc[key] += metrics_single_pass[key]
        else:
            metrics_mean_grad_acc[key] = metrics_single_pass[key]
    # Calculate mean values for losses, but accumulate intersections and unions, no early mean computation
    if internal_step == grad_acc - 1:
        for key in metrics_mean_grad_acc.keys():
            if metrics_mean_grad_acc[key] is not None:  # Exclude mean over intersections/unions
                if key not in ['obj_intersections', 'obj_unions', 'map_masks_intersections', 'map_masks_unions',
                               'map_masks_multi_ious_intersections', 'map_masks_multi_ious_unions',
                               'map_seg_thresholds']:
                    metrics_mean_grad_acc[key] = metrics_mean_grad_acc[key] / grad_acc  # Calculate mean
            else:
                metrics_mean_grad_acc[key] = None
    return metrics_mean_grad_acc



def gen_metrics(metrics: dict, train_task: str = 'both') -> None:
    """
    Computes the final metrics per batch after gradient accumulation

    Args:
        metrics (dict): metrics returned by the device specific model
        train_task (str): 'both', 'object' or 'map' -> enables control on the logging w.r.t. the respective tasks

    Returns:
        None
    """

    if train_task == 'both' or train_task == 'map':
        # single threshold IoUs (t=0.4)
        map_intersections_per_class = metrics['map_masks_intersections']
        map_unions_per_class = metrics['map_masks_unions']
        # multi threshold IoUs
        map_masks_multi_ious_intersections = metrics['map_masks_multi_ious_intersections']
        map_masks_multi_ious_unions = metrics['map_masks_multi_ious_unions']
        map_seg_thresholds = metrics['map_seg_thresholds']

        # map ious:
        # single threshold iou
        map_iou_all = (map_intersections_per_class / (map_unions_per_class + 1e-4))
        map_mean_iou = map_iou_all.sum(dim=0) / torch.count_nonzero(map_iou_all, dim=0)

        metrics['drivable_iou'] = map_iou_all[0].item()
        metrics['carpark_iou'] = map_iou_all[1].item()
        metrics['ped_cross_iou'] = map_iou_all[2].item()
        metrics['walkway_iou'] = map_iou_all[3].item()
        #metrics['stop_line_iou'] = map_iou_all[4].item()
        #metrics['road_divider_iou'] = map_iou_all[5].item()
        #metrics['lane_divider_iou'] = map_iou_all[6].item()

        metrics['masks_mean_iou'] = map_mean_iou.item()

        # multi threshold ious:
        map_masks_multi_iou = map_masks_multi_ious_intersections / (map_masks_multi_ious_unions + 1e-4)  # 7,12
        best_map_ious, best_threshold_index = torch.max(map_masks_multi_iou, dim=1)
        best_map_mean_iou = best_map_ious.sum(dim=0) / torch.count_nonzero(best_map_ious, dim=0)
        best_thresholds = map_seg_thresholds[best_threshold_index]

        metrics['drivable_ious'] = map_masks_multi_iou[0]  # (1,12) tensor for all threshs
        metrics['carpark_ious'] = map_masks_multi_iou[1]
        metrics['ped_cross_ious'] = map_masks_multi_iou[2]
        metrics['walkway_ious'] = map_masks_multi_iou[3]
        # metrics['stop_line_ious'] = map_masks_multi_iou[4]
        # metrics['road_divider_ious'] = map_masks_multi_iou[5]
        # metrics['lane_divider_ious'] = map_masks_multi_iou[6]

        metrics['best_drivable_iou'] = best_map_ious[0]
        metrics['best_carpark_iou'] = best_map_ious[1]
        metrics['best_ped_cross_iou'] = best_map_ious[2]
        metrics['best_walkway_iou'] = best_map_ious[3]
        # metrics['best_stop_line_iou'] = best_map_ious[4]
        # metrics['best_road_divider_iou'] = best_map_ious[5]
        # metrics['best_lane_divider_iou'] = best_map_ious[6]

        metrics['best_map_mean_iou'] = best_map_mean_iou
        metrics['best_thresholds'] = best_thresholds

    if train_task == 'both' or train_task == 'object':
        obj_intersections = metrics['obj_intersections']
        obj_unions = metrics['obj_unions']

        # calc ious:
        obj_iou = obj_intersections / (obj_unions + 1e-4)
        metrics['obj_iou'] = obj_iou

    return 


def create_train_pool_dict(name: str, n_pool: int) -> tuple[dict, str]:
    """
    Creates a dictionary of training pools for tracking various metrics during training.

    Args:
        name (str): Name suffix for the pool dictionary keys.
        n_pool (int): Number of values included for the moving average.

    Returns:
        tuple[dict, str]: A tuple containing the dictionary of training pools and the name suffix.
            The dictionary includes pools for:
            - Total loss
            - Time
            - Object segmentation IoU
            - Map masks IoU for various classes (drivable area, carpark, pedestrian crossing, etc.)
            - Mean IoU for map masks
            - Specific losses for object and map segmentation
    """

    train_pool_dict = {
        # total loss
        'loss_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # time
        'time_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # object segmentation IoU
        'obj_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),

        # map masks
        'drivable_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'carpark_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'ped_cross_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'walkway_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # 'stop_line_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # 'road_divider_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # 'lane_divider_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # mean map maks iou
        'masks_mean_iou_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),

        # specific losses
        # object seg
        'ce_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'ce_weight_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        # map seg
        'fc_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
        'fc_map_weight_pool_' + name: utils.misc.SimplePool(n_pool, version='np'),
    }
    return train_pool_dict, name


# def normalize_pseudolidar(pseudolidar):
#     # list/tuple이면 붙여서 텐서로(이미 텐서면 무시)
#     if isinstance(pseudolidar, (list, tuple)):
#         pseudolidar = torch.nn.utils.rnn.pad_sequence(pseudolidar, batch_first=True)  # [B, R, D]
#     # sparse면 dense로
#     if getattr(pseudolidar, "is_sparse", False):
#         pseudolidar = pseudolidar.to_dense()
#     # [N, D] -> [1, N, D]
#     if pseudolidar.dim() == 2:
#         pseudolidar = pseudolidar.unsqueeze(0)
#     # 이제 3D만 남음
#     # 채널(피쳐) 19가 어디에 있는지에 따라 정렬
#     if pseudolidar.size(-1) == 3:
#         # [B, R, 19] 이미 OK
#         return pseudolidar
#     elif pseudolidar.size(1) == 3:
#         # [B, 19, R] -> [B, R, 19]
#         return pseudolidar.permute(0, 2, 1)
#     else:
#         raise ValueError(f"Unexpected pseudolidar shape {tuple(pseudolidar.shape)}; cannot find feature dim=19.")


def run_model(model, loss_fn, map_seg_loss_fn, d, device='cuda:0', sw=None, use_pseudolidar_encoder=None,
              pseudolidar_encoder_type=None, train_task='both', use_shallow_metadata=True,
              use_obj_layer_only_on_map=True):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    voxel_input_feature_buffer = None
    voxel_coordinate_buffer = None
    number_of_occupied_voxels = None
    in_occ_mem0 = None

    if pseudolidar_encoder_type == "voxel_net":
        # voxelnet
        imgs, rots, trans, intrins, seg_bev_g, \
            valid_bev_g, pseudolidar_data, bev_map_mask_g, bev_map_g, egocar_bev, \
            voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels,all_transformate, all_rotation, all_depth_img= d

        # VoxelNet preprocessing
        voxel_input_feature_buffer = voxel_input_feature_buffer[:, 0]
        voxel_coordinate_buffer = voxel_coordinate_buffer[:, 0]
        number_of_occupied_voxels = number_of_occupied_voxels[:, 0]
        voxel_input_feature_buffer = voxel_input_feature_buffer.to(device)
        voxel_coordinate_buffer = voxel_coordinate_buffer.to(device)
        number_of_occupied_voxels = number_of_occupied_voxels.to(device)

    else:
        imgs, rots, trans, intrins, seg_bev_g, \
            valid_bev_g, pseudolidar_data, bev_map_mask_g, bev_map_g, egocar_bev,all_transformate, all_rotation,all_depth_img = d

    B0, T, S, C, H, W = imgs.shape
    assert (T == 1)

    # eliminate the time dimension
    imgs = imgs[:, 0]
    rots = rots[:, 0]
    trans = trans[:, 0]
    intrins = intrins[:, 0]    # intrinsics for each cam --> shape:  [B,S,4,4]
    seg_bev_g = seg_bev_g[:, 0]
    valid_bev_g = valid_bev_g[:, 0]
    pseudolidar_data = pseudolidar_data[:, 0]
    # added bev_map_gt
    bev_map_mask_g = bev_map_mask_g[:, 0]
    if use_obj_layer_only_on_map:
        bev_map_mask_g = bev_map_mask_g[:, :-1]
    bev_map_g = bev_map_g[:, 0]
    # added egocar in bev plane
    egocar_bev = egocar_bev[:, 0]

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5  # go to -0.5, 0.5cd

    seg_bev_g = seg_bev_g.to(device)
    obj_seg_bev_e = torch.zeros_like(seg_bev_g)
    valid_bev_g = valid_bev_g.to(device)
    # added bev_map_gt
    bev_map_mask_g = bev_map_mask_g.to(device)
    bev_map_mask_e = torch.zeros_like(bev_map_mask_g)
    bev_map_g = bev_map_g.to(device)
    bev_map_e = torch.zeros_like(bev_map_g)
    # added egocar in bev plane
    egocar_bev = egocar_bev.to(device)

    # create ego car color plane
    ego_plane = torch.zeros_like(bev_map_g).to(device)
    ego_plane[:, [0, 2]] = 0.0
    ego_plane[:, 1] = 1.0
    # combine ego car and map
    ego_car_on_map_g = bev_map_g * (1 - egocar_bev) + ego_plane * egocar_bev

    # create other cars plane
    other_cars_plane = torch.zeros_like(bev_map_g).to(device)
    other_cars_plane[:, [0, 1]] = 0.0
    other_cars_plane[:, 2] = 1.0
    # combine ego car other cars and map
    ego_other_cars_on_map_g = ego_car_on_map_g * (1 - seg_bev_g) + other_cars_plane * seg_bev_g
    ego_other_cars_on_map_e = torch.zeros_like(ego_other_cars_on_map_g)

    rad_data = pseudolidar_data.to(device).permute(0, 2, 1)  # B, R,3 
    #rad_data = normalize_pseudolidar(pseudolidar_data)   # shape: [B, R, 3]

    xyz_rad = rad_data[:, :, :3]
    meta_rad = rad_data[:, :, 3:]
    shallow_meta_rad = rad_data[:, :, 5:8]

    B, S, C, H, W = rgb_camXs.shape

    def __p(x):
        # Wrapper function: e.g. unites B,S dim to B*S
        return utils.basic.pack_seqdim(x, B)

    def __u(x):
        # Wrapper function: e.g. splits B*S dim into B,S
        return utils.basic.unpack_seqdim(x, B)

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))

    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:, 0], xyz_rad)

    # voxel object representing the memory for the (pseudolidar) data
    vox_util = utils.vox.Vox_util(
        Z, Y, X,  # Z=200, Y=8, X=200
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    if not model.module.use_pseudolidar:
        in_occ_mem0 = None
    elif model.module.use_pseudolidar and (model.module.use_metapseudolidar or use_shallow_metadata):
        if use_pseudolidar_encoder and pseudolidar_encoder_type == 'voxel_net':
            voxelnet_feats_mem0 = voxel_input_feature_buffer, voxel_coordinate_buffer, number_of_occupied_voxels
            in_occ_mem0 = voxelnet_feats_mem0
        elif use_shallow_metadata:
            shallow_metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, shallow_meta_rad, Z, Y, X,
                                                                       assert_cube=False)
            in_occ_mem0 = shallow_metarad_occ_mem0
        else:  # use_metapseudolidar
            metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)
            in_occ_mem0 = metarad_occ_mem0
    elif model.module.use_pseudolidar:
        rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
        in_occ_mem0 = rad_occ_mem0
    elif model.module.use_metapseudolidar or use_shallow_metadata:
        assert False  # cannot use_metapseudolidar without use_pseudolidar


    seg_e, timing = model(
        rgb_camXs=rgb_camXs,
        pix_T_cams=pix_T_cams,
        cam0_T_camXs=cam0_T_camXs,
        vox_util=vox_util,
        rad_occ_mem0=in_occ_mem0)

    # get bev map from masks
    if train_task == 'both' or train_task == 'map':

        if train_task == 'both':
            bev_map_mask_e = seg_e[:, :-1]
            obj_seg_bev_e = seg_e[:, -1].unsqueeze(dim=1)
            obj_seg_bev = torch.sigmoid(obj_seg_bev_e)

            bev_map_only_mask_g = bev_map_mask_g  # [:, :-1]
        else:
            bev_map_mask_e = seg_e
            obj_seg_bev = seg_bev_g  # add gt vehicles on map (optional)
            bev_map_only_mask_g = bev_map_mask_g

        map_seg_threshold = 0.4
        bev_map_e = nuscenes_data.get_rgba_map_from_mask2_on_batch(
            torch.sigmoid(bev_map_mask_e).detach().cpu().numpy(),
            threshold=map_seg_threshold, a=0.4).to(device)

        # combine ego car and bev_map_e
        ego_car_on_map_e = bev_map_e * (1 - egocar_bev) + ego_plane * egocar_bev  # check dims

        # create other cars estimate plane
        other_cars_plane_e = torch.zeros_like(bev_map_e).to(device)
        other_cars_plane_e[:, [0, 1]] = 0.0
        other_cars_plane_e[:, 2] = 1.0

        # combine ego car other cars and map
        ego_other_cars_on_map_e = ego_car_on_map_e * (1 - obj_seg_bev) + other_cars_plane_e * obj_seg_bev

        # loss calculation
        map_seg_fc_loss = map_seg_loss_fn(bev_map_mask_e, bev_map_only_mask_g)
        #   map
        fc_map_factor = 1 / torch.exp(model.module.fc_map_weight)
        map_seg_fc_loss = 20.0 * map_seg_fc_loss * fc_map_factor  # 20.0
        # add to total loss
        total_loss += map_seg_fc_loss

        # MAP IoU calculation

        # ious for map segmentation:
        tp = ((torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & bev_map_mask_g.bool()).sum(dim=[2, 3])
        fp = ((torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & ~bev_map_mask_g.bool()).sum(dim=[2, 3])
        fn = (~(torch.sigmoid(bev_map_mask_e) >= map_seg_threshold).bool() & bev_map_mask_g.bool()).sum(dim=[2, 3])

        map_intersections_per_class = tp.sum(dim=0)  # sum over batch --> 7 intersection values
        map_unions_per_class = (
                    tp.sum(dim=0) + fp.sum(dim=0) + fn.sum(dim=0) + 1e-4)  # sum over batch --> 7 union values

        # ################# NEW MULTI-IOU CALCULATION #####################
        map_seg_thresholds = torch.Tensor([0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]).to(device)
        sig_map_bev_e_new = torch.sigmoid(bev_map_mask_e)[:, :, :, :, None] >= map_seg_thresholds
        bev_map_mask_g_new = bev_map_only_mask_g[:, :, :, :, None]

        tps = (sig_map_bev_e_new.bool() & bev_map_mask_g_new.bool()).sum(dim=[2, 3])  # (B,7,12)
        fps = (sig_map_bev_e_new.bool() & ~bev_map_mask_g_new.bool()).sum(dim=[2, 3])
        fns = (~sig_map_bev_e_new.bool() & bev_map_mask_g_new.bool()).sum(dim=[2, 3])

        # besti i/u
        map_masks_multi_ious_intersections = tps.sum(0)
        map_masks_multi_ious_unions = (tps.sum(0) + fps.sum(0) + fns.sum(0) + 1e-4)

        # metrics
        metrics['focal_loss_map'] = map_seg_fc_loss  # .item()
        metrics['fc_map_weight'] = model.module.fc_map_weight.item()
        # single threshold IoUs (t=0.4)
        metrics['map_masks_intersections'] = map_intersections_per_class
        metrics['map_masks_unions'] = map_unions_per_class
        # multi threshold IoUs
        metrics['map_masks_multi_ious_intersections'] = map_masks_multi_ious_intersections
        metrics['map_masks_multi_ious_unions'] = map_masks_multi_ious_unions
        metrics['map_seg_thresholds'] = map_seg_thresholds

        # Note that the following calculations are only done per gradient accumulation step and thus
        # not representative for the whole batch.
        # These values are computed again after gradient accumulation in 'gen_metrics()" but the
        # following computations may  help for debugging.

        # map ious:
        # single threshold iou
        # map_iou_all = (map_intersections_per_class / (map_unions_per_class + 1e-4))
        # map_mean_iou = map_iou_all.sum(dim=0) / torch.count_nonzero(map_iou_all, dim=0)
        eps = 1e-6
        map_iou_all = map_intersections_per_class / (map_unions_per_class + eps)  # [num_classes]
        valid = (map_unions_per_class > eps)                                      # union>0인 클래스만 평균
        denom = valid.sum().clamp_min(1)
        map_mean_iou = (map_iou_all[valid].sum() / denom)
        metrics['drivable_iou'] = map_iou_all[0].item()
        metrics['carpark_iou'] = map_iou_all[1].item()
        metrics['ped_cross_iou'] = map_iou_all[2].item()
        metrics['walkway_iou'] = map_iou_all[3].item()
        # metrics['stop_line_iou'] = map_iou_all[4].item()
        # metrics['road_divider_iou'] = map_iou_all[5].item()
        # metrics['lane_divider_iou'] = map_iou_all[6].item()

        metrics['masks_mean_iou'] = map_mean_iou.item()

        # multi threshold ious:
        map_masks_multi_iou = map_masks_multi_ious_intersections / (map_masks_multi_ious_unions + 1e-4)  # 7,12
        best_map_ious, best_threshold_index = torch.max(map_masks_multi_iou, dim=1)
        best_map_mean_iou = best_map_ious.sum(dim=0) / torch.count_nonzero(best_map_ious, dim=0)
        best_thresholds = map_seg_thresholds[best_threshold_index]

        metrics['drivable_ious'] = map_masks_multi_iou[0]  # (1,12) tensor for all threshs
        metrics['carpark_ious'] = map_masks_multi_iou[1]
        metrics['ped_cross_ious'] = map_masks_multi_iou[2]
        metrics['walkway_ious'] = map_masks_multi_iou[3]
        # metrics['stop_line_ious'] = map_masks_multi_iou[4]
        # metrics['road_divider_ious'] = map_masks_multi_iou[5]
        # metrics['lane_divider_ious'] = map_masks_multi_iou[6]

        metrics['best_drivable_iou'] = best_map_ious[0]
        metrics['best_carpark_iou'] = best_map_ious[1]
        metrics['best_ped_cross_iou'] = best_map_ious[2]
        metrics['best_walkway_iou'] = best_map_ious[3]
        # metrics['best_stop_line_iou'] = best_map_ious[4]
        # metrics['best_road_divider_iou'] = best_map_ious[5]
        # metrics['best_lane_divider_iou'] = best_map_ious[6]

        metrics['best_map_mean_iou'] = best_map_mean_iou
        metrics['best_thresholds'] = best_thresholds

    # object seg task
    if train_task == 'both' or train_task == 'object':
        if train_task == 'both':
            obj_seg_bev_e = seg_e[:, -1].unsqueeze(dim=1)
        else:  # 'object'
            obj_seg_bev_e = seg_e
            obj_seg_bev_e_sigmoid = torch.sigmoid(obj_seg_bev_e)
            ego_other_cars_on_map_e = ego_car_on_map_g * (1 - obj_seg_bev_e_sigmoid) + \
                other_cars_plane * obj_seg_bev_e_sigmoid
        # clc loss
        ce_loss = loss_fn(obj_seg_bev_e, seg_bev_g, valid_bev_g)
        # obj
        ce_factor = 1 / torch.exp(model.module.ce_weight)
        ce_loss = 10.0 * ce_loss * ce_factor  # 10.0
        total_loss += ce_loss

        # object IoUs
        obj_seg_bev_e_round = torch.sigmoid(obj_seg_bev_e).round()
        obj_intersection = (obj_seg_bev_e_round * seg_bev_g * valid_bev_g).sum(dim=[1, 2, 3])
        obj_union = ((obj_seg_bev_e_round + seg_bev_g) * valid_bev_g).clamp(0, 1).sum(dim=[1, 2, 3])

        obj_intersections = obj_intersection.sum()
        obj_unions = obj_union.sum()

        metrics['ce_loss'] = ce_loss  # .item()
        metrics['ce_weight'] = model.module.ce_weight.item()
        metrics['obj_intersections'] = obj_intersections  # .item()
        metrics['obj_unions'] = obj_unions  # .item()

        # calc ious:
        obj_iou = obj_intersections/(obj_unions + 1e-4)
        metrics['obj_iou'] = obj_iou

    return total_loss, metrics


def main(
        exp_name='segmam_debug',
        # training
        max_iters=75000,
        log_freq=1000,
        shuffle=True,
        dset='trainval',
        save_freq=1000,
        batch_size=8,
        grad_acc=5,
        lr=3e-4,
        use_scheduler=True,
        weight_decay=1e-7,
        nworkers=12,
        # data/log/save/load directories
        data_dir='../nuscenes/',
        custom_dataroot='../../../nuscenes/scaled_images',
        log_dir='logs_nuscenes_segmam',
        ckpt_dir='checkpoints/',
        keep_latest=1,
        init_dir='',
        ignore_load=None,
        load_step=False,
        load_optimizer=False,
        load_scheduler=False,
        # data
        final_dim=[448, 896],  # to match //8, //14, //16 and //32 in Vit
        rand_flip=True,
        rand_crop_and_resize=True,
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        maps = ['drivable_area','carpark_area','ped_crossing','walkway'],
        ncams= 6,
        nsweeps=5,
        # model
        encoder_type='dino_v2',
        pseudolidar_encoder_type='voxel_net',
        use_rpn_pseudolidar=False,
        train_task='both',
        use_pseudolidar=False,
        use_pseudolidar_filters=False,
        use_pseudolidar_encoder=False,
        use_metapseudolidar=False,
        use_shallow_metadata=False,
        use_pre_scaled_imgs=False,
        use_obj_layer_only_on_map=False,
        init_query_with_image_feats=True,
        do_rgbcompress=True,
        do_shuffle_cams=True,
        use_multi_scale_img_feats=False,
        num_layers=4,
        # cuda
        device_ids=[0],
        freeze_dino=True,
        do_feat_enc_dec=True,
        combine_feat_init_w_learned_q=True,
        model_type='transformer',
        use_pseudolidar_occupancy_map=False,
        learnable_fuse_query=True,
):
    assert model_type == 'transformer', "SegMam release only supports model_type='transformer'"
    B = batch_size
    assert (B % len(device_ids) == 0)  # batch size must be divisible by number of gpus
    if grad_acc > 1:
        print('effective batch size:', B * grad_acc)
    device = 'cuda:%d' % device_ids[0]

    # debug only
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Devices available: %d " % torch.cuda.device_count())
        print("Current CUDA device ID: %d" % torch.cuda.current_device())
        # device_ids[0])  # torch.cuda.current_device())
    else:
        print("CUDA is --- NOT --- available")

    # autogen a name
    model_name = "%d" % B
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    lrn = "%.1e" % lr  # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"

    import datetime
    model_date = datetime.datetime.now().strftime('%H-%M-%S')
    model_name = model_name + '_' + model_date

    model_name = exp_name + '_' + model_name
    print('model_name', model_name)

    # set up ckpt and logging
    ckpt_dir = os.path.join(ckpt_dir, model_name)
    writer_t = SummaryWriter(os.path.join(log_dir, model_name + '/t'), max_queue=10, flush_secs=60)

    print('resolution:', final_dim)

    if use_pseudolidar_encoder:
        print("Pseudolidar encoder: ", pseudolidar_encoder_type)
    else:
        print("NO PSEUDOLIDAR ENCODER")

   


    if rand_crop_and_resize:
        resize_lim = [0.8, 1.2]
        crop_offset = int(final_dim[0] * (1 - resize_lim[0]))
    else:
        resize_lim = [1.0, 1.0]
        crop_offset = 0

    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 
        'W': 1600,
        'cams': cams,
        'maps': maps,
        'ncams': ncams,
    }
    train_dataloader, _ = nuscenes_data.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z, Y, X),
        bsz=B,
        nworkers=nworkers,
        shuffle=shuffle,
        use_pseudolidar_filters=use_pseudolidar_filters,
        seqlen=1,
        nsweeps=nsweeps,
        do_shuffle_cams=do_shuffle_cams,
        pseudolidar_encoder_type=pseudolidar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
    )
    train_iterloader = iter(train_dataloader)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    # set up model & losses
    seg_loss_fn = SimpleLoss(2.13).to(device)  # value from lift-splat
    map_seg_loss_fn = SigmoidFocalLoss(alpha=0.25, gamma=3, reduction="sum_of_class_means").to(
        device)  # for map segmentation head

    model = SegnetTransformerLiftFuse(Z_cam=200, Y_cam=8, X_cam=200, Z_rad=Z, Y_rad=Y, X_rad=X, vox_util=None,
                                      use_pseudolidar=use_pseudolidar, use_metapseudolidar=use_metapseudolidar,
                                      use_shallow_metadata=use_shallow_metadata,
                                      use_pseudolidar_encoder=use_pseudolidar_encoder, do_rgbcompress=do_rgbcompress,
                                      encoder_type=encoder_type, pseudolidar_encoder_type=pseudolidar_encoder_type,
                                      rand_flip=rand_flip, train_task=train_task,
                                      init_query_with_image_feats=init_query_with_image_feats,
                                      use_obj_layer_only_on_map=use_obj_layer_only_on_map,
                                      do_feat_enc_dec=do_feat_enc_dec,
                                      use_multi_scale_img_feats=use_multi_scale_img_feats, num_layers=num_layers,
                                      latent_dim=128,
                                      combine_feat_init_w_learned_q=combine_feat_init_w_learned_q,
                                      use_rpn_pseudolidar=use_rpn_pseudolidar, use_pseudolidar_occupancy_map=use_pseudolidar_occupancy_map,
                                      freeze_dino=freeze_dino, learnable_fuse_query=learnable_fuse_query)

    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    parameters = list(model.parameters())
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, model.parameters())
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        scheduler = None

    # Counting trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params}')
    # Counting non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Non-trainable parameters: {non_trainable_params}')
    # Overall parameters
    total_params = trainable_params + non_trainable_params
    print('Total parameters (trainable + fixed)', total_params)

    # load checkpoint
    global_step = 0
    if init_dir:
        if load_step and load_optimizer and load_scheduler:
            # NOTE: pass model.module (not the DataParallel wrapper). Checkpoints are saved from
            # model.module (unprefixed keys); loading into the wrapper would key-mismatch every
            # param and, with strict=False, silently leave the model RANDOM-initialized.
            global_step = saverloader.load(init_dir, model.module, optimizer, scheduler=scheduler,
                                           ignore_load=ignore_load, device_ids=device_ids, is_DP=True)

        elif load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model.module, optimizer, ignore_load=ignore_load,
                                           device_ids=device_ids)
            print("global_step: ", global_step)
        elif load_step:
            global_step = saverloader.load(init_dir, model.module, ignore_load=ignore_load, device_ids=device_ids)
        else:
            _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
            global_step = 0
        print('checkpoint loaded...')

    model.train()

    # set up running logging pool
    n_pool_train = 10
    train_pool_dict, train_pool_dict_name = create_train_pool_dict(name='t', n_pool=n_pool_train)

    sw_t = None

    # training loop
    while global_step < max_iters:
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0

        metrics = {}
        metrics_mean_grad_acc = {}
        total_loss = 0.0

        for internal_step in range(grad_acc):
            # read sample
            read_start_time = time.time()

            if internal_step == grad_acc - 1:
                sw_t = utils.improc.Summ_writer(  # only save last sample of gradient accumulation
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=2,
                    scalar_freq=int(log_freq / 2),
                    just_gif=True)
            else:
                sw_t = None

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)

            read_time = time.time() - read_start_time
            iter_read_time += read_time

            # run training iteration
            total_loss_, metrics_ = run_model(model, seg_loss_fn,
                                              map_seg_loss_fn,
                                              sample, device, sw_t, use_pseudolidar_encoder, pseudolidar_encoder_type, train_task,
                                              use_shallow_metadata=use_shallow_metadata,
                                              use_obj_layer_only_on_map=use_obj_layer_only_on_map)

            (total_loss_ / grad_acc).backward()

            # collect total loss and metrics over grad_acc steps
            if internal_step == 0:
                metrics_mean_grad_acc = metrics_
                total_loss = total_loss_
            else:
                metrics = grad_acc_metrics(metrics_single_pass=metrics_, metrics_mean_grad_acc=metrics_mean_grad_acc,
                                           internal_step=internal_step, grad_acc=grad_acc)  # accumulates metrics and if end of grad_acc -> divides by grad_acc
                total_loss += total_loss_

            if internal_step == grad_acc - 1:
                total_loss = total_loss / grad_acc  # 12.9796

        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        gen_metrics(metrics=metrics, train_task=train_task)
        train_pool_dict['loss_pool_' + train_pool_dict_name].update([total_loss.item()])

        if train_task in ('both', 'object'):
            # object seg
            if 'obj_iou' in metrics:
                train_pool_dict['obj_iou_pool_' + train_pool_dict_name].update([metrics['obj_iou'].item()])
            if 'ce_loss' in metrics:
                train_pool_dict['ce_pool_' + train_pool_dict_name].update([metrics['ce_loss'].item()])
            if 'ce_weight' in metrics:
                train_pool_dict['ce_weight_pool_' + train_pool_dict_name].update([metrics['ce_weight']])

        if train_task in ('both', 'map'):
            # per-class IoU
            for label in ('drivable', 'carpark', 'ped_cross', 'walkway'):
                k = f'{label}_iou'
                if k in metrics:
                    train_pool_dict[f'{label}_iou_pool_' + train_pool_dict_name].update([metrics[k]])
            # mean IoU over map masks
            if 'masks_mean_iou' in metrics:
                train_pool_dict['masks_mean_iou_pool_' + train_pool_dict_name].update([metrics['masks_mean_iou']])
            # focal loss (map)
            if 'focal_loss_map' in metrics:
                train_pool_dict['fc_pool_' + train_pool_dict_name].update([metrics['focal_loss_map'].item()])
            if 'fc_map_weight' in metrics:
                train_pool_dict['fc_map_weight_pool_' + train_pool_dict_name].update([metrics['fc_map_weight']])

        # save model checkpoint
        if np.mod(global_step, save_freq) == 0:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, scheduler=scheduler,
                             keep_latest=keep_latest)

        model.train()

        # log lr and time
        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)

        iter_time = time.time() - iter_start_time
        train_pool_dict['time_pool_' + train_pool_dict_name].update([iter_time])



        if train_task == 'both':
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; DP_loss %.5f; obj_iou_t %.1f; '
                  ' map_iou_t %.1f;' % (
                   model_name, global_step, max_iters, iter_read_time, iter_time,
                   total_loss,
                   100 * train_pool_dict['obj_iou_pool_' + train_pool_dict_name].mean(),
                   100 * train_pool_dict['masks_mean_iou_pool_' + train_pool_dict_name].mean()))

        if train_task == 'object':
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; DP_loss %.5f; obj_iou_t %.1f; ' % (
                    model_name, global_step, max_iters, iter_read_time, iter_time,
                    total_loss,
                    100 * train_pool_dict['obj_iou_pool_' + train_pool_dict_name].mean()))

        if train_task == 'map':
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; DP_loss %.5f; '
                  'map_iou_t %.1f;' % (
                      model_name, global_step, max_iters, iter_read_time, iter_time,
                      total_loss,
                      100 * train_pool_dict['masks_mean_iou_pool_' + train_pool_dict_name].mean()))

    writer_t.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training with model-specific config.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(**config)