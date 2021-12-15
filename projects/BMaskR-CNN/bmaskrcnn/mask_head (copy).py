# Copyright (c) wondervictor. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
import pywt
import pywt.data
import torchvision
from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.modeling.poolers import ROIPooler


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def boundary_loss_func(boundary_logits, gtmasks):
    """
    Args:
        boundary_logits (Tensor): A tensor of shape (B, H, W) or (B, H, W)
        gtmasks (Tensor): A tensor of shape (B, H, W) or (B, H, W)
    """
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=boundary_logits.device).reshape(1, 1, 3, 3).requires_grad_(False)
    boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0

    if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
        boundary_targets = F.interpolate(
            boundary_targets, boundary_logits.shape[2:], mode='nearest')

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
    dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets)
    return bce_loss + dice_loss


def boundary_preserving_mask_loss(pred_mask_logits,pred_boundary_logits,instances,vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
 
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)
    
    # print("pred_mask_logits",pred_mask_logits.shape)

    # number_of_instances = LL2.shape[0]
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    bce_loss = F.binary_cross_entropy_with_logits(pred_boundary_logits, gt_masks)
    dice_loss = dice_loss_func(torch.sigmoid(pred_boundary_logits), gt_masks)
    boundary_loss = dice_loss + bce_loss
    return mask_loss, boundary_loss


@ROI_MASK_HEAD_REGISTRY.register()
class BoundaryPreservingHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(BoundaryPreservingHead, self).__init__()
        #for pooling our p2 features
        # in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        # pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        # sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        # pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE

        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        conv_norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_boundary_conv = cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            num_classes = 1
        #---------------
        # self.mask_pooler= ROIPooler(
        #     output_size=pooler_resolution,
        #     scales=pooler_scales,
        #     sampling_ratio=sampling_ratio,
        #     pooler_type=pooler_type,
        # )
        self.mask_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.mask_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_final_fusion = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu)

        self.downsample = Conv2d(
            conv_dim, conv_dim,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.boundary_fcns = []
        cur_channels = input_shape.channels
        for k in range(num_boundary_conv):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_fcns.append(conv)
            cur_channels = conv_dim

        self.mask_to_boundary = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.wav_conv_level1 = nn.Conv3d(
            3, 3,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
        )
        self.wav_conv_level2 = nn.Conv3d(
            4, 4,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
         
        )
        self.boundary_to_mask = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )
        self.p2_conv_shortcut = Conv2d(
            conv_dim, conv_dim,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu
        )

        self.mask_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        self.boundary_deconv = ConvTranspose2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0
        )
        self.boundary_predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.mask_fcns + self.boundary_fcns +\
                     [self.mask_deconv, self.boundary_deconv, self.boundary_to_mask, self.mask_to_boundary,
                      self.mask_final_fusion, self.downsample]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)
        self.our_roi_align = torchvision.ops.MultiScaleRoIAlign(['feat1'],14, 3)

    def forward(self, mask_features, boundary_features, instances: List[Instances],feature_p2):
        for layer in self.mask_fcns:
            mask_features = layer(mask_features)


        # downsample
        boundary_features = self.downsample(boundary_features)
        x = boundary_features.cpu().detach().numpy()
        LL2,(LH2, HL2, HH2),(LH1, HL1, HH1)  = pywt.wavedec2(x, 'haar', mode='periodization', level=2)
        # print("LL2","LL2",LL2.shape,"LH1",LH1.shape)
        number_of_instances = LL2.shape[0]
        ii ={}
        ii['feat1'] = feature_p2
        boxes = torch.rand(number_of_instances, 4) * 256
        boxes = boxes.cuda()
        image_sizes = [(512, 512)]
        output_p2 = self.our_roi_align(ii, [boxes], image_sizes)
        # 2 conv on output_p2
        conv_p2 = self.p2_conv_shortcut(output_p2)
        conv_p2 = self.p2_conv_shortcut(conv_p2)
        # print(conv_p2.shape,'saraaa')
        mask_features = conv_p2 + mask_features
        # print("---------> output_p2:",output_p2.shape)
        #wavelet ----------
        level_2_mask = torch.zeros([4,number_of_instances,256,4,4]).cuda()
        level_1_mask = torch.zeros([3,number_of_instances,256,7,7]).cuda()
  
        # level1
        level_1_mask[0] = torch.tensor(HH1)
        level_1_mask[1] = torch.tensor(HL1)
        level_1_mask[2] = torch.tensor(LH1)

        # #level2
        level_2_mask[0] = torch.tensor(HH2)
        level_2_mask[1] = torch.tensor(HL2)
        level_2_mask[2] = torch.tensor(LH2)
        level_2_mask[3] = torch.tensor(LL2)

        level_1_mask = torch.reshape(level_1_mask,(number_of_instances,3,256,7,7))
        x_level1 = self.wav_conv_level1(level_1_mask)
        x_level1 = self.wav_conv_level1(x_level1)
        

        level_2_mask = torch.reshape(level_2_mask,(number_of_instances,4,256,4,4))
        x_level2 = self.wav_conv_level2(level_2_mask)
        x_level2 = self.wav_conv_level2(x_level2)
        #reshape
        level_2_mask = torch.reshape(level_2_mask,(4,number_of_instances,256,4,4))
        level_1_mask = torch.reshape(level_1_mask,(3,number_of_instances,256,7,7))
        #level2
        HH2 = level_2_mask[0]
        HL2 = level_2_mask[1]
        LH2 = level_2_mask[2]
        LL2 = level_2_mask[3]
        #level1
        HH1 = level_1_mask[0]
        HL1 = level_1_mask[1]
        LH1 = level_1_mask[2]
        im =pywt.waverec2([LL2.cpu().numpy(),(LH2.cpu().numpy(), HL2.cpu().numpy(), HH2.cpu().numpy()),(LH1.cpu().numpy(), HL1.cpu().numpy(), HH1.cpu().numpy())], 'haar', mode='symmetric',axes=(-2, -1))
        # print('*********************',im.shape)
        im =torch.tensor(im)
        boundary_features = im.cuda()
        boundary_features = boundary_features + self.mask_to_boundary(mask_features)
        
        #hazf 2 ta conv bad fusion
        # for layer in self.boundary_fcns:
        #     boundary_features = layer(boundary_features)
        # boundary to mask fusion
        mask_features = self.boundary_to_mask(boundary_features) + mask_features
        mask_features = self.mask_final_fusion(mask_features)
       
        # mask prediction
        mask_features = F.relu(self.mask_deconv(mask_features))
        mask_logits = self.mask_predictor(mask_features)
        # print("injaaaaadaaaaa",mask_logits.shape)
        # boundary prediction
       
        boundary_features = F.relu(self.boundary_deconv(boundary_features))
       
        boundary_logits = self.boundary_predictor(boundary_features)
        # print("injaaaaaaaaaa",boundary_logits.shape)

        
        if self.training:
            loss_mask, loss_boundary = boundary_preserving_mask_loss(
                mask_logits, boundary_logits, instances)
            return {"loss_mask": loss_mask,
                    "loss_boundary": loss_boundary,

                    }
        else:
            mask_rcnn_inference(mask_logits, instances)
            return instances
