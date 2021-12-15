# Copyright (c) wondervictor. All Rights Reserved

from detectron2.config import CfgNode as CN
from detectron2 import model_zoo

def add_boundary_preserving_config(cfg):

    cfg.MODEL.BOUNDARY_MASK_HEAD = CN()
    cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV = 2
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

