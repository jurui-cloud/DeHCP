""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import importlib
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn

class HeterPyramidSingle(nn.Module):
    def __init__(self, args):
        super(HeterPyramidSingle, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.fix_modules = ['pyramid_backbone', 'cls_head', 'reg_head', 'dir_head']
        
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))

            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            if args.get("fix_encoder", False):
                self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

        """
        Would load from pretrain base.
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])
        """
        Shrink header, Would load from pretrain base.
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.fix_modules.append('shrink_conv')

        """
        Shared Heads, Would load from pretrain base.
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)

    # def model_train_init(self):
    #     for module in self.fix_modules:
    #         for p in eval(f"self.{module}").parameters():
    #             p.requires_grad_(False)
    #         eval(f"self.{module}").apply(fix_bn)
    
    def model_train_init(self):
        # 1. 执行原作者的全体冻结（锁死共享金字塔、检测头等基座）
        for module in self.fix_modules:
            for p in eval(f"self.{module}").parameters():
                p.requires_grad_(False)
            eval(f"self.{module}").apply(fix_bn)
            
        # ==================== [FG-SSPF: 极简轻量适配解冻策略] ====================
        print("\n[FG-SSPF 训练控制] Stage 2: 正在执行轻量适配解冻策略...")
        
        # a. 唤醒身份识别模块
        for p in self.pyramid_backbone.modality_embedding.parameters():
            p.requires_grad_(True)
        for p in self.pyramid_backbone.modality_proj.parameters():
            p.requires_grad_(True)
            
        # b. 唤醒缓冲投影层
        for p in self.pyramid_backbone.specific_proj.parameters():
            p.requires_grad_(True)
            
        # c. 唤醒门控调音台
        for p in self.pyramid_backbone.gate_mlp.parameters():
            p.requires_grad_(True)
            
        # ==================== [终极防爆手术] ====================
        # 极其关键：因为前面全局执行了 fix_bn (eval模式)，导致 specific_proj 
        # 还在用 Stage 1 雷达的极小方差，直接导致相机特征爆炸！
        # 我们必须强制把它拉回 train 模式，让它计算相机自己真实的方差来削峰！
        self.pyramid_backbone.specific_proj.train()
        print("[FG-SSPF 训练控制] 已强制唤醒 specific_proj 的 BN 层，防爆装甲就绪！")
        # =======================================================
            

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1
        modality_name = modality_name[0].lstrip('inputs_')

        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
        feature = eval(f"self.aligner_{modality_name}")(feature)

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop(
                    (int(H*eval(f"self.crop_ratio_H_{modality_name}")), int(W*eval(f"self.crop_ratio_W_{modality_name}")))
                )(feature)

            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                })
        
        # multiscale fusion. 
        # feature, occ_map_list = self.pyramid_backbone.forward_single(feature)

        # if self.shrink_flag:
        #     feature = self.shrink_conv(feature)

        # cls_preds = self.cls_head(feature)
        # reg_preds = self.reg_head(feature)
        # dir_preds = self.dir_head(feature)

        # output_dict.update({'cls_preds': cls_preds,
        #                     'reg_preds': reg_preds,
        #                     'dir_preds': dir_preds})
        # output_dict.update({'occ_single_list': 
        #                     occ_map_list})
        # return output_dict
        # [FG-SSPF 修改] 传入身份参数 modality_name，接收 4 个返回值
        feature, occ_map_list, shared_feature_out, alpha = self.pyramid_backbone.forward_single(feature, modality_name)

        if self.shrink_flag:
            feature = self.shrink_conv(feature)

        cls_preds = self.cls_head(feature)
        reg_preds = self.reg_head(feature)
        dir_preds = self.dir_head(feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        output_dict.update({'occ_single_list': occ_map_list})
        
        # [FG-SSPF 修改] 存入共享特征和 alpha 以备计算 Loss
        output_dict.update({
            'shared_feature_out': shared_feature_out,
            'alpha': alpha
        })
        return output_dict
