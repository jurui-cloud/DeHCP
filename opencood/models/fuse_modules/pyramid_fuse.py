# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature


def weighted_fuse(x, score, record_len, affine_matrix, align_corners):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)
    
    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)
        
    record_len : list
        shape: (B)
        
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3) 
    """

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
        scores_in_ego = torch.softmax(scores_in_ego, dim=0)
        scores_in_ego = torch.where(torch.isnan(scores_in_ego), 
                                    torch.zeros_like(scores_in_ego, device=scores_in_ego.device), 
                                    scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        # ==================== [FG-SSPF: 1. Modality Embedding 初始化] ====================
        self.emb_dim = 16 # 师兄设定的嵌入维度
        # 建立字符串到数字的映射字典
        self.modality_dict = {'m1': 0, 'm2': 1, 'm3': 2, 'm4': 3}
        # 注册 Embedding 层，容量为 4
        self.modality_embedding = nn.Embedding(len(self.modality_dict), self.emb_dim)
        # 因为输入的特征通道数默认是 input_channels=64，我们需要把 16 维映射到 64 维
        self.modality_proj = nn.Linear(self.emb_dim, input_channels)
        # print(f"\n[FG-SSPF 模块注入] ✅ 模态字典构建完毕! 容量: {len(self.modality_dict)}, 映射维度: {self.emb_dim} -> {input_channels}\n")
        # =================================================================================
        
        # ==================== [FG-SSPF: 方案 B - 增加协议投影层 (Projector)] ====================
        # 按照师兄蓝图：1x1卷积 + BN + ReLU，作为物理特征与身份标签融合后的缓冲搅拌机
        self.shared_proj = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        self.specific_proj = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        print("[FG-SSPF 模块注入] ✅ 共享/特有协议投影层 (Projector) 初始化完毕!\n")
        # =====================================================================================


        # ==================== [FG-SSPF: 克隆特有金字塔 & 门控 MLP] ====================
        import copy
        print("[FG-SSPF 模块注入] 正在克隆 Specific 金字塔分支 (Deepcopy)...")
        # 直接在内存中完美复制一套参数独立的 ResNet 和 Deblocks
        self.specific_resnet = copy.deepcopy(self.resnet)
        self.specific_deblocks = copy.deepcopy(self.deblocks)

        self.out_channels = 384
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.out_channels + self.emb_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.out_channels),
            nn.Sigmoid()
        )
        # 【黑魔法防崩】初始化 Gate 的偏置为负数，让训练初期 alpha 接近 0
        # 强迫网络优先把 Shared 分支学好，再慢慢打开 Specific 分支！
        nn.init.zeros_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, -2.0)
        print("[FG-SSPF 模块注入] ✅ 双线金字塔 & Gate MLP 部署完毕!\n")
        # ==============================================================================

        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    # def forward_single(self, spatial_features):
    #     """
    #     This is used for single agent pass.
    #     """
    #     feature_list = self.get_multiscale_feature(spatial_features)
    #     occ_map_list = []
    #     for i in range(self.num_levels):
    #         occ_map = eval(f"self.single_head_{i}")(feature_list[i])
    #         occ_map_list.append(occ_map)
    #     final_feature = self.decode_multiscale_feature(feature_list)

    #     return final_feature, occ_map_list
            
    # [FG-SSPF 修改] 增加 modality_name 参数接收身份
    def forward_single(self, spatial_features, modality_name):
        """
        This is used for single agent pass.
        """
        device = spatial_features.device
        
        # ==================== [FG-SSPF: 单车双分支前向传播] ====================
        # 1. 贴上身份标签
        mod_idx = self.modality_dict.get(modality_name, 0)
        mod_tensor = torch.tensor([mod_idx] * spatial_features.shape[0], dtype=torch.long, device=device)
        emb = self.modality_embedding(mod_tensor) 
        emb_proj = self.modality_proj(emb)        
        emb_broadcast = emb_proj.unsqueeze(-1).unsqueeze(-1)
        condition_features = spatial_features + emb_broadcast
        
        # 2. 经过缓冲投影层
        shared_feat_in = self.shared_proj(spatial_features)
        specific_feat_in = self.specific_proj(condition_features)

        # 3. 双线特征提取
        shared_feature_list = self.resnet(shared_feat_in)
        specific_feature_list = self.specific_resnet(specific_feat_in)

        # 4. 前景掩码（单车不需要协同加权，所以只算出来存着，不干预特征）
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(shared_feature_list[i])
            occ_map_list.append(occ_map)

        # 5. 解码
        shared_feature_out = self.decode_multiscale_feature(shared_feature_list)
        specific_feature_out = self.decode_specific_feature(specific_feature_list)

        # 6. 门控计算与终极融合
        import torch.nn.functional as F
        pooled_shared = F.adaptive_avg_pool2d(shared_feature_out, 1).flatten(1)
        gate_input = torch.cat([pooled_shared, emb], dim=1)
        alpha = self.gate_mlp(gate_input).unsqueeze(-1).unsqueeze(-1)

        final_feature = shared_feature_out + alpha * specific_feature_out


        # 把 shared_feature_out 和 alpha 也传出去
        return final_feature, occ_map_list, shared_feature_out, alpha

    

    def decode_specific_feature(self, x):
        """ [FG-SSPF 专用]: 为特有分支准备的解码器 (与父类 decode_multiscale_feature 结构完全一致) """
        ups = []
        for i in range(self.num_levels):
            if len(self.specific_deblocks) > 0:
                ups.append(self.specific_deblocks[i](x[i]))
            else:
                ups.append(x[i])
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.specific_deblocks) > self.num_levels:
            x = self.specific_deblocks[-1](x)
        return x

    def forward_collab(self, spatial_features, record_len, affine_matrix, agent_modality_list = None, cam_crop_info = None):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        
        # ==================== [FG-SSPF: 2. 生成模态条件化特征] ====================
        # agent_modality_list 长这样: ['m1', 'm2', 'm1']
        device = spatial_features.device
        N_total = spatial_features.shape[0] # 当前批次所有车辆的总数
        
        if agent_modality_list is not None:
            # 1. 把字符串 ['m1', 'm2'] 翻译成整数 [0, 1]
            mod_indices = [self.modality_dict.get(m, 0) for m in agent_modality_list] 
            mod_tensor = torch.tensor(mod_indices, dtype=torch.long, device=device) # [N]
            
            # 2. 提取 Embedding 并通过全连接层映射到 64 维
            emb = self.modality_embedding(mod_tensor) # [N, 16]
            emb_proj = self.modality_proj(emb)        # [N, 64]
            
            # 3. 变形为 [N, 64, 1, 1] 以便和 spatial_features [N, 64, H, W] 利用广播机制相加
            emb_broadcast = emb_proj.unsqueeze(-1).unsqueeze(-1)
            
            # 这个 condition_features 就是给 Specific Branch (特有分支) 准备的“方言”初始材料！
            condition_features = spatial_features + emb_broadcast

        else:
            # 防御性编程：如果没有传列表（比如推理异常），就不加条件
            condition_features = spatial_features
        # ==========================================================================

        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}


        feature_list = self.get_multiscale_feature(spatial_features)

        # # ==================== [专供 Figure 5 ] ====================
        # if not self.training: 
        #     # 只取第1层的第12通道，统一保存在 figure_5 文件夹下
        #     plot_feature(feature_list[0], channel=[12], save_path='vis_result/figure_5', flag='eval')
        # # =======================================================================

        # ==================== [FG-SSPF: 核心双分支前向传播] ====================
        # 1. 经过前置缓冲投影层
        shared_feat_in = self.shared_proj(spatial_features)
        specific_feat_in = self.specific_proj(condition_features)

        # 2. 提取多尺度特征 (走各自的金字塔骨干)
        shared_feature_list = self.resnet(shared_feat_in)
        specific_feature_list = self.specific_resnet(specific_feat_in)

        shared_fused_list = []
        specific_fused_list = []
        occ_map_list = []

        for i in range(self.num_levels):
            # 前景掩码统一由 shared 共享特征生成 (保证看哪里是一致的)
            occ_map = eval(f"self.single_head_{i}")(shared_feature_list[i])
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 
                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)
                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]
                score = score * cam_crop_mask

            # 3. 多车加权协同融合 (V2V传输)
            shared_fused_list.append(weighted_fuse(shared_feature_list[i], score, record_len, affine_matrix, self.align_corners))
            specific_fused_list.append(weighted_fuse(specific_feature_list[i], score, record_len, affine_matrix, self.align_corners))

        # 4. 多尺度解码还原
        shared_feature_out = self.decode_multiscale_feature(shared_fused_list)
        specific_feature_out = self.decode_specific_feature(specific_fused_list)

        # 5. 智能门控 (Gate Aggregation)
        # 获取 Batch 中每个 Ego 主车的身份证 (在 record_len 分段中，主车永远是第 0 个)
        ego_indices = []
        current_idx = 0
        for b_len in record_len:
            ego_indices.append(int(current_idx)) # 强制转 int，防止 PyTorch Tensor 污染
            current_idx += int(b_len)
            
        if agent_modality_list is not None:
            ego_modalities = []
            for idx in ego_indices:
                # ==================== [FG-SSPF: 防越界装甲] ====================
                # 如果数据集偷懒只传了一个 ['m1']，idx 就会越界，此时咱们安全地取第 0 个即可。
                # 如果是多模态混合，长度正常，就精准取出对应车辆的身份。
                safe_idx = idx if idx < len(agent_modality_list) else 0
                ego_modalities.append(self.modality_dict.get(agent_modality_list[safe_idx], 0))
                # ===============================================================
        else:
            ego_modalities = [0] * len(record_len)
            
        ego_tensor = torch.tensor(ego_modalities, dtype=torch.long, device=device) # [Batch]
        ego_emb = self.modality_embedding(ego_tensor) # [Batch, 16]

        # GAP 提取全局语义
        import torch.nn.functional as F
        pooled_shared = F.adaptive_avg_pool2d(shared_feature_out, 1).flatten(1) # [Batch, 384]
        gate_input = torch.cat([pooled_shared, ego_emb], dim=1) # [Batch, 400]
        
        # 计算 alpha 并扩充维度以便相乘
        alpha = self.gate_mlp(gate_input).unsqueeze(-1).unsqueeze(-1) # [Batch, 384, 1, 1]

        # 6. 终极奥义：物理共性 + \alpha * 模态特性
        fused_feature = shared_feature_out + alpha * specific_feature_out
        # =======================================================================

        return fused_feature, occ_map_list, shared_feature_out, alpha






        # fused_feature_list = []
        # occ_map_list = []
        # for i in range(self.num_levels):
        #     occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
        #     occ_map_list.append(occ_map)
        #     score = torch.sigmoid(occ_map) + 1e-4

        #     if crop_mask_flag and not self.training:
        #         cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
        #         _, _, H, W = cam_crop_mask.shape
        #         for cam_modality in cam_modality_set:
        #             crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
        #             crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

        #             start_h = int(H//2-crop_H//2)
        #             end_h = int(H//2+crop_H//2)
        #             start_w = int(W//2-crop_W//2)
        #             end_w = int(W//2+crop_W//2)

        #             cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
        #             cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

        #         score = score * cam_crop_mask

        #     fused_feature_list.append(weighted_fuse(feature_list[i], score, record_len, affine_matrix, self.align_corners))
        # fused_feature = self.decode_multiscale_feature(fused_feature_list)
        # print(f"\n[FG-SSPF] 最终融合特征的维度是: {fused_feature.shape}\n")
        # shared_feature_out = fused_feature
        
        # # return fused_feature, occ_map_list 
        # return fused_feature, occ_map_list, shared_feature_out