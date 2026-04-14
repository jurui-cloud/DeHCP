import torch

# 1. 加载 Stage 1 的极品金丹
ckpt = torch.load("opencood/logs/HEAL_m1_based/stage1/m1_base/net_epoch_bestval_at29.pth", map_location='cpu')
weights = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

# 2. 找出一个明确的 specific_resnet 层的 key (最好是带 weight 的卷积层)
specific_keys = [k for k in weights.keys() if 'specific_resnet' in k and 'weight' in k]

if not specific_keys:
    print("\n[检查结果] 连 specific_resnet 的参数都没找到！前 20 个键名是：")
    for k in list(weights.keys())[:20]:
        print("  ", k)
else:
    # 随便挑一个 specific 层
    s_key = specific_keys[0]
    # 构造出对应的 shared 层的 key (直接把 specific_resnet 替换成 resnet，绝对精准)
    shared_key = s_key.replace('specific_resnet', 'resnet')
    
    if shared_key in weights:
        shared_w = weights[shared_key]
        specific_w = weights[s_key]
        
        try:
            diff = torch.abs(shared_w - specific_w).mean().item()
            print("\n================ [ Stage 1 极品金丹质检报告 ] ================")
            print(f"Shared 抽样层:  {shared_key}")
            print(f"Specific 抽样层: {s_key}")
            print(f"--> 参数平均差异 (Mean Absolute Difference): {diff:.8f}")
            
            if diff == 0:
                print("\n结论：差异为 0！说明在 Stage 1 中 Specific 分支根本没学到属于自己的特有特征，它和 Shared 分支完全是克隆人！")
            else:
                print("\n结论：已有差异！说明在 Stage 1 中，Specific 分支已经成功形成了属于雷达的【特有互补空间】！")
            print("==============================================================")
        except Exception as e:
            print(f"\n对比时发生异常: {e}")
            print(f"Shared shape: {shared_w.shape}, Specific shape: {specific_w.shape}")
    else:
        print(f"\n找到了 specific 层 {s_key}，但它居然没有对应的 shared 层 {shared_key}！")