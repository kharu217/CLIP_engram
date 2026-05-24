import torch
from model.clip import CLIP
from model.model_configs import clip_config_set
import torchinfo

if __name__ == "__main__":
    # model = CLIP(clip_cfg=clip_config_set.clip_150M_normal).to(device="cuda")
    # print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B normal")

    # model = CLIP(clip_cfg=clip_config_set.clip_1_5B_moe).to(device="cuda")
    # print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B moe")
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model = CLIP(clip_cfg=clip_config_set.clip_130M_normal).to(device="cuda")

    temp_data = (torch.randn((10, 3, 224, 224), device='cuda'),torch.randint(low=0, high=100, size=(10, 77),device='cuda'))
    print(model(temp_data))
    print((torchinfo.summary(model=model, input_data=temp_data).total_params - 65706496)/10**9, "B engram")
