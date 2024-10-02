
from torchinfo import summary
import torch
import torch.nn as nn
from mivolo.model.cross_bottleneck_attn import CrossBottleneckAttn
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.layers.helpers import make_divisible
from timm.layers.mlp import Mlp
from timm.layers.trace_utils import _assert
from timm.layers.weight_init import trunc_normal_
# Import VOLO từ thư mục models trong timm

#from timm.models.volo import volo_d1_224

from mivolo.model.mivolo_model import MiVOLOModel
from mivolo.model.mivolo_model import PatchEmbed
#model = volo_d1_224(pretrained=True)
#model = mivolo_d2_224(pretrained=True)
model = PatchEmbed(
    img_size=224,        # Kích thước ảnh đầu vào
    stem_conv=True,      # Sử dụng stem conv
    stem_stride=2,       # Bước stride
    patch_size=8,        # Kích thước patch
    hidden_dim=64,       # Kích thước ẩn
    embed_dim=384        # Kích thước embedding
)

# Hiển thị thông tin mô hình bằng torchinfo
summary(model, input_size=(1, 3, 224, 224))  # Batch size = 16, ảnh 224x224