
from torchinfo import summary

# Import VOLO từ thư mục models trong timm

#from timm.models.volo import volo_d1_224

from mivolo.model.mivolo_model import MiVOLOModel, PatchEmbed

#from mivolo.model.mivolo_model import PatchEmbed
#model = volo_d1_224(pretrained=True)
#model = mivolo_d2_224(pretrained=True)
# model = PatchEmbed(
#     img_size=224,        # Kích thước ảnh đầu vào
#     stem_conv=True,      # Sử dụng stem conv
#     stem_stride=2,       # Bước stride
#     patch_size=8,        # Kích thước patch
#     hidden_dim=64,       # Kích thước ẩn
#     embed_dim=384        # Kích thước embedding
# )
model = MiVOLOModel(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12))

# Hiển thị thông tin mô hình bằng torchinfo
summary(model, input_size=(1, 3, 224, 224))  # Batch size = 16, ảnh 224x224