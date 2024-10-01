import torch
from torchinfo import summary

# Import VOLO từ thư mục models trong timm
from timm.models.volo import volo_d1_224  # Hoặc volo_d2, volo_d3 tùy thuộc vào phiên bản

# Khởi tạo mô hình VOLO
model = volo_d1_224(pretrained=True)

# Hiển thị thông tin mô hình bằng torchinfo
summary(model, input_size=(16, 3, 224, 224))  # Batch size = 16, ảnh 224x224