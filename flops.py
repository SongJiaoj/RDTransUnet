import torch
import time
# from nets.CV-Main import CV-Main as create_model
# from model.MySeg_Model import MySegNet as create_model
# from UC_Flops import UC_Res_Swin_ECA as create_model
from skip5_flop import skip5 as create_model
#from PVTFormer import PVTFormer as create_model
#from model.deeplabv3 import DeepLab
# from model.UNetPlus import UnetPlusPlus as create_model
from calflops import calculate_flops
import Config as config

model = create_model(1, 1)
flops, macs, params = calculate_flops(model, input_shape=(2, 1, 560, 560))
# model = DeepLab(
#         num_classes=1,
#         pretrained=False
#     )
#
# flops, macs, params = calculate_flops(model, input_shape=(2, 3, 128, 128))
print(flops, macs, params)


# def calculate_fps(model, input_image):
#     # 测量模型处理一帧图像所花费的时间
#     start_time = time.time()
#     net = model(1, 1)
#     # 假设模型接受的输入是图像 input_image，进行模型推理
#     output = net(input_image)
#     end_time = time.time()
#
#     # 计算处理一帧图像所花费的时间
#     processing_time = end_time - start_time
#
#     # 计算帧率
#     fps = 1 / processing_time
#
#     return fps
#
#
# # 假设 model 是你的分割模型，input_image 是输入的图像
# input_image = torch.randn(1, 1, 128, 128)
# fps = calculate_fps(create_model, input_image)
# print("FPS:", fps)
