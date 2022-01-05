import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# https://dev.to/afrozchakure/all-you-need-to-know-about-yolo-v3-you-only-look-once-e4m
# https://res.cloudinary.com/practicaldev/image/fetch/s--5kVLEyT3--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zdmk2adlckbnm8k9n0p8.png
# https://www.researchgate.net/profile/Dai-Wei-8/publication/348001479/figure/fig3/AS:974414304772108@1609329882943/YOLOv3-architecture-for-object-detection-Note-DBL-is-the-basic-component-of-Darknet53.png

# https://github.com/DeNA/PyTorch_YOLOv3
# https://github.com/mahdi-darvish/YOLOv3-from-Scratch-Analaysis-and-Implementation/blob/main/implementations/model.py

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_conv_layers = 0

class ConvBlock(nn.Sequential):
	def __init__(self, *args, **kwargs):
		super(ConvBlock, self).__init__()
		global total_conv_layers
		total_conv_layers += 1
		self.add_module('Conv2d', nn.Conv2d(*args, **kwargs))
		# self.add_module('BatchNorm2d', nn.BatchNorm2d(kwargs['out_channels']))
		# self.add_module('ReLU', nn.ReLU())

class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()		
		self.conv_block_0 = ConvBlock(in_channels=channels   , out_channels=channels//2, kernel_size=1)
		self.conv_block_1 = ConvBlock(in_channels=channels//2, out_channels=channels   , kernel_size=3, padding=1)

	def forward(self, x):
		residual = self.conv_block_0(x)
		residual = self.conv_block_1(residual)
		return x + residual

class DBLBlock(nn.Module):
	def __init__(self, channels_in, channels):
		super(DBLBlock, self).__init__()

		self.c1 = ConvBlock(in_channels=channels_in, out_channels=channels//2, kernel_size=1)
		self.c2 = ConvBlock(in_channels=channels//2, out_channels=channels   , kernel_size=3, padding=1)
		self.c3 = ConvBlock(in_channels=channels   , out_channels=channels//2, kernel_size=1)
		self.c4 = ConvBlock(in_channels=channels//2, out_channels=channels   , kernel_size=3, padding=1)
		self.c5 = ConvBlock(in_channels=channels   , out_channels=channels//2, kernel_size=1)
		
		# self.c6_left  = ConvBlock(in_channels=channels//2, out_channels=255, kernel_size=1) # Replace 255 with (5+n)*3
		# self.c6_right = ConvBlock(in_channels=channels//2, out_channels=channels//4, kernel_size=1)

	def forward(self, x):
		out = self.c1(x)
		out = self.c2(out)
		out = self.c3(out)
		out = self.c4(out)
		out = self.c5(out)
		return out

		# out_yolo = self.c6_left(out)
		# out = self.c6_right(out)

		# return out_yolo, out

class YoloBlock(nn.Module):
	def __init__(self, in_channels, n_classes):
		super(YoloBlock, self).__init__()

		self.anchor_size = ( 5 + n_classes )
		self.conv = ConvBlock(in_channels=in_channels, out_channels=self.anchor_size * 3, kernel_size=1)
	
	def forward(self, x):
		out = self.conv(x)
		print("[YoloBlock]", out.data_ptr(), out.shape)
		grid_cells = out.shape[-1]
		print(f"[YoloBlock] Grid size = {grid_cells}")
		out2 = out.view(-1, 3, self.anchor_size, grid_cells, grid_cells).permute(0, 3, 4, 1, 2)
		print("[YoloBlock]", out.data_ptr(), out.shape)
		print("[YoloBlock]", out2.data_ptr(),out2.shape)

class Darknet53(nn.Module):
	def __init__(self):
		super(Darknet53, self).__init__()

		self.first_layer  = ConvBlock(in_channels= 3, out_channels=32, kernel_size=3, padding=1)
		# 416x416
		self.downsample_1 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
		self.block_1 = nn.Sequential( *[ ResidualBlock(channels= 64) for i in range(1)] )
		# 208x208
		self.downsample_2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
		self.block_2 = nn.Sequential( *[ ResidualBlock(channels=128) for i in range(2)] )
		# 104x104
		self.downsample_3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
		self.block_3 = nn.Sequential( *[ ResidualBlock(channels=256) for i in range(8)] )
		# 52x52
		self.downsample_4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
		self.block_4 = nn.Sequential( *[ ResidualBlock(channels=512) for i in range(8)] )
		# 26x26
		self.downsample_5 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)
		self.block_5 = nn.Sequential( *[ ResidualBlock(channels=1024) for i in range(4)] )
		# 13x13
		
	def forward(self, x):

		out = self.first_layer(x)
		
		out = self.downsample_1(out)
		out = self.block_1(out)
		
		out = self.downsample_2(out)
		out = self.block_2(out)
		
		out = self.downsample_3(out)
		out = self.block_3(out)
		out_52x52 = out

		out = self.downsample_4(out)
		out = self.block_4(out)
		out_26x26 = out

		out = self.downsample_5(out)
		out = self.block_5(out)
		out_13x13 = out

		return out_52x52, out_26x26, out_13x13

class Upsampling(nn.Module):
	def __init__(self):
		super(Upsampling, self).__init__()
		self.darknet53 = Darknet53()

		self.DBL_1 = DBLBlock(channels_in=1024, channels=1024)

		self.yolo1_c1 = ConvBlock(in_channels= 512, out_channels=1024, kernel_size=3, padding=1)
		self.yolo_13x13 = YoloBlock(1024, 80)
		self.pre_upscale_conv_1 = ConvBlock(in_channels=512, out_channels=256, kernel_size=1)

		self.DBL_2 = DBLBlock(channels_in= 768, channels= 512)

		self.yolo2_c1 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		self.yolo2_c2 = ConvBlock(in_channels=512, out_channels=255, kernel_size=1)  # Replace 255 with (5+n)*3
		self.pre_upscale_conv_2 = ConvBlock(in_channels=256, out_channels=128, kernel_size=1)

		self.DBL_3 = DBLBlock(channels_in= 384, channels= 256)
		self.yolo3_c1 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.yolo3_c2 = ConvBlock(in_channels=256, out_channels=255, kernel_size=1)  # Replace 255 with (5+n)*3

	def forward(self, x):
		out_52x52, out_26x26, out_13x13 = self.darknet53(x)

		print("52x52", out_52x52.shape)
		print("26x26", out_26x26.shape)
		print("13x13", out_13x13.shape)

		out = yolo1 = self.DBL_1(out_13x13)
		yolo1 = self.yolo1_c1(yolo1)
		self.yolo_13x13(yolo1)

		out = self.pre_upscale_conv_1(out)
		out = F.interpolate(out, scale_factor=2)
		out = torch.cat((out, out_26x26), axis=1)

		out = yolo2 = self.DBL_2(out)
		yolo2 = self.yolo2_c1(yolo2)
		yolo2 = self.yolo2_c2(yolo2)

		out = self.pre_upscale_conv_2(out)
		out = F.interpolate(out, scale_factor=2)
		out = torch.cat((out, out_52x52), axis=1)

		yolo3 = self.DBL_3(out)
		yolo3 = self.yolo3_c1(yolo3)
		yolo3 = self.yolo3_c2(yolo3)

		print("yolo1", yolo1.shape)
		print("yolo2", yolo2.shape)
		print("yolo3", yolo3.shape)


		return yolo1, yolo2, yolo3

network = Upsampling()


# x = torch.rand(1, 3, 416, 416)
# y = network(x)

# torch.onnx.export(network, x, 'model.onnx', export_params=True)
# torch.onnx.export(network, torch.rand(1, 3, 416, 416), 'model_no_params.onnx', 
# 	export_params=False, training=torch.onnx.TrainingMode.TRAINING, )

print("Number of paramenters :", "{:,}".format(count_parameters(network)))
print(f"Total number of convolutional layers: {total_conv_layers}")

torch.onnx.export(network, torch.rand(1, 3, 416, 416), 'model_no_params.onnx', opset_version=12,
	export_params=False, training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)

