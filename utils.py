import torch
from torchvision.transforms import RandomResizedCrop



class RandomResizedCropVideo(RandomResizedCrop):
	"""
		crop first then resize(rescale)

		custom transforms for video clip	
		Ref: https://github.com/pytorch/vision/tree/master/torchvision/transforms
		Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

	"""
	def __init__(self, size, scale=(0.08, 1), 
				 ratio=(3.0 / 4.0, 4.0 / 3.0),
				 interpolation_mode = 'bilinear'):
		if isinstance(size, tuple):
			assert len(size) == 2, "size should be tuple (height, width)"
			self.size = size
		else:
			self.size = (size, size)

		self.interpolation_mode = interpolation_mode
		self.scale = scale
		self.ratio = ratio

	def __call__(self, clip):
		# transform the clip shape from [T, H, W, C] to [C, T, H, W]
		clip = clip.permute(3, 0, 1, 2)
		i, j, h, w = self.get_params(clip, self.scale, self.ratio)
		assert len(clip.size()) == 4
		clip = clip[..., i:i+h, j:j+w]
		clip = torch.nn.functional.interpolate(
        	clip.float(), size=self.size, mode=self.interpolation_mode)
		return clip

		

class ToTensorVideo(object):

	def __call__(self, clip):
		return clip.float() / 255.0


		
		


		
		

		