'''
	resnet 3D CNN + kinetics-400 pretrained + hmdb51 training
	Reference paper: Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?
'''

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummaryX import summary
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pretrained_model as model
import os
import time
import copy
import utils


best_val_acc = 0.0
best_val_model = None
EPOCH = 10
BSZ = 2

transform = transforms.Compose([
		utils.RandomResizedCropVideo(64),
		utils.ToTensorVideo()])

data_root = '../.data'

train_loader = DataLoader(datasets.HMDB51(root = os.path.join(data_root, 'hmdb51'), 
										  annotation_path = os.path.join(data_root,'splits'),
										  frames_per_clip = 8, fold = 1, train = True, 
										  transform = transform), 
						  batch_size = BSZ, shuffle = True)

val_loader = DataLoader(datasets.HMDB51(root = os.path.join(data_root, 'hmdb51'), 
										annotation_path = os.path.join(data_root, 'splits'),
				  					    frames_per_clip = 8, fold = 2, train = False, 
				  					    transform = transform), 
						batch_size = BSZ, shuffle = True)

test_loader = DataLoader(datasets.HMDB51(root = os.path.join(data_root, 'hmdb51'), 
										 annotation_path = os.path.join(data_root, 'splits'),
				   						 frames_per_clip = 8, fold = 3, train = False, 
				   						 transform = transform), 
						batch_size = BSZ, shuffle = True)



sample_size = next(iter(train_loader))[0].size()
assert sample_size == torch.Size([BSZ, 3, 8, 64, 64]), 'sample_size is {}'.format(sample_size)


model_ft = model.get_model()
criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters())
log_name = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
writer = SummaryWriter(os.path.join('log/', log_name))

# print(model_ft)
# summary(model_ft, torch.zeros(1, 3, 8, 64, 64))


for epoch in range(EPOCH):

	start_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
	print(f'epoch {epoch} | start time {start_time}')

	train_loss, train_acc = model.train(model_ft, train_loader, criterion, optimizer, epoch, writer)
	val_loss, val_acc = model.evaluate(model_ft, val_loader, criterion, epoch, writer)

	print(f'train loss {train_loss:03f} | train accuracy {train_acc:03f}')
	print(f'val loss {val_loss:03f} | val accuracy {val_acc:03f}\n')

	if val_acc > best_val_acc:
		best_val_acc = val_acc
		best_val_model = copy.deepcopy(model_ft.state_dict())

model_ft.load_state_dict(best_val_model)
test_loss, test_acc = model.evaluate(model_ft, test_loader, criterion, EPOCH+1, writer)
print(f'test_loss {test_loss:03f} | test acc {test_acc:03f}')



