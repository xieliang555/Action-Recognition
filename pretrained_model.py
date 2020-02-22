import torchvision
import torch.nn as nn
import copy
import utils
import time
from collections import OrderedDict

def get_model():
	model_ft = torchvision.models.video.r3d_18(pretrained = True)

	model_ft = nn.Sequential(OrderedDict([
					('stem', model_ft.stem),
					('avgpool', model_ft.avgpool),
					('flatten', nn.Flatten()),
					('fc', nn.Linear(64, 5))]))
	return model_ft


def train(model, train_loader, criterion, optimizer, epoch, writer):
	model.train()
	epoch_loss = 0.0
	epoch_accuracy = 0.0
	running_loss = 0.0
	running_accuracy = 0.0
	for batch_idx, batch in enumerate(train_loader):
		start_time = time.time()
		inputs = batch[0]
		target = batch[2]
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, target)
		loss.backward()
		optimizer.step()

		predict = outputs.max(1)[1]
		accuracy = predict.eq(target).sum().item() / len(target)
		epoch_accuracy += accuracy
		running_accuracy += accuracy
		epoch_loss += loss.item()
		running_loss += loss.item()

		# end_time = time.time()
		# mins = int(end_time - start_time) // 60
		# secs = int(end_time- start_time) % 60
		# print(f'\ttrain epoch {epoch} | {batch_idx}/{len(train_loader)} | '
		# 	f'duration {mins}m:{secs}s | loss {loss.item():03f} | accuracy {accuracy:03f}')

		if batch_idx % 1000 == 999:
			writer.add_scalar('train loss',
							   running_loss / 1000,
							   epoch * len(train_loader) + batch_idx)
			writer.add_scalar('train accuracy',
							   running_accuracy / 1000,
							   epoch * len(train_loader) +batch_idx)

			running_loss = 0.0
			running_accuracy = 0.0

	return epoch_loss / len(train_loader), epoch_accuracy / len(train_loader)


def evaluate(model, val_loader, criterion, epoch, writer):
	model.eval()
	epoch_loss = 0.0
	epoch_accuracy = 0.0
	best_epoch_acc = 0.0
	for batch_idx, batch in enumerate(val_loader):
		start_time = time.time()
		inputs = batch[0]
		target = batch[2]
		outputs = model(inputs)
		loss = criterion(outputs, target)
		loss.backward()

		predict = outputs.max(1)[1]
		accuracy = predict.eq(target).sum().item() / len(target)
		epoch_accuracy += accuracy
		epoch_loss += loss.item()

		error_idx = predict.ne(target)
		error_predict = predict[error_idx]
		error_target = target[error_idx]
		# transform shape from [N, C, T, H, W] to [N, T, C, H, W]
		error_video = inputs[error_idx].permute(0, 2, 1, 3, 4)
		ground_truth = ['brush hair', 'cartwheel', 'catch', 'chew', 'clap']
		writer.add_text('predict vs ground truth', 
						ground_truth[error_predict] +'/' + ground_truth[error_target],
						epoch * len(val_loader) + batch_idx)
		writer.add_video('video',
						 error_video,
						 epoch * len(val_loader) + batch_idx)


		# end_time = time.time()
		# mins = int(end_time - start_time) // 60
		# secs = int(end_time - start_time) % 60
		# print(f'\teval epoch {epoch} | {batch_idx}/{len(val_loader)} | '
		# 	f'duration {mins}m:{secs}s | loss {loss.item():03f} | accuracy {accuracy:03f}')

	return epoch_loss / len(val_loader) , epoch_accuracy / len(val_loader)


