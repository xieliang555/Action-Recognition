import torchvision
import torch.nn as nn
import torch.nn.functional a F
import copy
import utils
import time
from collections import OrderedDict
import matplotlib.pyplot as plt


def get_model():
    model_ft = torchvision.models.video.r3d_18(pretrained=True)
    inp_feature = model_ft.fc.in_features
    model_ft.fc = nn.Linear(inp_feature, 51)

#    model_ft = nn.Sequential(OrderedDict([
#                ('stem', model_ft.stem),
#                ('avgpool', model_ft.avgpool),
#                ('flatten', nn.Flatten()),
#                ('fc', nn.Linear(64, 51))]))
    return model_ft


def train(model, train_loader, criterion, optimizer, epoch, writer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for batch_idx, batch in enumerate(train_loader):
        start_time = time.time()
        inputs = batch[0].to(device)
        target = batch[2].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        predict = outputs.max(1)[1]
        running_accuracy += predict.eq(target).sum().item() / len(target)
        running_loss += loss.item()

        if batch_idx % 1000 == 999:
            writer.add_scalar('train loss',
                              running_loss / 1000,
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train accuracy',
                              running_accuracy / 1000,
                              epoch * len(train_loader) + batch_idx)

            running_loss = 0.0
            running_accuracy = 0.0
            break


def evaluate(model, val_loader, writer, device, classes, test_mode=False):
    model.eval()
    epoch_accuracy = 0.0
    for batch_idx, batch in enumerate(val_loader):
        start_time = time.time()
        inputs = batch[0].to(device)
        target = batch[2].to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        probs, predict = outputs.max(1)
        epoch_accuracy += predict.eq(target).sum().item() / len(target)

        if test_mode:
            # change input shape from [N, C, T, H, W] to [N, T, C, H, W]
            inputs = inputs.permute(0,2,1,3,4).mean(0)
            error_selected = target.nq(predict)
            writer.add_video('video for wrong prediction from test dataset',
                             inputs[error_selected],
                             batch_idx)
            
        if batch_idx == 25:
            break
            
    return epoch_accuracy / len(val_loader)