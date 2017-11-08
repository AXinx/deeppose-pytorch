
from alex_net import AlexNet
from mean_squared_error import mean_squared_error

import os
import random
import time
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from torch.autograd import Variable


def load_dataset(path):
    images = []
    poses = []
    visibilities = []
    for line in open(path):
        line_split = line[:-1].split(',')
        images.append(line_split[0])
        x = torch.Tensor(map(float, line_split[1:]))
        x = x.view(-1, 3)
        pose = x[:, :2]
        visibility = x[:, 2].clone().view(-1, 1).expand_as(pose)
        poses.append(pose)
        visibilities.append(visibility)
    return images, poses, visibilities


def get_optimizer(opt, model):
    if opt == 'MomentumSGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters())
    return optimizer

def train(model, optimizer, train_iter, log_interval, start_time, gpu=False):
    model.train()
    for iteration, batch in enumerate(tqdm(train_iter, desc='this epoch'), 1):
        image, pose, visibility = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if gpu:
            image, pose, visibility = image.cuda(), pose.cuda(), visibility.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = mean_squared_error(output, pose, visibility)
        loss.backward()
        optimizer.step()

#model
model = AlexNet()
optimizer = get_optimizer('MomentumSGD', model)

#train
images, poses, visibilities = load_dataset(path)

start_time = time.time()
train(model, optimizer, train_iter, start_time)


