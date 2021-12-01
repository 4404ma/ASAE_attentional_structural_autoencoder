import torch
from torchvision import transforms
import numpy as np
import utils
from model.attentional_structural_autoencoder import ASAE
from torchvision.datasets import CIFAR10


def train():

    # set hyperparameter
    EPOCH = 20
    pre_epoch = 0
    batch_size = 100
    LR = 0.01

    # DataLoader
    train_dataset = CIFAR10(root='./dataset_cifar10', transform=transforms.ToTensor(), train=True, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # define model
    model = ASAE()

    # define loss funtion & optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    # train
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        total = 0.0
        for i, (images, _) in enumerate(train_loader):
            # prepare dataset
            length = len(train_loader)
            images= images
            optimizer.zero_grad()

            # forward & backward
            outputs = model(images)
            recon_loss = torch.mean(torch.abs(images - outputs))
            loss = recon_loss * np.prod((batch_size, 3, 32, 32)) * 0.01 # TODO: use hyperparameter
            loss.backward()
            optimizer.step()

            # print loss in each 200th batch
            sum_loss += loss.item()
            if i % 200 == 199:
                pic = outputs.squeeze(0)
                pic = transforms.ToPILImage()(pic[0])
                pic.save('./results/results_{epoch}_{iter}.jpg'.format(epoch=epoch, iter=i))
                print('[epoch:%d, iter:%d] Loss: %.03f '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1)))
                torch.save(model, './results/2021_11_30_{epoch}_{iter}.pt'.format(epoch=epoch, iter=i))
