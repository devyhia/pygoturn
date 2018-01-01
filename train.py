# necessary imp rts
import os
import time
import copy
from src import datasets
import argparse
from src import model
import torch
from torch.autograd import Variable
from torchvision import transforms
from src.helper import ToTensor, Normalize, show_batch, scale_ratio, unscale_ratio
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from visdom import Visdom
from PIL import Image, ImageDraw

vis = Visdom()
# constants
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('==> GPU is available :)')

parser = argparse.ArgumentParser(description='GOTURN Training')
parser.add_argument('-n', '--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-fep', '--from-epoch', default=0, type=int, help='starts the sprint from which epoch')
parser.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size (default: 1)')
parser.add_argument('-lr', '--learning-rate', default=1e-6, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('-dir', '--save-directory', default='../checkpoints/', type=str, help='path to save directory')
parser.add_argument('-r', '--resume', default=None, type=str, help='path to resume from')
parser.add_argument('-pr', '--print-every', default=1, type=int, help='print every x amount')

def main():
    global args
    args = parser.parse_args()

    print(args)
    # load dataset
    transform = transforms.Compose([Normalize(), ToTensor()])
    alov = datasets.ALOVDataset('../ALOV/Frames/',
                                '../ALOV/GT/',
                                transform)
    dataloader = DataLoader(alov, batch_size=args.batch_size, shuffle=True)

    # load model
    net = model.GoNet()
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint)

    loss_fn = torch.nn.L1Loss(size_average = False)
    if use_gpu:
        net = net.cuda()
        loss_fn = loss_fn.cuda()
    optimizer = optim.Adam(net.classifier.parameters(), lr=args.learning_rate)  # 

    if os.path.exists(args.save_directory):
        print('Directory %s already exists' % (args.save_directory))
    else:
        os.makedirs(args.save_directory)

    # start training
    net = train_model(net, dataloader, loss_fn, optimizer, args.epochs, args.learning_rate, args.save_directory)

def train_model(model, dataloader, criterion, optimizer, num_epochs, lr, save_dir):
    since = time.time()
    dataset_size = dataloader.dataset.len
    best_loss = np.inf

    loss_history = []

    for epoch in range(args.from_epoch, num_epochs):
        since_epoch = time.time()
        since_batch = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        optimizer = exp_lr_scheduler(optimizer, epoch, lr)
        running_loss = 0.0
        i = 0
        total_i = len(dataloader)
        # iterate over data
        for data in dataloader:
            # get the inputs and labels
            x1, x2, y = data['previmg'], data['currimg'], data['currbb']

            # wrap them in Variable
            if use_gpu:
                x1, x2, y = Variable(x1.cuda()), \
                    Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
            else:
                x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            output = model(x1, x2)
            loss = criterion(output, y)

            # Logging
            vis.text("""
                     Ground Truth: {} <br/>
                     Prediction: {} <br/>
                     Loss: {}
                    """.format(y, output, loss.data[0]),
                    win="Iteration__Text")

            # Visualization
            search_image = data['currimg'].numpy()
            search_image = search_image.reshape(search_image.shape[1:]).transpose((1,2,0))
            search_image += [104, 117, 123]
            search_image = search_image.astype(np.uint8)

            search_image = Image.fromarray(search_image)
            draw = ImageDraw.Draw(search_image)

            draw.rectangle(unscale_ratio * y.data.numpy(), outline='green')
            draw.rectangle(unscale_ratio * output.data.numpy(), outline='red')

            del draw

            vis.image(np.array(search_image).transpose(2, 0, 1), win="Iteration__Image")
            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            if i % args.print_every == 0:
                runtime = time.time() - since_batch
                print('[training] epoch = %d, i = %d / %d, loss = %f, running_loss = %f, runtime= %dm %ds' % (epoch, i, total_i, loss.data[0], running_loss / (i+1), runtime / 60, runtime % 60))
                since_batch = time.time()
            i = i + 1
            running_loss += loss.data[0]


        epoch_loss = running_loss / dataset_size
        runtime = time.time() - since_epoch
        print('+++ Epoch Loss: {:.4f} - Runtime: {:.0f}m {:.0f}s'.format(epoch_loss, runtime / 60, runtime % 60))
        val_loss = evaluate(model, dataloader, criterion, epoch)
        print('+++ Validation Loss: {:.4f}'.format(val_loss))
        path = save_dir + 'model_best_loss.pth'

        loss_history += [[ epoch_loss, val_loss ]]
        vis.line(
            Y=np.array(loss_history),
            win="Iteration__Loss",
            opts=dict(
                legend= ['Training Loss', 'Validation Loss']
            )
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path)

        since_epoch = time.time()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

def evaluate(model, dataloader, criterion, epoch):
    model.eval()
    dataset = dataloader.dataset
    running_loss = 0
    # test on a sample sequence from training set itself
    for i in range(min(64, len(dataloader))):
        sample = dataset[i]
        sample['currimg'] = sample['currimg'][None,:,:,:]
        sample['previmg'] = sample['previmg'][None,:,:,:]
        x1, x2 = sample['previmg'], sample['currimg']
        y = sample['currbb']
        x1 = Variable(x1.cuda() if use_gpu else x1)
        x2 = Variable(x2.cuda() if use_gpu else x2)
        y = Variable(y.cuda() if use_gpu else y, requires_grad=False)
        output = model(x1, x2)

        # Visualization
        search_image = sample['currimg'].numpy()
        search_image = search_image.reshape(search_image.shape[1:]).transpose((1,2,0))
        search_image += [104, 117, 123]
        search_image = search_image.astype(np.uint8)

        search_image = Image.fromarray(search_image)
        draw = ImageDraw.Draw(search_image)

        draw.rectangle(unscale_ratio * y.data.numpy(), outline='green')
        draw.rectangle(unscale_ratio * output.data.numpy(), outline='red')

        del draw

        vis.image(np.array(search_image).transpose(2, 0, 1), win="Iteration__ValidationImage", opts={ 'caption': "Validation Image # {}".format(i) })

        loss = criterion(output, y)
        running_loss += loss.data[0]
        print('[validation] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.data[0]))

    seq_loss = running_loss/64
    return seq_loss

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

if __name__ == "__main__":
    main()