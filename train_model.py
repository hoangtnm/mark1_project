import os
import time
import copy
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MAX_SPEED = 8


# Hyper-parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500

# << The directory containing the cooked data from the previous step >>
COOKED_DATA_DIR = './cooked_data/'

# << The directory in which the model output will be placed >>
MODEL_OUTPUT_DIR = './models/'

train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')

# Use ROI of [78,144,27,227] for FOV 60 with Formula car
data_generator = DriveDataGenerator(rescale=1. / 255.,
                                    horizontal_flip=False,
                                    brighten_range=0.4)
train_generator = data_generator.flow(train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=BATCH_SIZE,
                                      zero_drop_percentage=0.95, roi=[78, 144, 27, 227])
eval_generator = data_generator.flow(eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=BATCH_SIZE,
                                     zero_drop_percentage=0.95, roi=[78, 144, 27, 227])


class AirSimDataset(Dataset):

    def __init__(self, data_dir, transforms=None):
        """
        Args:
            data_dir (string): Dataset directory.
            transforms (callable, optional): Optional transforms to be applied
                on a sample.
        Details:
            data_dir:
                airsim_rec.txt: AirSim log
                images: folder of images
        """
        self.data_dir = data_dir
        self.dataframe = pd.read_csv(os.path.join(
            self.data_dir, 'airsim_rec.txt'), sep='\t')
        self.transforms = transforms

    def __len__(self):
        # reads images from index 1 to n-1
        return self.dataframe.shape[0] - 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.dataframe.iloc[idx + 1]
        image = Image.open(os.path.join(
            self.data_dir, 'images', data['ImageFile']))
        image_rgb = image.convert('RGB')
        image_np = np.asarray(image_rgb)

        # Normalize steering: between 0 and 1
        norm_steering = [
            (float(self.dataframe.iloc[idx][['Steering']]) + 1) / 2.0]
        # norm_throttle = [float(self.dataframe.iloc[idx][['Throttle']])]
        # # Normalize speed: between 0 and 1
        # norm_speed = [
        #     float(self.dataframe.iloc[idx][['Speed']]) / MAX_SPEED]

        # previous_state = norm_steering + norm_throttle + norm_speed   # Append lists

        # compute average steering over 3 consecutive recorded images, this will serve as the label
        norm_steering0 = (
            float(self.dataframe.iloc[idx][['Steering']]) + 1) / 2.0
        norm_steering1 = (
            float(self.dataframe.iloc[idx+2][['Steering']]) + 1) / 2.0

        temp_sum_steering = norm_steering[0] + \
            norm_steering0 + norm_steering1
        average_steering = temp_sum_steering / 3.0

        if self.transforms:
            image_np = self.transforms(image_np)

        return image_np, torch.tensor(average_steering, dtype=torch.float32)


class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2))),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2))),
            ('relu3', nn.ReLU(True)),
            ('drop1', nn.Dropout())
        ]))

        self.block2 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1))),
            ('relu4', nn.ReLU(True)),
            ('conv5', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))),
            ('relu5', nn.ReLU(True))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(64 * 25 * 11, 100)),
            ('fc3', nn.Linear(100, 50)),
            ('fc4', nn.Linear(50, 10)),
            ('fc5', nn.Linear(10, 1)),
            ('out', nn.Sigmoid())
        ]))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_model(model, dataloaders, criterion, optimizer, device,
                output_dir, writer=None, scheduler=None, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.9

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    steering = model(inputs)
                    loss = criterion(steering, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_loss = loss.item()
                running_loss += batch_loss * inputs.size(0)
                if phase == 'train':
                    writer.add_scalar('training_loss', batch_loss,
                                      epoch * len(dataloaders[phase]) + i)

            # Used for StepLR scheduler
            if phase == 'train' and (scheduler is not None):
                scheduler.step()

            # epoch_loss = running_loss / total_batches
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # Used for ReduceLROnPlateau scheduler
            # if phase == 'val' and (scheduler is not None):
            #     scheduler.step(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {100*(1-epoch_loss)}%')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(
                    output_dir, 'checkpoint.pth'))
            if phase == 'val':
                writer.add_scalar('validation_loss', batch_loss,
                                  epoch * len(dataloaders[phase]) + i)

    time_elapsed = time.time() - since
    print(f'Training completed in \
          {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # TensorBoard setup
    writer = SummaryWriter('runs')

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if train_on_gpu else 'cpu')
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU ...')
    else:
        print('CUDA is available! Training on GPU ...')

    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = os.path.join('raw_data', '2019_09_24')
    image_datasets = {x: AirSimDataset(os.path.join(data_dir, x),
                                       transforms=data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # initializes a neural network for training
    model = NeuralNet()
    # dataiter = iter(dataloaders['train'])
    # images, labels = dataiter.next()
    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('images', grid, 0)
    # writer.add_graph(model, images)
    model.to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=3, min_lr=0, verbose=True)

    model = train_model(model, dataloaders, criterion, optimizer, device,
                        MODEL_OUTPUT_DIR, writer, num_epochs=NUM_EPOCHS)
    writer.close()
