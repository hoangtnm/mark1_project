import os
import time
import json
import h5py
import math
import copy
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers.convolutional import Convolution2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from tensorflow.keras.layers.core import Activation
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Generator import DriveDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MAX_SPEED = 8


# Hyper-parameters
batch_size = 32
learning_rate = 0.0001
number_of_epochs = 500

# Activation functions
activation = 'relu'
out_activation = 'sigmoid'

# Stop training if in the last 20 epochs, there was no change of the best recorded validation loss
training_patience = 20

# << The directory containing the cooked data from the previous step >>
COOKED_DATA_DIR = './cooked_data/'

# << The directory in which the model output will be placed >>
MODEL_OUTPUT_DIR = './models/'

train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')

num_train_examples = train_dataset['image'].shape[0]
num_eval_examples = eval_dataset['image'].shape[0]

# Use ROI of [78,144,27,227] for FOV 60 with Formula car
data_generator = DriveDataGenerator(rescale=1. / 255.,
                                    horizontal_flip=False,
                                    brighten_range=0.4)
train_generator = data_generator.flow(train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=batch_size,
                                      zero_drop_percentage=0.95, roi=[78, 144, 27, 227])
eval_generator = data_generator.flow(eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=batch_size,
                                     zero_drop_percentage=0.95, roi=[78, 144, 27, 227])

[sample_batch_train_data, sample_batch_test_data] = next(train_generator)

image_input_shape = sample_batch_train_data[0].shape[1:]

pic_input = Input(shape=image_input_shape)

# Network definition
img_stack = Conv2D(24, (5, 5), name="conv1", strides=(2, 2), padding="valid", activation=activation,
                   kernel_initializer="he_normal")(pic_input)
img_stack = Conv2D(36, (5, 5), name="conv2", strides=(2, 2), padding="valid", activation=activation,
                   kernel_initializer="he_normal")(img_stack)
img_stack = Conv2D(48, (5, 5), name="conv3", strides=(2, 2), padding="valid", activation=activation,
                   kernel_initializer="he_normal")(img_stack)

img_stack = Dropout(0.5)(img_stack)

img_stack = Conv2D(64, (3, 3), name="conv4", strides=(1, 1), padding="valid", activation=activation,
                   kernel_initializer="he_normal")(img_stack)
img_stack = Conv2D(64, (3, 3), name="conv5", strides=(1, 1), padding="valid", activation=activation,
                   kernel_initializer="he_normal")(img_stack)

img_stack = Flatten(name='flatten')(img_stack)

img_stack = Dense(100, name="fc2", activation=activation,
                  kernel_initializer="he_normal")(img_stack)
img_stack = Dense(50, name="fc3", activation=activation,
                  kernel_initializer="he_normal")(img_stack)
img_stack = Dense(10, name="fc4", activation=activation,
                  kernel_initializer="he_normal")(img_stack)
img_stack = Dense(1, name="output", activation=out_activation,
                  kernel_initializer="he_normal")(img_stack)

adam = Adam(lr=learning_rate, beta_1=0.9,
            beta_2=0.999, epsilon=1e-08, decay=0.0)

model = Model(inputs=[pic_input], outputs=img_stack)
model.compile(optimizer=adam, loss='mse')

model.summary()

plateau_callback = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=learning_rate, verbose=1)
csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))
checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'fresh_models',
                                   '{0}_model.{1}-{2}.h5'.format('model', '{epoch:02d}', '{val_loss:.7f}'))
checkpoint_callback = ModelCheckpoint(
    checkpoint_filepath, save_best_only=True, verbose=1)
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=training_patience, verbose=1)
callbacks = [plateau_callback, csv_callback, checkpoint_callback,
             early_stopping_callback, TQDMNotebookCallback()]

history = model.fit_generator(train_generator, steps_per_epoch=num_train_examples // batch_size,
                              epochs=number_of_epochs, callbacks=callbacks,
                              validation_data=eval_generator, validation_steps=num_eval_examples // batch_size,
                              verbose=2)


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
        norm_throttle = [float(self.dataframe.iloc[idx][['Throttle']])]
        # Normalize speed: between 0 and 1
        norm_speed = [
            float(self.dataframe.iloc[idx][['Speed']]) / MAX_SPEED]

        previous_state = norm_steering + norm_throttle + norm_speed   # Append lists

        # compute average steering over 3 consecutive recorded images, this will serve as the label
        norm_steering0 = (
            float(self.dataframe.iloc[idx][['Steering']]) + 1) / 2.0
        norm_steering1 = (
            float(self.dataframe.iloc[idx+2][['Steering']]) + 1) / 2.0

        temp_sum_steering = norm_steering[0] + \
            norm_steering0 + norm_steering1
        average_steering = temp_sum_steering / 3.0

        current_label = [average_steering]

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


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, output_dir, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.9

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                running_loss += loss.item() * inputs.size(0)

            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / (inputs.size(0) * (i+1))

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, output_dir)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
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

    # datasets
    trainset = AirSimDataset(os.path.join(
        'raw_data', '2019_09_24'), transforms=data_transforms['train'])

    # dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, batch_size=32,
                                             shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(trainset, batch_size=32,
                                           shuffle=False, num_workers=4)
    }

    # initializes a neural network for training
    model = NeuralNet()
    model.to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, dataloaders, criterion, optimizer,
                        exp_lr_scheduler, device, MODEL_OUTPUT_DIR, num_epochs=500)
