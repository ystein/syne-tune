"""
Example script that shows how to fine-tune models with jumpstart
"""
import requests
import copy
import logging
import os
import tarfile
import time
from filelock import SoftFileLock, Timeout

from argparse import ArgumentParser

import torch
from torchvision import datasets
from torchvision import transforms
from torchvision import models

from syne_tune.report import Reporter


# TODO: use dataset specific normalizing statistics
VALIDATION_SPLIT = 0.20
RANDOM_RESIZED_CROP = 224

NUM_WORKERS = 0
TRAIN = "train"
VAL = "val"
CLASS_LABEL_TO_PREDICTION_INDEX_JSON = "class_label_to_prediction_index.json"

FLOWER_NORM_STATS = dict()
FLOWER_NORM_STATS['mean'] = [0.485, 0.456, 0.406]
FLOWER_NORM_STATS['std'] = [0.229, 0.224, 0.225]

CALTECH101_NORM_STATS = dict()
CALTECH101_NORM_STATS['mean'] = [0.5380, 0.5094, 0.4798]
CALTECH101_NORM_STATS['std'] = [0.2331, 0.2293, 0.2297]

CALTECH256_NORM_STATS = dict()
CALTECH256_NORM_STATS['mean'] = [0.5392, 0.5125, 0.4809]
CALTECH256_NORM_STATS['std'] = [0.2265, 0.2249, 0.2253]

NORMALIZATION_STATS = {'flower': FLOWER_NORM_STATS, 'caltech101': CALTECH101_NORM_STATS, 'caltech256': CALTECH256_NORM_STATS}

report = Reporter()


def download_dataset(dataset_name, dataset_dir):

    if dataset_name == 'flower':

        dst = os.path.join(dataset_dir, 'flower_dataset', 'flower_photos')
        if os.path.isdir(dst):
            return dst

        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        r = requests.get(dataset_url)
        with open(os.path.join(dataset_dir, 'flower_dataset.tgz'), 'wb') as f:
            f.write(r.content)
        with tarfile.open(os.path.join(dataset_dir, 'flower_dataset.tgz')) as saved_dataset:
            saved_dataset.extractall(os.path.join(dataset_dir, 'flower_dataset/'))

        return dst

    if dataset_name == 'caltech256':
        cal = datasets.Caltech256(dataset_dir, download=True)
        return cal.root + '/256_ObjectCategories'

    elif dataset_name == 'caltech101':

        cal = datasets.Caltech101(dataset_dir, download=True)
        return cal.root + '/101_ObjectCategories'


def get_model(model_id):
    if model_id == 'resnet18':
        return models.resnet18(pretrained=True)
    elif model_id == 'mobilenet':
        return models.mobilenet_v2(pretrained=True)


def change_pred_layer_size(model, model_id, _size):
    if any(model in model_id for model in ("resnet", "resnext", "shufflenet", "googlenet")):
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, _size)
        return
    elif any(model in model_id for model in ("mobilenet", "alexnet", "vgg")):
        pos_linear_layer = len(model.classifier) - 1
        num_features = model.classifier[pos_linear_layer].in_features
        model.classifier._modules[str(pos_linear_layer)] = torch.nn.Linear(num_features, _size)
        return
    elif "densenet" in model_id:
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, _size)
        return
    elif "squeezenet" in model_id:
        model.classifier._modules["1"] = torch.nn.Conv2d(512, _size, kernel_size=(1, 1))
        return
    else:
        raise NotImplementedError


def _prepare_dataloader(data_dir, batch_size, dataset_name):
    # Data augmentation and normalization for training
    # Just normalization for validation

    normalizing_mean = NORMALIZATION_STATS[dataset_name]['mean']
    normalizing_std = NORMALIZATION_STATS[dataset_name]['std']

    data_transforms = {
        TRAIN: transforms.Compose(
            [
                transforms.RandomResizedCrop(RANDOM_RESIZED_CROP),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(normalizing_mean, normalizing_std),
            ]
        ),
        VAL: transforms.Compose(
            [
                transforms.Resize(RANDOM_RESIZED_CROP),
                transforms.CenterCrop(RANDOM_RESIZED_CROP),
                transforms.ToTensor(),
                transforms.Normalize(normalizing_mean, normalizing_std),
            ]
        ),
    }
    full_dataset = datasets.ImageFolder(data_dir)
    class_to_idx = full_dataset.class_to_idx
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = data_transforms[VAL]
    train_dataset.dataset.transform = data_transforms[TRAIN]
    val_dataset.dataset.transforms = data_transforms[VAL]
    train_dataset.dataset.transforms = data_transforms[TRAIN]

    dataloaders = {
        TRAIN: torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
        ),
        VAL: torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True
        ),
    }
    dataset_sizes = {TRAIN: len(train_dataset), VAL: len(val_dataset)}

    return dataloaders, dataset_sizes, class_to_idx


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs, device, resume_from, st_checkpoint_dir):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(resume_from, num_epochs):
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == TRAIN:
                if scheduler is not None:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == VAL:

                print('checkpoint')

                os.makedirs(st_checkpoint_dir, exist_ok=True)
                local_filename = os.path.join(st_checkpoint_dir, 'checkpoint')
                data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
                torch.save(data, local_filename)

                report(
                    epoch=epoch + 1,
                    objective=epoch_acc.item(),
                    time_step=time.time(),
                    elapsed_time=time.time() - since)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    logging.info("Best val Acc: {:4f}".format(best_acc))

    return best_model_wts


def objective(config):

    model_id = config['model_id']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    weight_decay = config['weight_decay']

    dataset_dir = config['dataset_dir']
    dataset_name = config['dataset_name']

    os.makedirs(dataset_dir, exist_ok=True)
    # find GPU device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    lock_path = os.path.join(dataset_dir, 'lock_dataset')
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            dst = download_dataset(dataset_name, dataset_dir)
    except Timeout:
        print(
            "WARNING: Could not obtain lock for dataset files. Trying anyway...",
            flush=True)
        dst = download_dataset(dataset_name, dataset_dir)

    dataloaders, dataset_sizes, class_to_idx = _prepare_dataloader(dst,
                                                                   batch_size=batch_size,
                                                                   dataset_name=dataset_name)

    logging.info(f"dataset sizes: {dataset_sizes}")
    logging.info(f"prediction class indices mapping to input training data labels: {class_to_idx}")
    num_labels = len(class_to_idx.keys())

    # load model
    lock_path = os.path.join(dataset_dir, 'lock_model')
    lock = SoftFileLock(lock_path)
    try:
        with lock.acquire(timeout=120, poll_intervall=1):
            model = get_model(model_id=model_id)
    except Timeout:
        print(
            "WARNING: Could not obtain lock for model files. Trying anyway...",
            flush=True)
        model = get_model(model_id=model_id)

    for param in model.parameters():
        param.requires_grad = False

    change_pred_layer_size(model, model_id, num_labels)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0]).to(device)

    model.eval()

    # init optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=adam_learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True,
                                weight_decay=weight_decay, momentum=momentum)

    if config['st_checkpoint_dir'] is not None and os.path.isdir(config['st_checkpoint_dir']):
        print('load checkpoint', flush=True)
        local_filename = os.path.join(config['st_checkpoint_dir'], 'checkpoint')
        checkpoint = torch.load(local_filename)
        resume_from = int(checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        resume_from = 0

    if config['scheduler_type'] == 'none' or config['scheduler_type'] == 'const':
        scheduler = None
    elif config['scheduler_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif config['scheduler_type'] == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

    # start training
    best_model_wts = train_model(
        dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=epochs,
        device=device, resume_from=resume_from, st_checkpoint_dir=config['st_checkpoint_dir']
    )


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--scheduler_type", type=str, default='none')
    parser.add_argument('--st_checkpoint_dir', type=str)
    parser.add_argument("--trial_id", type=int, default=5)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
