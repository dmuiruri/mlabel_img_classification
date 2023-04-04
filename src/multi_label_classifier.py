import os
import sys
import torch

import torchvision.models as models
import numpy as np
from PIL import Image
import torch.utils.data as data
from itertools import combinations
import json
import imagedata

print(f'PyTorch version {torch.__version__}')


def epoch_time(start_time, end_time):
    """Calcuate the time a training epoch took to train"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def accuracy(outputs, labels, threshold=0.5):
    """Calculate accuracy"""
    predicted = outputs > threshold
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    acc = correct / total
    return acc

def train_and_val_model(train_loader, val_loader, class_names):
    """
    Train a multi-label model 
    """
    lr_rate = 1e-4
    num_epochs = 10

    # Initialize models
    model = models.resnet50(pretrained=True) # Resnet50
    #model = models.vgg16(pretrained=True) # VGG16
    #model = models.inception_v3(pretrained=True)

    # Test effect of unfreezing more layers (3 and 4)
    # for name, param in model.named_parameters():
    #     if "layer4" in name or "layer3" in name:
    #         param.requires_grad = True

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze some layers
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name:
            param.requires_grad = True
    # Replace the final fully connected layer
    # ResNet50
    num_classes = len(class_names)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=2048, out_features=1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features=1024, out_features=num_classes))

    # VGG16
    # num_features = model.classifier[-1].in_features
    # num_classes = len(class_names)
    # model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

    # Inception_v3
    # num_classes = len(class_names)
    # num_features = model.fc.in_features
    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(num_features, 1024),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(p=0.4),
    #     torch.nn.Linear(1024, num_classes))

    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    # Train model
    model.to(device)
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0
        model.train()
        for i, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images)
            # outputs =  outputs.logits # ToDo: inception specific line
            # calculate loss
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # update model parameters
            optimizer.step()

            # Calculate accuracy
            acc = accuracy(outputs, labels)
            train_acc += acc
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(images)
                acc = accuracy(outputs, labels)
                loss = criterion(outputs, labels)
                val_acc += acc
                val_loss += loss.item()

        # Display results
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train_acc: {:.4f}, Val acc: {:.4f}'
              .format(epoch+1, num_epochs,
                      train_loss/len(train_loader), val_loss/len(val_loader),
                      train_acc/len(train_loader), val_acc/len(val_loader)))
    return model

def test_model(test_loader, model):
    """
    Test the model with a test datset
    """
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    predictions = {}
    threshold = 0.4
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['images'].to(device)
            filenames = batch['filenames']
            outputs = model(images)
            # predicted_labels = torch.round(torch.sigmoid(outputs))
            predicted_labels = torch.sigmoid(outputs)
            # Apply thresholding to convert probabilities to binary predictions
            batch_predictions = (predicted_labels >= threshold).int()
            predicted_labels = batch_predictions.detach().cpu().numpy()
            for i in range(len(filenames)):
                predictions[filenames[i]] = predicted_labels[i].tolist()
    return predictions

if __name__ == '__main__':
    """Train and validate model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    validation_split = 0.2

    # Create a dataset object
    dataset = imagedata.CustomDataset('./')
    test_dataset = imagedata.TestDataset('./test_images')

    # Split the data into train and validation sets
    dataset_size = len(dataset)
    classnames = dataset.classnames
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # May not want to shuffle due to class imbalance
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]

    # Create training and validation samplers
    train_sampler = data.SubsetRandomSampler(train_indices)
    validation_sampler = data.SubsetRandomSampler(validation_indices)
    test_sampler = data.SubsetRandomSampler(range(len(test_dataset)))
    print(f'Length of train set {len(train_sampler)} and validation set {len(validation_sampler)}')

    # Create the dataloaders
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=train_sampler, collate_fn=imagedata.collate_fn)
    validationloader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=validation_sampler, collate_fn=imagedata.collate_fn)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, sampler=test_sampler, collate_fn=imagedata.test_collate_fn)

    model = train_and_val_model(trainloader, validationloader, classnames)
    predictions = test_model(testloader, model)
    json_obj = json.dumps(predictions)
    with open('predictions_resnet50.json', 'w') as f:
        f.write(json_obj)
