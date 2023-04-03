import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import torch.utils.data as data
from itertools import combinations
import json

print(f'PyTorch version {torch.__version__}')

class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filename_to_class = {}
        self.classname_to_filenames = {}
        self.classnames = set()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[116.022, 106.491, 95.719],
            #                      std=[75.824, 72.377, 74.867]),
        ])# resnet and vgg size 128, 128 inception size 299,299

        # Create a dictionary mapping classnames to the list of image filenames
        for filename in os.listdir(os.path.join(self.root_dir, "annotations")):
            if filename.endswith(".txt"):
                class_name = os.path.splitext(filename)[0]
                with open(os.path.join(self.root_dir, "annotations", filename)) as f:
                          image_numbers = f.readlines()
                          image_filenames = ["im{}.jpg".format(n.strip()) for n in image_numbers]
                          self.classname_to_filenames[class_name] = image_filenames
                          self.classnames.add(class_name)

        # Create a dictionary with multi labels
        for class_name in self.classnames: #os.listdir(os.path.join(self.root_dir, "annotations")):
            with open(os.path.join(self.root_dir, "annotations", class_name + '.txt'), "r") as f:
                images = f.readlines()
                images = [int(x.strip()) for x in images]
            for image in images:
                labels = np.zeros(len(self.classnames))
                image_filename = "im{}.jpg".format(image)
                # check if image is in a class and store the label of that class
                for i, class_name in enumerate(self.classnames):
                    if image_filename in self.classname_to_filenames[class_name]:
                        labels[i] = 1
                self.filename_to_class[image_filename] = labels
        print(f'show a sample labels {self.filename_to_class[list(self.filename_to_class.keys())[0]]}')

    def __len__(self):
        """The size of the dataset"""
        return len(self.filename_to_class)

    def __getitem__(self, index):
        """Get a specific image and label

        Read the corresponding image and convert it into a PyTorch
        Tensor

        """
        filename = list(self.filename_to_class.keys())[index]
        image_path = os.path.join(self.root_dir, "images", filename)
        image = Image.open(image_path).convert('RGB')
        label = self.filename_to_class[filename]

        # Apply transformations
        image = self.transform(image)

        # Create a dictionary containing the image and the label
        return {'images': image, 'labels': label}

class TestDataset(data.Dataset):
    """
    Create a dataset for testing where we dont have the ground truth
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)), # 128, 128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # transforms.Normalize(mean=[116.022, 106.491, 95.719],
            #                      std=[75.824, 72.377, 74.867]),])

        self.images = os.listdir(os.path.join(root_dir, "images"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image
        filename = self.images[index]
        image_path = os.path.join(self.root_dir, "images", filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {'images':image, 'filenames':filename}

def epoch_time(start_time, end_time):
    """Calcuate the time a training epoch took to train"""
    elapsed_time = end_time - start_time
    elapsped_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time = elapsed_mins ( 60))
    return elapsed_mins, elapsed_secs

def collate_fn(batch):
    """Customize the batching process of the dataloader in the multi-label
    setting.

    The collate function is an important component of the PyTorch
    DataLoader, as it allows you to customize how batches are formed
    from individual samples. In the case of multi-label
    classification, where the samples may have different numbers of
    labels, the default collate function may not work, which is why a
    custom collate function is needed to handle this case properly.

    """
    # Get maximum image size
    max_size = max([image['images'].shape[-1] for image in batch])

    # Pad images to the same size
    padded_images = []
    labels = []
    for image in batch:
        padded_image = torch.nn.functional.pad(image['images'], pad=(0, 0, max_size - image['images'].shape[-1], 0), mode='constant', value=0)
        padded_images.append(padded_image)
        labels.append(image['labels'])
    images = torch.stack(padded_images)
    labels = torch.tensor(np.array([item for item in labels]))
    return {'images': images, 'labels': labels}

def test_collate_fn(batch):
    images = torch.stack([sample['images'] for sample in batch])
    ids = [item['filenames'] for item in batch]
    return {'images': images, 'filenames': ids }

# def test_collate_fn(batch):
#     images = torch.stack([sample['images'] for sample in batch])
#     return {'images': images}


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
    lr_rate = 1e-7
    num_epochs = 30

    # Initialize model
    #model = models.resnet50(pretrained=True) # Resnet50
    #model = models.vgg16(pretrained=True) # VGG16
    model = models.inception_v3(pretrained=True)

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    # ResNet50
    # num_classes = len(class_names)
    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(in_features=2048, out_features=1024),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(p=0.3),
    #     torch.nn.Linear(in_features=1024, out_features=num_classes),
    #     torch.nn.Sigmoid())

    # VGG16
    # num_features = model.classifier[-1].in_features
    # num_classes = len(class_names)
    # model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

    # Inception_v3
    num_classes = len(class_names)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.4),
        torch.nn.Linear(1024, num_classes),
        torch.nn.Sigmoid())

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
            outputs =  outputs.logits # inception specific
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

# def test_model(test_loader, model):
#     """
#     Test the model with a test datset and return the predictions
#     """
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             images = batch['images'].to(device)
#             outputs = model(images)
#             predicted_labels = torch.round(torch.sigmoid(outputs))
#             print(predicted_labels)
#             predictions.extend(predicted_labels.cpu().numpy())

#     return predictions

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
    dataset = CustomDataset('./')
    test_dataset = TestDataset('./test_images')

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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=train_sampler, collate_fn=collate_fn)
    validationloader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=validation_sampler, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, sampler=test_sampler, collate_fn=test_collate_fn)

    model = train_and_val_model(trainloader, validationloader, classnames)
    predictions = test_model(testloader, model)
    #json_obj = json.dumps({"predictions": [pred.tolist() for pred in predictions]}) # when predictions is array
    json_obj = json.dumps(predictions)
    #with open('predictions_resnet50.json', 'w') as f:
    with open('predictions_inception_v3.json', 'w') as f:
        f.write(json_obj)
