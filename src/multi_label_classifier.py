import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import torch.utils.data as data
from itertools import combinations

print(f'PyTorch version {torch.__version__}')

def data_mean_std(image_dir):
    """A helper function to determine the mean and std of the dataset

    """
    image_dir = os.path.realpath(image_dir)

    # Empty arrays to store pixel values
    r_channel = []
    g_channel = []
    b_channel = []
    gray_img = []
    color_images = 0
    gray_images = 0
    print(f'>> Generating mean and std of the data at {image_dir}')
    # Loop over all images in the dir
    for filename in os.listdir(image_dir):
        # sys.stdout.write('.')
        # sys.stdout.flush()
        # Load the image and convert it to numpy array
        img = np.array(Image.open(os.path.join(image_dir, filename)))

        # Append pixel values to respective channel array
        if len(img.shape) == 3:
            color_images += 1
            r_channel.append(img[:,:,0])
            g_channel.append(img[:,:,1])
            b_channel.append(img[:,:,2])
        else:
            # Determine proportion of non RGB images
            gray_images += 1
            gray_img.append(img)
    # Calculate mean and standard deviation for each channel
    rgb_mean = [np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)]
    rgb_std = [np.std(r_channel), np.std(g_channel), np.std(b_channel)]
    gray_mean = np.mean(gray_img)
    gray_std = np.std(gray_img)
    print(f'Color images: {color_images}, gray scale images {gray_images}')
    return rgb_mean, rgb_std, gray_mean, gray_std, color_images, gray_images

def check_multilabels(root_dir):
    """
    Helper function to test for the presence of multilabels
    """
    item_classes = {}
    num_of_classes = len(os.listdir(os.path.join(root_dir, "annotations")))
    print(f'Number of classes: {num_of_classes}')
    for class_name in os.listdir(os.path.join(root_dir, "annotations")):
        if class_name.endswith('.txt'):
            label = class_name.split('.')[0]
            with open(os.path.join(root_dir, "annotations", class_name)) as f:
                      images = [int(line.strip()) for line in f.readlines()]
                      item_classes[label]=images
    labels = list(item_classes.keys())
    pairs = combinations(labels, 2)
    for pair in pairs:
        # images with multi-labels in these classes
        num_multi_label = len(set(item_classes[pair[0]]) &
                              set(item_classes[pair[1]]))
        print(f'{num_multi_label} images have labels '
              f'{pair[0]} and {pair[1]}')
    return

def check_duplicate_images(root_dir):
    """
    Check for duplicates of images
    """
    image_files = os.listdir(os.path.join(root_dir, "images"))
    num_of_images = len(image_files)
    num_image_non_duplicates = len(set(image_files))
    if num_of_images == num_image_non_duplicates:
        print(f"No duplicates {num_of_images} / {num_image_non_duplicates}")
    else:
        print(f'{num_of_images - num_image_non_duplicates} {num_of_images} / {num_image_non_duplicates}')

def transform_data():
    """
    Perform transformations of the data
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[116.022, 106.491, 95.719],
                             std=[75.824, 72.377, 74.867]),
    ])
    return transform

class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filename_to_class = {}
        self.classname_to_filenames = {}
        self.classnames = set()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[116.022, 106.491, 95.719],
                                 std=[75.824, 72.377, 74.867]),
        ])

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
        for class_name in os.listdir(os.path.join(self.root_dir, "annotations")):
            if class_name.endswith(".txt"):
                with open(os.path.join(self.root_dir, "annotations", class_name), "r") as f:
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
        image = transforms.Resize((224, 224))(image)
        image = self.transform(image)

        # Create a dictionary containing the image and the label
        return {'images': image, 'labels': label}

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
    #    return {'images': torch.stack(padded_images), 'labels': labels}

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
    lr_rate = 1e-6
    num_epochs = 10

    # Initialize model
    #model = models.resnet50(pretrained=True) # Resnet50
    model = models.vgg16(pretrained=True)

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    # ResNet50
    # num_classes = len(class_names)
    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(in_features=2048, out_features=1024),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(p=0.2),
    #     torch.nn.Linear(in_features=1024, out_features=num_classes),
    #     torch.nn.Sigmoid())

    # VGG16
    num_features = model.classifier[-1].in_features
    num_classes = len(class_names)
    model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

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
    test_loss = 0
    test_acc = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad()
    for batch_idx, batch in enumerate(test_loader):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        ouputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(ouputs, labels)
        test_loss += loss.item()
        test_acc += acc
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'
          .format(test_loss/len(test_loader),
                  test_acc/len(test_loader)))
    return

if __name__ == '__main__':
    """Train and validate model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    validation_split = 0.2

    # Create a dataset object
    dataset = CustomDataset('./')
    test_dataset = CustomDataset('./test_images')

    # Split the data into train and validation sets
    dataset_size = len(dataset)
    classnames = dataset.classnames
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # May not want to shuffle due to class imbalance
    # np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]

    # Create training and validation samplers
    train_sampler = data.SubsetRandomSampler(train_indices)
    validation_sampler = data.SubsetRandomSampler(validation_indices)
    test_sampler = data.SubsetRandomSampler(range(len(test_dataset)))
    print(f'Length of train set {len(train_sampler)} and validation set {len(validation_sampler)}')

    # Create the dataloaders
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
    validationloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=validation_sampler, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, sampler=test_sampler, collate_fn=collate_fn)

    model = train_and_val_model(trainloader, validationloader, classnames)
    test_model(testloader, model, criterion)
