import os
import sys
import torch
import torchvision.transforms as transforms
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
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[116.022, 106.491, 95.719],
                                 std=[75.824, 72.377, 74.867]),
        ])

        # Create a dictionary mapping image filename to their class labels
        # for class_name in os.listdir(os.path.join(self.root_dir, "annotations")):
        #     if class_name.endswith(".txt"):
        #         with open(os.path.join(self.root_dir, "annotations", class_name), "r") as f:
        #             images = f.readlines()
        #             images = [int(x.strip()) for x in images]
        #         for image in images:
        #             image_filename = "im{}.jpg".format(image)
        #             self.filename_to_class[image_filename] = class_name.split(".")[0]

        # Create a dictionary mapping classnames to the list of image filenames
        for filename in os.listdir(os.path.join(self.root_dir, "annotations")):
            if filename.endswith(".txt"):
                class_name = os.path.splitext(filename)[0]
                with open(os.path.join(self.root_dir, "annotations", filename)) as f:
                          image_numbers = f.readlines()
                          image_filenames = ["im{}.jpg".format(n.strip()) for n in image_numbers]
                          self.classname_to_filenames[class_name] = image_filenames

        # Create a dictionary mapping image filename to their class labels

        # Explored this approach but realized there are some unlabeled
        # images in the dataset so its easier to pick images from the
        # annotation files.
        
        # for image_file in os.listdir(os.path.join(self.root_dir, "images")):
        #     labels = []
        #     for key in self.classname_to_filenames.keys():
        #         if image_file in self.classname_to_filenames[key]:
        #             labels.append(key)
        #     print(f'{(image_file, labels)}')
        #     self.filename_to_class[image_file] = labels

        # Create a dictionary with multi labels
        for class_name in os.listdir(os.path.join(self.root_dir, "annotations")):
            if class_name.endswith(".txt"):
                with open(os.path.join(self.root_dir, "annotations", class_name), "r") as f:
                    images = f.readlines()
                    images = [int(x.strip()) for x in images]
                for image in images:
                    labels = []
                    image_filename = "im{}.jpg".format(image)
                    # check if image is in a class and store the label of that class
                    for key in self.classname_to_filenames:
                        if image_filename in self.classname_to_filenames[key]:
                            labels.append(key)
                    self.filename_to_class[image_filename] = labels #class_name.split(".")[0]
        print(f'show a sample labels {self.filename_to_class[list(self.filename_to_class.keys())[0]]}')

        # Check that all images have the same size
        # Result: All images have same size
        # image_sizes = set()
        # for filename in self.filename_to_class:
        #     image_path = os.path.join(self.root_dir, "images", filename)
        #     with Image.open(image_path) as img:
        #         sys.stdout.write('.')
        #         sys.stdout.flush()
        #         image_sizes.add(img.size)
        #         if len(image_sizes) > 1:
        #             print(image_sizes)
        #             raise ValueError("Images have different sizes")

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

def train_model(loader):
    """
    Train a multi-label model
    """
    # Initialize model

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train model
    model.train()
    for i, batch in enumerate(loader):
        pass

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

    # Return batch
    return {'images': torch.stack(padded_images), 'labels': labels}

if __name__ == '__main__':
    # res = data_mean_std('../images')
    # print(f'rgb mean {res[0]}, rgd_std {res[1]}')
    # print(f'gray mean {res[2]}, gray std {res[3]}')
    # print(f'Color images {res[4]}, gray images {res[5]}')

    # check_multilabels('./')
    # check_duplicate_images('.')

    validation_split = 0.2

    # Create a dataset object
    dataset = CustomDataset('./')
    print(f'Length of dataset {len(dataset)}')

    # Split the data into train and test sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # way or may not want to shuffle due to class imbalance
    # np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]

    # Create training and validation samplers
    train_sampler = data.SubsetRandomSampler(train_indices)
    validation_sampler = data.SubsetRandomSampler(validation_indices)
    print(f'Length of train set {len(train_sampler)} and validation set {len(validation_sampler)}')

    # Create the dataloaders
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=validation_sampler, collate_fn=collate_fn)

    for i, batch in enumerate(trainloader):
        pass
