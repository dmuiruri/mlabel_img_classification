import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

"""
This module contains custom custom dataset classes and related functions
"""

class CustomDataset(data.Dataset):
    """
    A custom dataset class create a training and validation dataloaders

    # Image sizes to be changed depending with model resnet and vgg size 128x128 inception 299x299
    """
    def __init__(self, root_dir, size=128):
        self.root_dir = root_dir
        self.filename_to_class = {}
        self.classname_to_filenames = {}
        self.classnames = set()
        self.transform = transforms.Compose([
            transforms.Resize((size, size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[116.022, 106.491, 95.719],std=[75.824, 72.377, 74.867])]) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] 

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
    def __init__(self, root_dir, size=128):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[116.022, 106.491, 95.719],std=[75.824, 72.377, 74.867])]) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

if __name__ == '__main__':
    # Create a dataset object
    dataset = CustomDataset('./')
    test_dataset = TestDataset('./test_images')

    # Create a test for the CustomDataset class
    print(f'Number of images: {len(dataset)}')
    print(f'Number of classes: {len(dataset.classnames)}')
    print(f'Class names: {dataset.classnames}')
    print(f'Image filenames: {list(dataset.filename_to_class.keys())[:5]}')
    print(f'Image labels: {list(dataset.filename_to_class.values())[:5]}')
