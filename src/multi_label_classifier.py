import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.utils.data as data

def data_mean_std(image_dir):
    """
    Generate the mean and std of the dataset
    """
    image_dir = os.path.realpath(image_dir)

    # Empty arrays to store pixel values
    r_channel = []
    g_channel = []
    b_channel = []

    # Loop over all images in the dir
    for filename in os.listdir(image_dir):
        # Load the image and convert it to numpy array
        img = np.array(Image.open(os.path.join(image_dir, filename)))

        # Append pixel values to respective channel array
        r_channel.append(img[:,:,0])
        g_channel.append(img[:,:,1])
        b_channel.append(img[:,:,2])
    # Calculate mean and standard deviation for each channel
    mean = [np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)]
    std = [np.std(r_channel), np.std(g_channel), np.std(b_channel)]
    return mean, std
    
class CustomDataset(data.Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = , std =[])
            ])
        self.annotations = []

        # Read the annoations files and store the annotations in a list
        for filename in os.listdir(self.annotations_dir):
            with open(os.path.join(self.annoations_dir, filename), 'r') as f:
                annotation = [int(line.strip()) for line in f]
                self.annotations.append(annotation)

    def __len__(self):
        """The size of the dataset"""
        return len(os.listdir(self.images_dir))

    def __getitem__(self, index):
        """Get a specific image

        Read the corresponding image and convert it into a PyTorch
        Tensor

        """
        image_name = 'im{:05d}.jpg'.format(index + i)
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # Read the corresponding annotation and return it along with
        # the image tensor
        annoation = torch.tensor(self.annoations[index])
        return image, annotation

    if __name__ == '__main__':
        mean, std = data_mean_std()
        print(mean, std)
