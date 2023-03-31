import os
import sys
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
    
class CustomDataset(data.Dataset):
    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_filenames = []
        self.annotations = []
        self.filename_to_annotation = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[116.022, 106.491, 95.719], std=[75.824, 72.377, 74.867]),
            transforms.ToTensor()
            ])

        # Create a list of image filenames
        for filename in os.listdir(self.annotations_dir):
            if filename.endswith(".txt"):
                self.image_filenames.append(filename)
        # Read the annoations files and store the annotations in a list
        for filename in os.listdir(self.annotations_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.annotations_dir, filename), 'r') as f:
                    annotations = f.readlines()
                    annotations = [int(line.strip()) for line in annotations]
                    image_filename = "{}".format(filename.split(".")[0])
                    self.filename_to_annotation[image_filename] = annotations

    def __len__(self):
        """The size of the dataset"""
        return len(os.listdir(self.images_dir))

    def __getitem__(self, index):
        """Get a specific image

        Read the corresponding image and convert it into a PyTorch
        Tensor

        """
        filename = self.image_filenames[index + 1]
        filename = os.path.splitext(filename)[0]
        image_path = os.path.join(self.images_dir, filename + '.jpg')
        image = Image.open(image_path).convert('RGB')
        annotation = self.filename_to_annotation[filename]

        # Apply transformations also transforms to tensor
        image = self.transform(image)

        # Create a dictionary containing the image and the annotation
        sample = {'image': image, 'annotation': annotation}
        print(f'index {index} retrived')
        return sample


if __name__ == '__main__':
    # res = data_mean_std('../images')
    # # rgb_mean, rgb_std, gray_mean, gray_std, color_images, gray_images
    # print(f'rgb mean {res[0]}, rgd_std {res[1]}')
    # print(f'gray mean {res[2]}, gray std {res[3]}')
    # print(f'Color images {res[4]}, gray images {res[5]}')

    # create a dataset object
    dataset = CustomDataset('../images', '../annotations')

    # create a dataloader object
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for i, batch in enumerate(dataloader):
        batch.shape()
        break
