import os

# TODO: Fix this file paths may not work after moving from model file

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

def check_test_data_labels(root_dir):
    """
    Check to see if data labels are suitable for the test data
    """
    classname_to_filenames = {}
    image_names = []
    # Create a dictionary of class to images
    for filename in os.listdir(os.path.join(root_dir, "annotations")):
        if filename.endswith(".txt"):
            class_name = os.path.splitext(filename)[0]
            with open(os.path.join(root_dir, "annotations", filename)) as f:
                image_numbers = f.readlines()
                image_filenames = ["im{}.jpg".format(n.strip()) for n in image_numbers]
                classname_to_filenames[class_name] = image_filenames

    # Get all image names into one list
    for key in classname_to_filenames:
        image_names += classname_to_filenames[key]

    # list all images
    image_files = os.listdir(os.path.join(root_dir, "images"))
    missing_annotations = set(image_files) - set(image_names)
    return len(missing_annotations)

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

if __name__ == '__main__':
    # res = data_mean_std('../images')
    # print(f'rgb mean {res[0]}, rgd_std {res[1]}')
    # print(f'gray mean {res[2]}, gray std {res[3]}')
    # print(f'Color images {res[4]}, gray images {res[5]}')

    # check_multilabels('./')
    # check_duplicate_images('.')
    print(check_test_data_labels('./test_images'))
    
