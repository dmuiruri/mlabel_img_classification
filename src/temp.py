
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filename_to_class = {}
        self.classname_to_filenames = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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

