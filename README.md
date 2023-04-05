# mlabel_img_classification

This project is a multi-label image classification in computer
vision. One image can have multiple identifiable objects and the task
is to train a model to identify these objects.

The project was implemented using the PyTorch framework in particular
version *PyTorch version 2.0.0+cu117*, a more comprehensive
requirements.txt can be found in the repo.

## Data

The dataset used contains 20000 images and their annotations which are
stored in *images* and *annotations* folder respectively. Annotations
are contained in text files named by *<classname>.txt*. The annotations
folder contains 14 files which implies that an image can have upto 14
labels.

An example of image files in the image directory
```
ls images | head
im10000.jpg
im10001.jpg
im10002.jpg
...
```

An example of annotation files
```
baby.txt
bird.txt
car.txt
...
```
An example of contents in an annotation file
```
$cat annotations/car.txt | head
18077
13634
16466
...
```

### Data processing

We need to create a data class that will be used to create dataloaders
for training an validation purposes. For this purpose we create a
custom dataset class by inheriting the data.Dataset object in
torch. Typically, we need to implement(overide) the __init__, __len__
and __getitem__ functions of the parent class.

Because of the multiple labels per image, we need to create a
multi-hot encoded label for an image. The resulting data

## Methodology

Transfer learning was used as the training approach, this would help
avoid a cold start and improve training efficiency. Three models were
tested: ResNet60, VGG16 and Inception_v3.

First attempt based on only training the fully connected layers: Model
tends to overfitt as we note that the validation loss is lower than
the training loss and the validation accuracy is higher tnan the
training accuracy.

Second approach, we allow for more layers to update their parameters
and this improves the results as the training loss is now lower than
the validation loss and the accuracy of the training is higher than
the accuracy of the validation set, this is atleast the expected
behaviour.

## Results
Results from training process 
Training model: Resnet50
Epoch [1/10], Train Loss: 0.2582, Val Loss: 0.1551, Train_acc: 0.9056, Val acc: 0.9344
Epoch [2/10], Train Loss: 0.1457, Val Loss: 0.1239, Train_acc: 0.9383, Val acc: 0.9485
Epoch [3/10], Train Loss: 0.1139, Val Loss: 0.1139, Train_acc: 0.9530, Val acc: 0.9532
Epoch [4/10], Train Loss: 0.0905, Val Loss: 0.1122, Train_acc: 0.9642, Val acc: 0.9549
Epoch [5/10], Train Loss: 0.0718, Val Loss: 0.1148, Train_acc: 0.9724, Val acc: 0.9544
Epoch [6/10], Train Loss: 0.0599, Val Loss: 0.1195, Train_acc: 0.9776, Val acc: 0.9530
Epoch [7/10], Train Loss: 0.0503, Val Loss: 0.1224, Train_acc: 0.9811, Val acc: 0.9548
Epoch [8/10], Train Loss: 0.0430, Val Loss: 0.1241, Train_acc: 0.9842, Val acc: 0.9546
Epoch [9/10], Train Loss: 0.0364, Val Loss: 0.1270, Train_acc: 0.9867, Val acc: 0.9548
Epoch [10/10], Train Loss: 0.0303, Val Loss: 0.1347, Train_acc: 0.9889, Val acc: 0.9552
Training model: VGG16
Epoch [1/10], Train Loss: 0.4531, Val Loss: 0.3498, Train_acc: 0.8766, Val acc: 0.8889
Epoch [2/10], Train Loss: 0.2929, Val Loss: 0.2796, Train_acc: 0.8913, Val acc: 0.8929
Epoch [3/10], Train Loss: 0.2508, Val Loss: 0.2484, Train_acc: 0.8968, Val acc: 0.8988
Epoch [4/10], Train Loss: 0.2293, Val Loss: 0.2299, Train_acc: 0.9019, Val acc: 0.9048
Epoch [5/10], Train Loss: 0.2150, Val Loss: 0.2170, Train_acc: 0.9070, Val acc: 0.9092
Epoch [6/10], Train Loss: 0.2042, Val Loss: 0.2065, Train_acc: 0.9099, Val acc: 0.9143
Epoch [7/10], Train Loss: 0.1951, Val Loss: 0.1984, Train_acc: 0.9134, Val acc: 0.9170
Epoch [8/10], Train Loss: 0.1878, Val Loss: 0.1915, Train_acc: 0.9166, Val acc: 0.9187
Epoch [9/10], Train Loss: 0.1822, Val Loss: 0.1860, Train_acc: 0.9182, Val acc: 0.9207
Epoch [10/10], Train Loss: 0.1771, Val Loss: 0.1815, Train_acc: 0.9202, Val acc: 0.9225
Training model: Inception_v3
Epoch [1/10], Train Loss: 0.3602, Val Loss: 0.2870, Train_acc: 0.8758, Val acc: 0.8907
Epoch [2/10], Train Loss: 0.2722, Val Loss: 0.2583, Train_acc: 0.8883, Val acc: 0.8932
Epoch [3/10], Train Loss: 0.2481, Val Loss: 0.2376, Train_acc: 0.8951, Val acc: 0.8974
Epoch [4/10], Train Loss: 0.2293, Val Loss: 0.2227, Train_acc: 0.9021, Val acc: 0.9027
Epoch [5/10], Train Loss: 0.2165, Val Loss: 0.2094, Train_acc: 0.9069, Val acc: 0.9103
Epoch [6/10], Train Loss: 0.2064, Val Loss: 0.2000, Train_acc: 0.9111, Val acc: 0.9135
Epoch [7/10], Train Loss: 0.1993, Val Loss: 0.1938, Train_acc: 0.9140, Val acc: 0.9152
Epoch [8/10], Train Loss: 0.1938, Val Loss: 0.1874, Train_acc: 0.9162, Val acc: 0.9183
Epoch [9/10], Train Loss: 0.1904, Val Loss: 0.1826, Train_acc: 0.9172, Val acc: 0.9196
Epoch [10/10], Train Loss: 0.1849, Val Loss: 0.1803, Train_acc: 0.9191, Val acc: 0.9201

## ToDo List
- [x] Model initialization using pretrained weights is generating a warning
  because Pretrained=True # is depricated, need to use suggested
  approach

- [x] Refactor the code to move the model creation out of the training
  loop so multiple models can be supported in a much cleaner manner

- [x] Update the mechanism to store predictions(json files) so that each
  model can store its results without manually changing the names

- [ ] Upload results from all models

- [ ] Generate a requirements.txt file for the project

- [ ] Create a single data class that splits the training and
  validation dataset internally.

- [ ] Implement distrubuted training to speed up the training time