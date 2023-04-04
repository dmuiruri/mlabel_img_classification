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

- [ ] Upload results from ResNet50
- [ ] Upload results from VGG16
- [ ] Upload results from Inception_v3