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
| Epoch | Train Loss | Val Loss | Train acc | Val acc |
|-------|------------|----------|-----------|---------|
| 1/10  | 0.2582     | 0.1551   | 0.9056    | 0.9344  |
| 2/10  | 0.1457     | 0.1239   | 0.9383    | 0.9485  |
| 3/10  | 0.1139     | 0.1139   | 0.9530    | 0.9532  |
| 4/10  | 0.0905     | 0.1122   | 0.9642    | 0.9549  |
| 5/10  | 0.0718     | 0.1148   | 0.9724    | 0.9544  |
| 6/10  | 0.0599     | 0.1195   | 0.9776    | 0.9530  |
| 7/10  | 0.0503     | 0.1224   | 0.9811    | 0.9548  |
| 8/10  | 0.0430     | 0.1241   | 0.9842    | 0.9546  |
| 9/10  | 0.0364     | 0.1270   | 0.9867    | 0.9548  |
| 10/10 | 0.0303     | 0.1347   | 0.9889    | 0.9552  |

Training model: VGG16
| Epoch | Train Loss | Val Loss | Train_acc | Val acc |
|-------|------------|----------|-----------|---------|
| 1/10  | 0.4531     | 0.3498   | 0.8766    | 0.8889  |
| 2/10  | 0.2929     | 0.2796   | 0.8913    | 0.8929  |
| 3/10  | 0.2508     | 0.2484   | 0.8968    | 0.8988  |
| 4/10  | 0.2293     | 0.2299   | 0.9019    | 0.9048  |
| 5/10  | 0.2150     | 0.2170   | 0.9070    | 0.9092  |
| 6/10  | 0.2042     | 0.2065   | 0.9099    | 0.9143  |
| 7/10  | 0.1951     | 0.1984   | 0.9134    | 0.9170  |
| 8/10  | 0.1878     | 0.1915   | 0.9166    | 0.9187  |
| 9/10  | 0.1822     | 0.1860   | 0.9182    | 0.9207  |
| 10/10 | 0.1771     | 0.1815   | 0.9202    | 0.9225  |


Training model: Inception_v3
| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|-----------|---------|
| 1/10  | 0.3602     | 0.2870   | 0.8758    | 0.8907  |
| 2/10  | 0.2722     | 0.2583   | 0.8883    | 0.8932  |
| 3/10  | 0.2481     | 0.2376   | 0.8951    | 0.8974  |
| 4/10  | 0.2293     | 0.2227   | 0.9021    | 0.9027  |
| 5/10  | 0.2165     | 0.2094   | 0.9069    | 0.9103  |
| 6/10  | 0.2064     | 0.2000   | 0.9111    | 0.9135  |
| 7/10  | 0.1993     | 0.1938   | 0.9140    | 0.9152  |
| 8/10  | 0.1938     | 0.1874   | 0.9162    | 0.9183  |
| 9/10  | 0.1904     | 0.1826   | 0.9172    | 0.9196  |
| 10/10 | 0.1849     | 0.1803   | 0.9191    | 0.9201  |

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