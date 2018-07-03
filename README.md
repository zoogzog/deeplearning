# deeplearning
Utility scripts for deep learning

For testing a smaller subset the butterfly [database](http://www.josiahwang.com/dataset/leedsbutterfly/) is used

__Keras__

List of classes
- DatasetGenerator - class for loading image defined in a file for training with a neural network
- ImageTransform - a support class, that allows perform transformations of the images, being loaded
with the DatasetGenerator class
- DenseNet - the primary class for building [dense neural networks](https://arxiv.org/pdf/1608.06993.pdf)
- DenseNetScale - a class that describes a custom layer for dense neural networks
- DenseNetSettings - a wrapper class for encapsulating network parameters
- NetworkCoach - class that provides wrappers for training and testing procedures

Examples
- exampleDatasetGenerator - shows how to use the DatasetGenerator and ImageTransform
- exampleDenseNet - shows how to define a densenet with the DenseNet class
- exampleNetworkCoach - shows how to use the NetworkCoach class 

__PyTorch__
Tested on: pytorch 0.4.0, torchvision 0.2.1, CUDA 9.0

The following classes allow to conduct the whole cycle of training, validation and testing
of neural networks for solving image classification problem. Different networks, datasets,
databases could be tested by speciying parameters of the task.

The main script is *LauncherMain.py*, which launches sub-script *LauncherTask.py* for each individual 
task. Output includes: training log, trained model, accuracy on test (AUROC, TP, FP, F-Score)

List of classes
- AccuracyCalculator - class for computing various statistics for evaluating trained network accuracy
- DatasetGenerator - dataset loader, which handles processing of dataset files (see ../database/dataset1.txt)
- LauncherMain - main class for launching execution of all tasks
- LauncherTask - class that launches training for each individual task
- LossZoo - different types of losses
- NetworkCoach - class for training and testing
- NetworkZoo - various models (from torchvision, that allows higher level of customization)
- TaskManager - class that controls states: training, validation, testing, and outputs logs, results
 
