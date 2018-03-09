# deeplearning
Utility scripts for deep learning

For testing the small version of the butterfly [database](http://www.josiahwang.com/dataset/leedsbutterfly/) is used

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
 
