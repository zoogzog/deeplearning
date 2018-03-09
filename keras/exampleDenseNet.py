#--------------------------------------------------------------------------------
#---- This examples shows how to use a DenseNet class
#---- DenseNetSettings - is a helper class that encapsulates network parameters
#---- DenseNetScale - is a custom layer class, required for DenseNet
#---- There are several ways to get a DenseNet model
#---- * By selecting a predefined model from model zoo (dense-121, 169, 201, 264)
#---- * By manually specifying parameters of the network
#---- * By passing network parameters using the DenseNetSettings class
#--------------------------------------------------------------------------------

from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator

from DenseNet import DenseNet
from DenseNetSettings import DenseNetSettings

#--------------------------------------------------------------------------------

#---- Path to a directory where images are located
#---- For this example the Keras ImageDataGenerator will be used for loading images
rootDirectory = '../database'

#---- Parameters for training the neural network
imgWidth = 224
imgHeight = 224
batchsize = 16

nnEpochs = 3

nnLoss = binary_crossentropy
nnOptimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

generatorImage = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
generatorFlow = generatorImage.flow_from_directory(rootDirectory, target_size=(imgWidth, imgHeight), batch_size=batchsize, class_mode='binary')

#---- Parameters of the network
optInputShape = (imgWidth, imgHeight, 3)
optActivation = 'sigmoid'
optClassCount = 1
optCompression = 0.5
optDenseBlockCount = 4
optGrowthRate = 32
optConvKernelCount = 64
optLayersPerBlock = [6, 12, 24, 16]
optBottleneck = True
optDropoutRate = 0.0
optWeightDecay = 1e-4
optSubsampleInitBlock = True


#---- A model from model zoo (121, 169, 201, 264) can be obtained like this
#model = DenseNet.getModelNet121((imgWidth, imgHeight, 3), 'sigmoid', 1, optCompression=0.5)

#---- A model with custom number of layers can be obtained like this
#model = DenseNet.getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock,
#				  optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)

#---- A model can be created using DenseNetSettings class
settings = DenseNetSettings(imgWidth, imgHeight, optClassCount, optGrowthRate, optLayersPerBlock, optConvKernelCount,
                            optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation)
model = DenseNet.getModelFromSettings(settings)

#---- Compile and train the defined
model.compile(loss=nnLoss, optimizer=nnOptimizer, metrics=['accuracy'])
model.fit_generator(generatorFlow, epochs = nnEpochs,  steps_per_epoch = 50, shuffle = True)

