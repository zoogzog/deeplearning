from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Concatenate, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import load_model

from DenseNetSettings import DenseNetSettings
from DenseNetScale import Scale
from LossFunction import LossFunction

#----------------------------------------------------------------------------------------
#------ This class is an implementation of the DenseNN
#------ Uses Tensorflow as the backend.
#------ Python 3.5.4, Keras 2.0, Tensorflow 1.3.0
#------ Keras should be switched to 'channels last' mode!
#------ This implementation is based on titu1994's implementation.
#------ Implemented as a separate 'static' class. Added optional dropout in transition block
#------ User is forced to specify all the parameters of the networks, since this is much safer.
#------ Depth parameter is removed. For DenseNet in paper 3 dense blocks number of layer are same 
#------ and #layer = (depth - 4) / 3
#------ Paper: https://arxiv.org/abs/1608.06993
#------ Alternative implementation 1: https://github.com/liuzhuang13/DenseNet
#------ Alternative implementation 2: https://github.com/titu1994/DenseNet
#------ Alternative implementation 3: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet
#----------------------------------------------------------------------------------------

class DenseNet:

	#---- Some cheaty coefficient claimed to be obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
	#---- Probably avoid using bottleneck at all
	defaultBottleNeck = 4

	#----- If channel first then 1, channel last then -1
	defaultConcatAxis = -1
	
	#---- MODEL GENERATION ERRORS
	ERCODE_ACTF = "1"
	ERCODE_ARSIZE = "3"
	ERCODE_CMPRS = "4"
	
	ERROR_CODE_INFO = {}
	ERROR_CODE_INFO[ERCODE_ACTF] = "Activation function should be 'sigmoid' or 'softmax'"
	ERROR_CODE_INFO[ERCODE_ARSIZE] = "Size of the array, specifying number of layers should be equal to the number of blocks"
	ERROR_CODE_INFO[ERCODE_CMPRS] = "Compression rate should be in range (0, 1]"
	
	USE_LAYER_SCALE = True
	
	#-------------------------------------------------------------------------
	#---- Desription: builds convolution block
	#---- Convolution block = BatchNorm + Relu + <Conv2D[1,1]> + Conv2D[3,3] + <DropOut>
	#---- Bottleneck layer and dropout layers are optional
	#---- In: network - link to the previous layer of the network/model, ie keras tensor
	#---- In: optConvKernelCount - (int) number of convolution filters to apply
	#---- In: optBottleneck - use bottleneck/not
	#---- In: optDropoutRate - if specified, then adds a dropout layer to the network
	#---- In optWeightDecay - (double) weight decay, default 1E-4
	#---- Out: network - layer configuration
	#-------------------------------------------------------------------------
	def buildBlockConvolution (network, optConvKernelCount, optBottleneck=False, optDropoutRate=None, optWeightDecay=1E-4):
	
		
		#>>>>>>>> BOTTLENECK >>>>>>>>
		if optBottleneck:
			network = BatchNormalization(axis=DenseNet.defaultConcatAxis, epsilon=1.1e-5)(network)
			if DenseNet.USE_LAYER_SCALE == True: network = Scale(axis=3)(network)
			network = Activation('relu')(network)
			network = Conv2D(optConvKernelCount * DenseNet.defaultBottleNeck, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(optWeightDecay))(network)
		#<<<<<<<< BOTTLENECK <<<<<<<<
		
		#>>>>>>>> DROPOUT >>>>>>>>
		if optDropoutRate:
			network = Dropout(optDropoutRate)(network)
		#<<<<<<<< DROPOUT <<<<<<<<
		
		network = BatchNormalization(axis=DenseNet.defaultConcatAxis, epsilon=1.1e-5)(network)
		if DenseNet.USE_LAYER_SCALE == True: network = Scale(axis=3)(network)
		network = Activation('relu')(network)
		network = ZeroPadding2D((1, 1))(network)
		network = Conv2D(optConvKernelCount, (3, 3), use_bias=False)(network)
		
		#>>>>>>>> DROPOUT >>>>>>>>
		if optDropoutRate:
			network = Dropout(optDropoutRate)(network)
		#<<<<<<<< DROPOUT <<<<<<<<
		
		return network
	
	#------------------------------------------------------------------------
	#---- Desription: builds a convolution block
	#---- Transition block = BatchNorm + Relu + Conv2D[1,1] * # + <DropOut> + AvPool[2,2][2,2]
	#---- Dropout layer is optional
	#---- In: network - link to the previous layer of the network/model, ie keras tensor
	#---- In: optConvKernelCount - (int) number of convolution filters to apply
	#---- In: optWeightDecay - (double) weight decay, default 1E-4
	#---- In: optCompression - compression factor, for BC use  optCompression=0.5
	#---- Out: network - layer configuration
	#-------------------------------------------------------------------------
	def buildBlockTransition (network, optConvKernelCount, optWeightDecay, optCompression, optDropoutRate=None):
	
		network = BatchNormalization(axis=DenseNet.defaultConcatAxis, epsilon=1.1e-5)(network)
		if DenseNet.USE_LAYER_SCALE == True: network = Scale(axis=3)(network)
		network = Activation('relu')(network)
		network = Conv2D(int(optConvKernelCount * optCompression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(optWeightDecay))(network)
		
		#>>>>>>>> DROPOUT >>>>>>>>
		if optDropoutRate:
			network = Dropout(optDropoutRate)(network)
		#<<<<<<<< DROPOUT <<<<<<<<		
			
		network = AveragePooling2D((2, 2), strides=(2, 2))(network)

		return network
	
	#-------------------------------------------------------------------------
	#---- Desription: builds a dense block - consists of several transition blocks
	#---- In: network - link to the previous layer of the network/model, ie keras tensor
	#---- In: optLayersConv - number of convolution layers in the dense block
	#---- In: optConvKernelCount - (int) number of convolution filters to apply
	#---- In: optGrowthRate - number of filters to add
	#---- In: optBottleneck - use bottleneck/not
	#---- In: optDropoutRate - if specified, then adds a dropout layer to the network
	#---- In: optWeightDecay - (double) weight decay, default 1E-4
	#---- Out: network - layer configuration
	def buildBlockDense (network, optLayersConv, optConvKernelCount, optGrowthRate, optBottleneck=False, optDropoutRate=None, optWeightDecay=1e-4):
	
		#---- This list is necessary to tie together each convolution layers of 
		#---- the dense block to subsequent convolution layers.
		listNetworkConnectivity = [network]
		
		#---- Add convolution layers to the dense block
		for i in range(optLayersConv):
		
			#----1: Build a new convolution layer
			network = DenseNet.buildBlockConvolution (network, optGrowthRate, optBottleneck, optDropoutRate, optWeightDecay)
			
			#----2: Add this layer to the list of layers
			listNetworkConnectivity.append(network)
			
			#----3: Merge previous layers with this layer
			network = Concatenate(axis=DenseNet.defaultConcatAxis)(listNetworkConnectivity)
			
			#----4: Update growth rate counter
			optConvKernelCount += optGrowthRate

		return network, optConvKernelCount
	
	#-------------------------------------------------------------------------
	#---- Description: builds a network architecture
	#---- In: optInputShape - shape descriptor (width, height, channels) or None
	#---- In: optDenseBlockCount - number of dense blocks in the network
	#---- In: optGrowthRate - number of filters to add per dense block
	#---- In: optConvKernelCount - initial number of convolution filters. If -1, then equal 2 * growth rate
	#---- In: optLayersPerBlock - list of the number of layers per each block, example: [6, 12, 32, 32]
	#---- In: optBottleneck - False or True, flag that switches bottleneck layer on or off
	#---- In: optCompression - (0, 1] = 1 - reduction (thetta)
	#---- In: optWeightDecay - weight decay rate
	#---- In: optSubsampleInitBlock - False or True, if true then additional convolution and MaxPooling2D added
	#---- In: optActivation - activation function: sigmoid for binary, softmax for multi-class
	#---- In: optClassCount - number of classes in the classification problem
	#---- Out: network architecture as list of layers
	
	def buildDenseNet (optInputTensor, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock, 
					   optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount):

		#---- 1: Add initial convolution with or without subsampling (zero padding included)
		
		if optSubsampleInitBlock:
			kernelInit = (7, 7)
			strideInit = (2, 2)
			
			network = ZeroPadding2D((3, 3))(optInputTensor)
			network = Conv2D (optConvKernelCount, kernelInit, name='initConv2D', strides=strideInit, use_bias=False, kernel_regularizer=l2(optWeightDecay))(network)
			network = BatchNormalization(axis=DenseNet.defaultConcatAxis, epsilon=1.1e-5, name='initSubsampleBN')(network)
			if DenseNet.USE_LAYER_SCALE == True: network = Scale(axis=3)(network)
			network = Activation('relu')(network)	
			network = ZeroPadding2D((1, 1))(network)
			network = MaxPooling2D((3, 3), strides=(2, 2), name='initSubsampleMP')(network)
		else:
			kernelInit = (3, 3)
			strideInit = (1, 1)
			
			network = Conv2D (optConvKernelCount, kernelInit, kernel_initializer='he_normal', padding='same', name='initConv2D', strides=strideInit, use_bias=False, kernel_regularizer=l2(optWeightDecay))(optInputTensor)

					  
		#---- 2: Construct dense block architecture
		for blockID in range (optDenseBlockCount -1):
		
			#---- Add dense block
			network, optConvKernelCount = DenseNet.buildBlockDense(network, optLayersPerBlock[blockID], optConvKernelCount, optGrowthRate, optBottleneck, optDropoutRate, optWeightDecay)
			
			#---- Add transition block
			network = DenseNet.buildBlockTransition (network, optConvKernelCount, optWeightDecay, optCompression, optDropoutRate)
			
			#---- this for compression 
			optConvKernelCount = int (optConvKernelCount * optCompression)
					   
					  
		#---- 3: Construct the last dense block without a transition block
		network, optConvKernelCount = DenseNet.buildBlockDense(network, optLayersPerBlock[-1], optConvKernelCount, optGrowthRate, optBottleneck, optDropoutRate, optWeightDecay)
		
		network = BatchNormalization(axis = DenseNet.defaultConcatAxis, epsilon=1.1e-5)(network)
		if DenseNet.USE_LAYER_SCALE == True: network = Scale(axis=3)(network)
		network = Activation('relu')(network)
		network = GlobalAveragePooling2D()(network)
					  
		#---- 4: Add fully connected layer
		network = Dense(optClassCount)(network)
		network = Activation(optActivation)(network)
					   
		return network
		
	#-----------------------------------------------------------------------
	#---- Description: generates a dense network with custom architecture
	#---- In: optInputShape - shape descriptor (width, height, channels) or None
	#---- In: optDenseBlockCount - number of dense blocks in the network
	#---- In: optGrowthRate - number of feature maps generated with each layer transition
	#---- In: optConvKernelCount - initial number of convolution filters. If -1, then equal 2 * growth rate
	#---- In: optLayersPerBlock - list of the number of layers per each block, example: [6, 12, 32, 32]
	#---- In: optBottleneck - False or True, flag that switches bottleneck layer on or off
	#---- In: optCompression - (0, 1] = 1 - reduction (thetta)
	#---- In: optDropoutRate - dropout rate
	#---- In: optWeightDecay - weight decay rate
	#---- In: optSubsampleInitBlock - False or True, if true then additional convolution and MaxPooling2D added
	#---- In: optActivation - activation function: sigmoid for binary, softmax for multi-class
	#---- In: optClassCount - number of classes in the classification problem
	#---- Out: Keras model or errorcode (string) if something goes wrong, check error info
		
	def getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock, optBottleneck, 
				  optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount):
	
		#++++++++++++++++++++++++++++++++++++++++++
		#---- Do additional input check here
		
		#---- 1. Check if activation function is set up correctly: only sigmoid and softmax
		if optActivation not in ['softmax', 'sigmoid']:
			return ERCODE_ACTF
			
		#----2. Check if number of dense blocks equals number of elements in array, that specifies dense layers
		if len(optLayersPerBlock) < optDenseBlockCount:
			return ERCODE_ARSIZE
			
		#----3. Compression rate should be in range (0, 1]
		if optCompression <= 0 or optCompression > 1:
			return ERCODE_CMPRS
			
		#++++++++++++++++++++++++++++++++++++++++++
		
		networkInput = Input(shape=optInputShape)
		networkOutput = DenseNet.buildDenseNet(networkInput, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock, optBottleneck, 
											   optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)
									  
		networkModel = Model(networkInput, networkOutput)
		
		return networkModel
	
	#-----------------------------------------------------------------------
	#---- Give me my model, quickly!
	#---- Same as getModel, with less paramters to define 
	#---- Be careful when defining models with small images width and height < 32
	
	def getModelQuick (optInputShape, optGrowthRate, optLayersPerBlock, optClassCount):
	
		optDenseBlockCount = len(optLayersPerBlock)
		optConvKernelCount = 2 * optGrowthRate
		optBottleneck = True
		optCompression = 0.5
		optDropoutRate = 0.0
		optWeightDecay = 1e-4
		optSubsampleInitBlock = True
		
		if optClassCount > 1:
			optActivation = 'softmax'
		elif optClassCount == 1:
			optActivation = 'sigmoid'
			
		return DenseNet.getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock, optBottleneck, 
						 optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)
	
	#-----------------------------------------------------------------------
	#---- Builds model from  settings specified by class DenseNetSettings
	
	def getModelFromSettings (settings):
		return DenseNet.getModel(
			settings.sInputShape, 
			settings.sDenseBlockCount, 
			settings.sGrowthRate, 
			settings.sConvKernelCount, 
			settings.sLayersPerBlock, 
			settings.sBottleneck, 
			settings.sCompression, 
			settings.sDropoutRate, 
			settings.sWeightDecay, 
			settings.sSubsampleInitBlock, 
			settings.sActivation, 
			settings.sClassCount
			)
			
	#-----------------------------------------------------------------------
	#---- Description: generates a dense network with 121-layer architecture
	#---- In: optInputShape - shape of the input tensor (x, x, 3)
	#---- In: optActivation - activation function 'softmax' or 'sigmoid'
	#---- In: optClassCount - number of output classes
	#---- Out: Keras model or errorcode (string) if something goes wrong, check error info
	
	def getModelNet121 (optInputShape, optActivation, optClassCount, optCompression = 1):
		optDenseBlockCount = 4
		optGrowthRate = 32
		optConvKernelCount = 64
		optLayersPerBlock = [6, 12, 24, 16]
		optBottleneck = True
		optDropoutRate = 0.0
		optWeightDecay = 1e-4
		optSubsampleInitBlock = True
		
		return DenseNet.getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock,
				  optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)
		
	def getSettingsModelNet121 ():
		
		return "DenseNet-121: dense_block_count=4, gr_rate=32, ker_count=64, layer_conf=[6,12,24,16], bottle=true, compression=1, dropout=0, wdecay=1e-4, subsamp=true"
	
	#-----------------------------------------------------------------------
	#---- Description: generates a dense network with 169-layer architecture
	#---- In: optInputShape - shape of the input tensor (x, x, 3)
	#---- In: optActivation - activation function 'softmax' or 'sigmoid'
	#---- In: optClassCount - number of output classes
	#---- Out: Keras model or errorcode (string) if something goes wrong, check error info
	
	def getModelNet169 (optInputShape, optActivation, optClassCount, optCompression = 1):
		optDenseBlockCount = 4
		optGrowthRate = 32
		optConvKernelCount = 64
		optLayersPerBlock = [6, 12, 32, 32]
		optBottleneck = True
		optDropoutRate = 0.0
		optWeightDecay = 1e-4
		optSubsampleInitBlock = True
		
		return getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock,
				  optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)
		
	def getSettingsModelNet169 ():
		
		return "DenseNet-169: dense_block_count=4, gr_rate=32, ker_count=64, layer_conf=[6,12,32,32], bottle=true, compression=1, dropout=0, wdecay=1e-4, subsamp=true"

	#-----------------------------------------------------------------------
		#---- Description: generates a dense network with 201-layer architecture
	#---- In: optInputShape - shape of the input tensor (x, x, 3)
	#---- In: optActivation - activation function 'softmax' or 'sigmoid'
	#---- In: optClassCount - number of output classes
	#---- Out: Keras model or errorcode (string) if something goes wrong, check error info
	
	def getModelNet201 (optInputShape, optActivation, optClassCount, optCompression = 1):
		optDenseBlockCount = 4
		optGrowthRate = 32
		optConvKernelCount = 64
		optLayersPerBlock = [6, 12, 48, 32]
		optBottleneck = True
		optDropoutRate = 0.0
		optWeightDecay = 1e-4
		optSubsampleInitBlock = True
		
		return getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock,
				  optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)
		
	def getSettingsModelNet201():
	
		return "DenseNet-201: dense_block_count=4, gr_rate=32, ker_count=64, layer_conf=[6,12,48,32], bottle=true, compression=1, dropout=0, wdecay=1e-4, subsamp=true"
				  
	#-----------------------------------------------------------------------
	#---- Description: generates a dense network with 264-layer architecture
	#---- In: optInputShape - shape of the input tensor (x, x, 3)
	#---- In: optActivation - activation function 'softmax' or 'sigmoid'
	#---- In: optClassCount - number of output classes
	#---- Out: Keras model or errorcode (string) if something goes wrong, check error info
	
	def getModelNet264 (optInputShape, optActivation, optClassCount, optCompression = 1):
		optDenseBlockCount = 4
		optGrowthRate = 32
		optConvKernelCount = 64
		optLayersPerBlock = [6, 12, 64, 48]
		optBottleneck = True
		optDropoutRate = 0.0
		optWeightDecay = 1e-4
		optSubsampleInitBlock = True
		
		return getModel (optInputShape, optDenseBlockCount, optGrowthRate, optConvKernelCount, optLayersPerBlock,
				  optBottleneck, optCompression, optDropoutRate, optWeightDecay, optSubsampleInitBlock, optActivation, optClassCount)
	
	def getSettingsModelNet264():
	
		return "DenseNet-264: dense_block_count=4, gr_rate=32, ker_count=64, layer_conf=[6,12,64,48], bottle=true, compression=1, dropout=0, wdecay=1e-4, subsamp=true"
	
	#-----------------------------------------------------------------------
	
	def loadModel (path):
		
		model = load_model(path, custom_objects={'Scale':Scale, 'lossSumBinaryCrossEntropy':LossFunction.lossSumBinaryCrossEntropy})
		
		return model

	#---- Load weights into a dense net model, rearrange model to classify into classes
	def loadWeightsImageNet (model, pathWeights, optClassCount, optActivation):
		model.load_weights(pathWeights)

		model.layers.pop()
		model.layers.pop()

		x = Dense(14)(model.layers[-1].output)
		o = Activation('sigmoid')(x)

		return Model(model.input, output=[o])

		
#----------------------------------------------------------------------------------------

