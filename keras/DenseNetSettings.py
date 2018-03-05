#--------------------------------------------------------------------------------
#------ Python 3.5.4, Keras 2.0, Tensorflow 1.3.0
#------ Keras should be switched to 'channels last' mode!
#------ Class, that incapsulates all the DenseNN settings for convenience
#------ Implementation by A.G.
#--------------------------------------------------------------------------------

class DenseNetSettings:
	
		#---- (INT, INT, 3) - tensor for resolutions of input image
		sInputShape = (32, 32, 3)
		
		#---- [INT] - number of filters generated on every conv layer in a dense block
		sGrowthRate = 12
		
		#---- [INT_ARRAY] - number of layers per dense blocks
		sLayersPerBlock = [32, 32, 32]
		
		#---- [INT] - number of classes 
		sClassCount = 10
		
		#---- Must not be defined by user
		sDenseBlockCount = len(sLayersPerBlock)
		
		#---- [INT] - initial number of filters in conv layer
		sConvKernelCount = 2 * sGrowthRate
		
		#---- [BOOL] - is use bottleneck layer in dense block
		sBottleneck = True
		
		#---- [FLOAT] - compression rate 
		sCompression = 0.5
		
		#---- [FLOAR] - dropout rate
		sDropoutRate = 0.0
		
		#---- [DOUBLE] - weight decay
		sWeightDecay = 1e-4
		
		#---- [BOOL] - is perform subsampling 
		sSubsampleInitBlock = False
		
		#---- [softmax, sigmoid] - activation function
		sActivation = 'softmax'
		
		def __init__ (self, inImgWidth, inImgHeight, outClassCount, nnGrowthRate, nnLayersPerBlock, nnConvKernelCount, nnBottleNeck, nnCompression, nnDropout, nnWeightDecay, nnSubsample, nnActivation):
			self.sInputShape = (inImgWidth, inImgHeight, 3)
			self.sGrowthRate = nnGrowthRate
			self.sLayersPerBlock = nnLayersPerBlock
			self.sClassCount = outClassCount
			self.sDenseBlockCount = len(nnLayersPerBlock)
			self.sConvKernelCount = nnConvKernelCount
			self.sBottleneck = nnBottleNeck
			self.sCompression = nnCompression
			self.sDropoutRate = nnDropout
			self.sWeightDecay = nnWeightDecay
			self.sSubsampleInitBlock = nnSubsample
			self.sActivation = nnActivation
			
#--------------------------------------------------------------------------------