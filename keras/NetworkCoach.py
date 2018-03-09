from sklearn.metrics import roc_auc_score

from DatasetGenerator import DatasetGenerator

#----------------------------------------------------------------------------------------
#---- This class allows provides methods for training, testing and evaluating networks.
#---- This wrapper is useful when several networks with different parameters has to be
#---- trained and tested

class NetworkCoach:

    #----------------------------------------------------------------------------------------
    #---- Trains the network, using the specified parameters
    #---- In: pathDatabase - path to the directory that contains images
    #---- In: pathFileTrain - path to a file with the training set, where each line is <img path><target vector>
    #---- In: pathFileValidate - path to a file with the validation set, where each line is <img path><target vector>
    #---- In: imgTransformSequence - image transformation sequence
    #---- In: imgWidth - width of the image, that is input into the network
    #---- In: imgHeight - height of the image, that is input into the network
    #---- In: nnModel - network model (no need to compile it)
    #---- In: nnLoss - loss function
    #---- In: nnOptimizer - optimizer for the network
    #---- In: nnBatchSize - size of the bach to use when training the network
    #---- In: nnEpoch - number of epochs to do training
    #---- In: callbackList - list of callback, that specifies what is happening after each epoch / batch is ended
    #---- In: pathModelOutput - path and name of the file to save the trained model

    def netTrainAuto (pathDatabase, pathFileTrain, pathFileValidate, imgTransformTrain, imgTransformVal, imgWidth, imgHeight, nnModel, nnLoss, nnOptimizer, nnBatchsize, nnEpochs, callbackList, pathModelOutput):

        #---- Define generators for training and validation sets
        datasetGeneratorTrain = DatasetGenerator(pathDatabase, pathFileTrain, imgTransformTrain, imgWidth, imgHeight, nnBatchsize, True)
        datasetGeneratorValidate = DatasetGenerator(pathDatabase, pathFileValidate, imgTransformVal, imgWidth, imgHeight, nnBatchsize, False)

        #---- Count number of steps we need to do to lap over the dataset
        stepsTrain = datasetGeneratorTrain.getSize() / nnBatchsize
        stepsValidate = datasetGeneratorValidate.getSize() / nnBatchsize

        nnModel.compile(loss=nnLoss, optimizer=nnOptimizer, metrics=['accuracy'])

        nnModel.fit_generator(
            datasetGeneratorTrain.generate(),
            epochs=nnEpochs,
            steps_per_epoch=stepsTrain,
            validation_data=datasetGeneratorValidate .generate(),
            validation_steps=stepsValidate,
            shuffle=True,
            callbacks=callbackList)

        nnModel.save(pathModelOutput)

    #----------------------------------------------------------------------------------------
    #---- Performs tests of the network, calculates AUROC, TP (for each dimension)
    #---- In: pathDatabase -- path to a directory with images
    #---- In: pathFileTest - path to the file with test set, format <img path><target vector>
    #---- In: imgHeight - height of the image that will be fed to NN
    #---- In: imgWidth - width of the image that will be fed to NN
    #---- In: nnBatchsize - size of the batch during testing phase
    #---- In: nnTargetDim - dimension of the target vector

    def netTestAuto (pathDatabase, pathFileTest, imgTransformTest, imgWidth, imgHeight, nnModel, nnBatchsize, nnTargetDim):

        #---- Define the generator
        datasetGeneratorTest = DatasetGenerator(pathDatabase, pathFileTest, imgTransformTest, imgWidth, imgHeight, nnBatchsize, False)

        #---- Steps to do, to lap over the set once
        databaseSize = int(datasetGeneratorTest.getSize())
        stepsTest = int(databaseSize / nnBatchsize)

        #---- Indexes of test samples
        indexList = datasetGeneratorTest.generateIndexList()

        #---- These are containers to store predicted & ground truth data for AUROC evatluation
        outdataGT = []
        outdataPR = []

        #---- This contatiner is for storing number of true positive examples
        outputTP = []

        #---- Initialize all the containers
        for dim in range(0, nnTargetDim):
            outputTP.append(0)
            outdataGT.append([])
            outdataPR.append([])

        #---- Predict values for each batch

        for i in range(0, stepsTest):

            x, yGT = datasetGeneratorTest.generateBatch(indexList[i * nnBatchsize: (i + 1) * nnBatchsize])
            yPR = nnModel.predict_on_batch(x)

            for sampleID in range(0, nnBatchsize):

                for k in range(0, nnTargetDim):

                    outdataGT[k].append(yGT[sampleID][k])
                    outdataPR[k].append(yPR[sampleID][k])

                    if yGT[sampleID][k] == 1 and yPR[sampleID][k] >= 0.5:  outputTP[k] += 1
                    if yGT[sampleID][k] == 0 and yPR[sampleID][k] < 0.5:  outputTP[k] += 1

        #---- Calculate AUROC, TP accuracy
        auroc = []

        for k in range(0, nnTargetDim):
            outputTP[k] /= databaseSize
            score = roc_auc_score(outdataGT[k], outdataPR[k])
            auroc.append(score)

        #---- Output auroc scores and true positive accuracy
        return auroc, outputTP

    #----------------------------------------------------------------------------------------
    #---- Trains the network, using the specified parameters
    #---- Same functionality as the method above, but custom generator can be specified as parameters
    #---- In: datasetGeneratorTrain - generator for the training set
    #---- In: datasetGeneratorValidate - generator for the validation set
    #---- In: nnModel - network model
    #---- In: nnEpochs - number of epochs to train the model
    #---- In: nnBatchSize - size of the batches
    #---- In: nnLoss - loss function
    #---- In: nnOptimizer - optimizer, learning method
    #---- In: callbackList - callbacks
    #---- pathModelOutput - name of the file to save the model to

    def netTrain (datasetGeneratorTrain, datasetGeneratorValidate, nnModel, nnEpochs, nnBatchsize, nnLoss, nnOptimizer, callbackList, pathModelOutput):
        #---- Count number of steps we need to do to lap over the dataset
        stepsTrain = datasetGeneratorTrain.getSize() / nnBatchsize
        stepsValidate = datasetGeneratorValidate.getSize() / nnBatchsize

        nnModel.compile(loss=nnLoss, optimizer=nnOptimizer, metrics=['accuracy'])

        nnModel.fit_generator(
            datasetGeneratorTrain.generate(),
            epochs=nnEpochs,
            steps_per_epoch=stepsTrain,
            validation_data=datasetGeneratorValidate .generate(),
            validation_steps=stepsValidate,
            shuffle=True,
            callbacks=callbackList)

        nnModel.save(pathModelOutput)

    #----------------------------------------------------------------------------------------
    #---- Performs tests of the network, calculates AUROC, TP (for each dimension)
    #---- Same as the test function above, but the generator should be send as a parameter instead
    #---- In: datasetGeneratorTest - dataset generator
    #---- In: nnModel - network model
    #---- In: nnBatchsize - batchsize
    #---- In: nnTargetDim - dimension of the output vector

    def netTest (datasetGeneratorTest, nnModel, nnBatchsize, nnTargetDim):

        databaseSize = int(datasetGeneratorTest.getSize())
        stepsTest = int(databaseSize / nnBatchsize)
        indexList = datasetGeneratorTest.generateIndexList()
        #---- These are containers to store predicted & ground truth data for AUROC evatluation
        outdataGT = []
        outdataPR = []

        #---- This contatiner is for storing number of true positive examples
        outputTP = []

        #---- Initialize all the containers
        for dim in range(0, nnTargetDim):
            outputTP.append(0)
            outdataGT.append([])
            outdataPR.append([])

        for i in range(0, stepsTest):

            x, yGT = datasetGeneratorTest.generateBatch(indexList[i * nnBatchsize: (i + 1) * nnBatchsize])
            yPR = nnModel.predict_on_batch(x)

            for sampleID in range(0, nnBatchsize):

                for k in range(0, nnTargetDim):

                    outdataGT[k].append(yGT[sampleID][k])
                    outdataPR[k].append(yPR[sampleID][k])

                    if yGT[sampleID][k] == 1 and yPR[sampleID][k] >= 0.5:  outputTP[k] += 1
                    if yGT[sampleID][k] == 0 and yPR[sampleID][k] < 0.5:  outputTP[k] += 1

        #---- Calculate AUROC, TP accuracy
        auroc = []

        for k in range(0, nnTargetDim):
            outputTP[k] /= databaseSize
            score = roc_auc_score(outdataGT[k], outdataPR[k])
            auroc.append(score)

        #---- Output auroc scores and true positive accuracy
        return auroc, outputTP




