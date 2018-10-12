import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

sys.path.append('../')
from task.TaskSettings import *
from algorithm.AlgorithmClassification import *

# --------------------------------------------------------------------------------

class TaskClassification():

    # --------------------------------------------------------------------------------

    def train(self, model, pathLog, pathModel, settings, workerCount = 0, isSilentMode = False):

        datasetTrain = settings.getDatageneratorTrain()
        datasetValidation = settings.getDatageneratorValidation()

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=settings.getBatchSize(), shuffle=True, num_workers=workerCount, pin_memory=False)
        dataLoaderValidation = DataLoader(dataset=datasetValidation, batch_size=settings.getBatchSize(), shuffle=False, num_workers=workerCount, pin_memory=False)

        optimizer = optim.Adam(model.parameters(), lr=settings.getLearningRate(), betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min')

        loss = settings.getLoss(datasetTrain.getweights())

        epochs = settings.getEpoch()

        AlgorithmClassification.train(dataLoaderTrain, dataLoaderValidation, model, epochs, loss, optimizer, scheduler, pathModel, pathLog, isSilentMode)

    # --------------------------------------------------------------------------------

    def test(self, model, pathOutputAcc, pathOutputPred, settings, workerCount = 0, isSilentMode = False):

        datasetTest = settings.getDatageneratorTest()

        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size = settings.getBatchSize(), shuffle=False, num_workers=workerCount, pin_memory=False)

        AlgorithmClassification.test(dataLoaderTest, model, pathOutputAcc, pathOutputPred, isSilentMode)


    # --------------------------------------------------------------------------------

    def launch(self, settings, workerCount = 0, isSilentMode = False):
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%d%m%Y")
        timestampStart = timestampDate + '-' + timestampTime

        fileModel = "m-" + timestampStart + ".pth.tar"
        fileLog = "log-" + timestampStart + ".txt"
        fileAcc = "acc-" + timestampStart + ".txt"
        filePred = "pred-" + timestampStart + ".txt"

        pathOutputModel = os.path.join(settings.getPathOutModel(), fileModel)
        pathOutputLog = os.path.join(settings.getPathOutLog(), fileLog)
        pathOutputAcc = os.path.join(settings.getPathOutAccuracy(), fileAcc)
        pathOutputPred = os.path.join(settings.getPathOutAccuracy(), filePred)

        model = settings.getNetwork()
        model = torch.nn.DataParallel(model).cuda()

        # ------------------ TRAIN ------------------
        if isSilentMode == False:
            print("--------------------------------------------------")
            print(settings.getsettings())
            print("--------------------------------------------------")
            print("[TASK-CLASSIFICATION] Launching training procedure")

        ostreamLog = open(pathOutputLog, 'w')
        ostreamLog.write(settings.getsettings())
        ostreamLog.close()

        self.train(model, pathOutputLog, pathOutputModel, settings, workerCount, isSilentMode)

        # ------------------ TEST -------------------
        if isSilentMode == False:
            print("--------------------------------------------------")
            print("[TASK-CLASSIFICATION] Launching testing procedure")

        self.test(model, pathOutputAcc, pathOutputPred, settings, workerCount, isSilentMode)


# --------------------------------------------------------------------------------