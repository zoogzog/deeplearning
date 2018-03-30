import numpy as np

from sklearn.metrics.ranking import roc_auc_score


#-------------------------------------------------------------------------------
#---- This class provides method for computing various statistis for
#---- evaluation the accuracy of the trained neural network

class AccuracyCalculator():

    #-------------------------------------------------------------------------------
    #----- This method computes AUROC value for each class
    #----- In: dataGT - a numpy array of ground truth data vectors
    #----- In: dataPRED - a numpy array of predicted data vectors
    #----- In: dim - dimension of the vectors
    #----- Out: array, where each element is an AUROC for a class

    def computeAUROC(dataGT, dataPRED, dim):
        outAUROC = []

        for i in range(dim):
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))

        return outAUROC

    #--------------------------------------------------------------------------------
    #---- This method computes various statistics: true positive, true negative, etc
    #----- In: dataGT - a numpy array of ground truth data vectors
    #----- In: dataPRED - a numpy array of predicted data vectors
    #----- In: dim - dimension of the vectors
    #----- Output:

    def computeAccuracyStatistics(dataGT, dataPRED, dim, threshold = 0.5):

        #---- True positive: 1 predicted as 1
        outTruePositive = [0] * dim

        #---- False positive: 0 predicted as 1
        outFalsePositive = [0] * dim

        #---- True negative: 0 predicted as 0
        outTrueNegative = [0] * dim

        #---- False negative: 1 predicted as 0
        outFalseNegative = [0] * dim

        length = len(dataGT)

        for i in range (0, length):
            vectorGroundTruth = dataGT[i]
            vectorPredicted = dataPRED[i]

            for k in range (0, dim):

                if (vectorGroundTruth[k] == 1 and vectorPredicted[k] >= threshold): outTruePositive[k] += 1
                if (vectorGroundTruth[k] == 0 and vectorPredicted[k] >= threshold): outFalsePositive[k] += 1
                if (vectorGroundTruth[k] == 0 and vectorPredicted[k] < threshold): outTrueNegative[k] += 1
                if (vectorGroundTruth[k] == 1 and vectorPredicted[k] < threshold): outFalseNegative[k] += 1

        return outTruePositive, outFalsePositive, outTrueNegative, outFalseNegative

    # --------------------------------------------------------------------------------

    def computeAccuracyPerClass (stTP, stFP, stTN, stFN):

        dim = len(stTP)

        outAccuracy = [0] * dim

        for i in range(0, dim):

            total = stTP[i] + stFP[i] + stTN[i] + stFN[i]

            outAccuracy[i] = (stTP[i] + stTN[i]) / total

        return outAccuracy

    def computeAccuracyFscore (stTP, stFP, stTN, stFN):

        dim = len(stTP)

        outFscore = [0] * dim

        for i in range(0, dim):

            norm = 2* stTP[i] + stFP[i] + stFN[i]

            outFscore[i] = (2 * stTP[i]) / norm

        return outFscore