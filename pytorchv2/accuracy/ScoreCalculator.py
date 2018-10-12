from sklearn.metrics.ranking import roc_auc_score

# -------------------------------------------------------------------------------
# This class provides methods for computing various statistics for
# evaluation the accuracy of the trained neural networks.
# -------------------------------------------------------------------------------

class ScoreCalculator():

    # -------------------------------------------------------------------------------

    def computeAUROC(dataGT, dataPRED, dim):
        """
        Computes the per class AUROC score
        :param dataGT: numpy array of ground truth data vectors
        :param dataPRED: numpy array of predicted data vectors
        :param dim: dimension of vectors
        :return: (array) - each i-th element is an AUROC score for the i-th class
        """
        outAUROC = []

        for i in range(dim):

            try:
                outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
            except (RuntimeError, TypeError, NameError, ValueError):
                outAUROC.append(-1.0)

        return outAUROC

    # -------------------------------------------------------------------------------

    def computeAccuracyStatistics(dataGT, dataPRED, dim, threshold=0.5):
        """
        Calculate accuracy statistics - true positive, false positive, true negative, false negative
        :param dataGT: numpy array of ground truth vectors
        :param dataPRED: numpy array of predicted vectors
        :param dim: dimension of vectors
        :param threshold:
        :return: (array) - array of scores TP, FP, TN, FN
        """

        # ---- True positive: 1 predicted as 1
        outTruePositive = [0] * dim

        # ---- False positive: 0 predicted as 1
        outFalsePositive = [0] * dim

        # ---- True negative: 0 predicted as 0
        outTrueNegative = [0] * dim

        # ---- False negative: 1 predicted as 0
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
        """
        Calculates accuracy score for each class
        :param stTP: vector of true positive values for each class
        :param stFP: vector of false positive values for each class
        :param stTN: vector of true negative values for each class
        :param stFN: vector of false negative values for each class
        :return: (array) - accuracy for each class
        """

        dim = len(stTP)

        outAccuracy = [0] * dim

        for i in range(0, dim):

            total = stTP[i] + stFP[i] + stTN[i] + stFN[i]

            if total != 0:
                outAccuracy[i] = (stTP[i] + stTN[i]) / total
            else:
                outAccuracy[i] = -1

        return outAccuracy

    # --------------------------------------------------------------------------------

    def computeAccuracyFscore (stTP, stFP, stTN, stFN):
        """
        Calculates per-class f-score
        :param stTP: vector of true positive values for each class
        :param stFP: vector of false positive values for each class
        :param stTN: vector of true negative values for each class
        :param stFN: vector of false negative values for each class
        :return: (array) of f-scores for each class
        """
        dim = len(stTP)

        outFscore = [0] * dim

        for i in range(0, dim):

            norm = 2*stTP[i] + stFP[i] + stFN[i]

            if norm != 0:
                outFscore[i] = (2 * stTP[i]) / norm
            else:
                outFscore[i] = -1

        return outFscore