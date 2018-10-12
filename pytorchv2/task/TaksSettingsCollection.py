import json
from pprint import pprint
import sys

sys.path.append('../')
from task.TaskSettings import TaskSettings

class TaskSettingsCollection():

    # --------------------------------------------------------------------------------
    TASK_LIST = "tasklist"

    TASK_TYPE = "tasktype"

    TASK_DATABASE = "database"
    TASK_DATABASE_TRAIN = "dataset_train"
    TASK_DATABASE_VALIDATE = "dataset_validate"
    TASK_DATABASE_TEST = "dataset_test"

    DATAGEN_TYPE = "datagen"

    TASK_OUTPUT_LOG = "output_log"
    TASK_OUTPUT_MODEL = "output_model"
    TASK_OUTPUT_ACCURACY = "output_accuracy"

    TASK_NN_MODEL = "network"
    TASK_NN_ISTRAINED = "network_istrained"
    TASK_NN_CLASSCOUNT = "network_classcount"
    TASK_NN_ACTIVATION = "activation"

    TASK_TRNSFRM_TRAIN = "trnsfrm_train"
    TASK_TRNSFRM_TRAIN_PARAM = "trnsfrm_train_param"

    TASK_TRNSFRM_VALIDATE = "trnsfrm_validate"
    TASK_TRNSFRM_VALIDATE_PARAM = "trnsfrm_validate_param"

    TASK_TRNSFRM_TEST = "trnsfrm_test"
    TASK_TRNSFRM_TEST_PARAM = "trnsfrm_test_param"

    TASK_LOSS = "loss"
    TASK_EPOCH = "epoch"
    TASK_LRATE = "lrate"
    TASK_BATCH = "batch"
    # --------------------------------------------------------------------------------

    def __init__(self):
        self.taskcollection = []
        self.taskcount = 0

    # --------------------------------------------------------------------------------

    def load(self, path):
        """
        Load tasks from a JSON file
        :param path: path to the JSON file that contains task settings data
        """
        with open(path) as f:
            data = json.load(f)

        self.taskcount = len(data[TaskSettingsCollection.TASK_LIST])

        for i in range(0, self.taskcount):

            task = TaskSettings()

            taskdata = data[TaskSettingsCollection.TASK_LIST][i]

            for key, value in taskdata.items():

                if key == TaskSettingsCollection.TASK_DATABASE: task.setPathInRoot(value)
                if key == TaskSettingsCollection.TASK_DATABASE_TRAIN: task.setPathInTrain(value)
                if key == TaskSettingsCollection.TASK_DATABASE_VALIDATE: task.setPathInValidate(value)
                if key == TaskSettingsCollection.TASK_DATABASE_TEST: task.setPathInTest(value)

                if key == TaskSettingsCollection.TASK_OUTPUT_ACCURACY: task.setPathOutAccuracy(value)
                if key == TaskSettingsCollection.TASK_OUTPUT_LOG: task.setPathOutLog(value)
                if key == TaskSettingsCollection.TASK_OUTPUT_MODEL: task.setPathOutModel(value)

                if key == TaskSettingsCollection.TASK_NN_ACTIVATION: task.setActivationType(TaskSettings.ACTIVATION_GETID(value))
                if key == TaskSettingsCollection.TASK_NN_CLASSCOUNT: task.setNetworkClassCount(value)
                if key == TaskSettingsCollection.TASK_NN_ISTRAINED: task.setNetworkIsTrained(value)
                if key == TaskSettingsCollection.TASK_NN_MODEL: task.setNetworkType(TaskSettings.NN_GETID(value))

                if key == TaskSettingsCollection.TASK_TRNSFRM_TRAIN: task.setTransformSequenceTrain(value, True)
                if key == TaskSettingsCollection.TASK_TRNSFRM_TRAIN_PARAM: task.setTransformSequenceTrainParameters(value)
                if key == TaskSettingsCollection.TASK_TRNSFRM_VALIDATE: task.setTransformSequenceValidate(value, True)
                if key == TaskSettingsCollection.TASK_TRNSFRM_VALIDATE_PARAM: task.setTransformSequenceValidateParameters(value)
                if key == TaskSettingsCollection.TASK_TRNSFRM_TEST: task.setTransformSequenceTest(value, True)
                if key == TaskSettingsCollection.TASK_TRNSFRM_TEST_PARAM: task.setTransformSequenceTestParameters(value)

                if key == TaskSettingsCollection.TASK_LOSS: task.setLoss(TaskSettings.LOSS_GETID(value))
                if key == TaskSettingsCollection.TASK_EPOCH: task.setEpoch(value)
                if key == TaskSettingsCollection.TASK_LRATE: task.setLearningRate(value)
                if key == TaskSettingsCollection.TASK_BATCH: task.setBatchSize(value)

                if key == TaskSettingsCollection.TASK_TYPE: task.setTaskType(value)
                if key == TaskSettingsCollection.DATAGEN_TYPE: task.setDatagenType(TaskSettings.DATAGEN_GETID(value))

            self.taskcollection.append(task)

    def gettask(self, index):
        """
        Get task by its index
        :param index: index of the desired task
        :return: (TaskSettings) - task settings
        """
        if(index >= 0) and (index < self.taskcount):
            return self.taskcollection[index]

        return None