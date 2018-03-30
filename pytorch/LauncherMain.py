import os

#----------------------------------------------------------------------------------------
TASKLIST= [
    #
    [
        #--- This is a path to the database, that contains raw images
        "../database",
                      
        #--- This is a path to the file that contains TRAINING dataset
        #--- Each line in this file is a PATH to image file, followed by  
        #--- target vector
        "../database/dataset1.txt",
        
        #---- This is a path to the file that contains VALIDATION dataset
        "../database/dataset1.txt",
        
        #---- This is a path to the file that contains TEST dataset
        "../database/dataset1.txt",
        
        #---- In the folder specified here trained models will be saved
        "../",
        
        #---- In the folder specified here logs of the training process will be saved
        "../",
        
        #---- In the folder specified here accuracy of the trained network on the TEST set 
        #---- will be saved
        "../",
        
        #----- NN MODEL: choose one [alex, inception, vgg, res50, res101, dense121, dense169, dense201]
        "dense121", 
        
        #---- If "true" a pre-trained version is used, if "false" then training from scratch
        "true", 
        
        #---- Dimension of the output vector (1 for binary classification)
        1, 
        
        #---- Loss function. Currently supported 
        #---- BINARY CROSS ENTROPY - "bce"
        #---- WEIGHTED BINARY CROSS ENTROPY (for unbalanced datasets) - "wbce"
        "bce", 
        
        #---- ACTIVATION FUNCTION: "sigmoid", "softmax", "none"
        "sigmoid", 
        
        #---- Image transformation scale factor (see TaskManager.py)
        256, 
        
        #---- Image transformation crop factor (see TaskManager.py)
        224, 
        
        #---- Batch size
        16, 
        
        #---- Epochs
        50
    ],

    [
        "../database",
        "../database/dataset1.txt",
        "../database/dataset1.txt",
        "../database/dataset1.txt",
        "../",
        "../",
        "../",
        "res50", "true", 1, "bce", "sigmoid", 256, 224, 16, 50
    ]
]
#----------------------------------------------------------------------------------------

TASKARGS = ["-r", "-t", "-v", "-e", "-m", "-l", "-a", "-n", "-i", "-d", "-f", "-s", "-q", "-p", "-b", "-x"]


for taskID in range(0, len(TASKLIST)):

    command = "python LauncherTask.py "

    print("----------------------------------------------------------------")
    print("Launching task: " + str(taskID) + " | " + str(len(TASKLIST)))

    for argID in range (0, len(TASKARGS)):
        print (str(TASKLIST[taskID][argID]))
        command += TASKARGS[argID] + " " + str(TASKLIST[taskID][argID]) + " "

    os.system(command)
    print("----------------------------------------------------------------")