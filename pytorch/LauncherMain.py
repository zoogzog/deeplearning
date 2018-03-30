import os

#----------------------------------------------------------------------------------------
TASKLIST= [
    #[alex, inception, vgg, res50, res101, dense121, dense169, dense201]
    [
        "../database",
        "../database/dataset1.txt",
        "../database/dataset1.txt",
        "../database/dataset1.txt",
        "../",
        "../",
        "../",
        "dense121", "true", 1, "bce", "sigmoid", 256, 224, 16, 50
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