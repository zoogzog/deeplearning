from datagen.DatagenClassification import DatagenClassification
from datagen.DatagenOversampling import DatagenOversampling

from algorithm.AlgorithmStyleTransfer import AlgorithmStyleTransfer
from algorithm.AlgorithmClassification import AlgorithmClassification


from task.TaskSettings import TaskSettings
from task.TaksSettingsCollection import TaskSettingsCollection
from task.TaskClassification import TaskClassification


#pathInput = "test4.jpg"
#pathStyle = "tt.jpg"

#algorithm = AlgorithmStyleTransfer()
#output = algorithm.run(pathInput, pathStyle, pathInput, style_weight=8000, content_weight=1, num_steps=200)
#output.save("test.jpg")



collection = TaskSettingsCollection()
collection.load('task.json')
task = collection.gettask(0)

runner = TaskClassification()
runner.launch(task)

