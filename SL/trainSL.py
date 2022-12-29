from alexNet import alex
import numpy as np

WIDTH = 9
HEIGHT = 9
LR = 1e-3
EPOCH = 20
modelName = 'SL-collect-{}-{}-{}-epochs.model'.format(LR, 'SL', EPOCH)

model = alex(WIDTH, HEIGHT, LR)
trainData = np.load('finalData.npy', allow_pickle=True)
train = trainData[:-500]
test = trainData[-500:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
Y = [i[1] for i in train]

testX = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
testY = [i[1] for i in test]

model.fit(X, Y, n_epoch=EPOCH,
          validation_set = (testX, testY),
          snapshot_step=500, show_metric=True, run_id=modelName)

model.save(modelName)

# tensorboard --logdir=foo:C:\Users\Edene\Desktop\Seminar\SL\log
