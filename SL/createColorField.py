import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
from PIL import Image

trainData = np.load('balancedData.npy', allow_pickle=True)
print(len(trainData))
df = pd.DataFrame(trainData)
print(df)


    
def getScreen(room):
    pConvert = 1
    colConvert = 2
    saveConvert = 3
    boundConvert = 4
    xConvert = 5
    fireConvert = 6

          #BLUE GREEN RED
    d = {1:(255, 0, 255),#PLAYER PURPLE
         2:(255, 0, 0),  #COL BLUE
         3:(0, 255, 0),  #SAVE GREEN
         4:(0, 255, 255),#BOUND YELLOW
         5:(0, 150, 255),#X ORANGE
         6:(0, 0, 255)}  #FIRE RED
    
    collect = ["W","I","N","N","E","R"]
    bounds = ['▲','◄','►','▼']
    env = np.zeros((9,9,3), dtype = np.uint8)
    
    posx = 0
    posy = 0
    for j in room:
        for i in j:
            if i == "F":
                env[posy][posx] = d[fireConvert]
            if i in bounds:
                env[posy][posx] = d[boundConvert]
            if i == "X":
                env[posy][posx] = d[xConvert]
            if i == "$":
                env[posy][posx] = d[saveConvert]
            if i in collect:
                env[posy][posx] = d[colConvert]
            if i == "P":
                env[posy][posx] = d[pConvert]
            posx+=1
        posy+=1
        posx=0
    
    screenPic = Image.fromarray(env, 'RGB')
    return screenPic

for data in trainData:
    colorDawg = data[0]
    colord = np.array(getScreen(colorDawg))
    data[0] = colord
    
df2 = pd.DataFrame(trainData)
print(df2)

np.save("finalData.npy", np.array(trainData, dtype = object))

