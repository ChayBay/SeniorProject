import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

trainData = np.load('trainingData.npy', allow_pickle=True)
print(len(trainData))
df = pd.DataFrame(trainData)
print(df)

up = []
left = []
right = []
down = []
ul = []
ur = []
dl = []
dr = []


shuffle(trainData)

for data in trainData:
    screen = data[0]
    choice = data[1]
    print(choice)

    if choice == [1,0,0,
                  0,0,
                  0,0,0]:
        for i in range(5):
            ul.append([screen,choice])
    elif choice == [0,1,0,
                    0,0,
                    0,0,0]:
        for i in range(5):
            up.append([screen,choice])
    elif choice == [0,0,1,
                    0,0,
                    0,0,0]:
        for i in range(5):
            ur.append([screen,choice])
    elif choice == [0,0,0,
                    1,0,
                    0,0,0]:
        for i in range(5):
            left.append([screen,choice])
    elif choice == [0,0,0,
                    0,1,
                    0,0,0]:
        for i in range(5):
            right.append([screen,choice])
    elif choice == [0,0,0,
                    0,0,
                    1,0,0]:
        for i in range(5):
            dl.append([screen,choice])
    elif choice == [0,0,0,
                    0,0,
                    0,1,0]:
        for i in range(5):
            down.append([screen,choice])
    elif choice == [0,0,0,
                    0,0,
                    0,0,1]:
        for i in range(5):
            dr.append([screen,choice])
    else:
        print("ERROR")

shuffle(up)
shuffle(left)
shuffle(down)
shuffle(right)
shuffle(ul)
shuffle(ur)
shuffle(dl)
shuffle(dr)

'''
print("up")
print(len(up))
print("left")
print(len(left))
print("right")
print(len(right))
print("down")
print(len(down))
print("ul")
print(len(ul))
print("ur")
print(len(ur))
print("dl")
print(len(dl))
print("dr")
print(len(dr))
'''

#this makes sure that all the arrays have the same length of data
right = right[:len(left)][:len(ur)][:len(dl)][:len(ul)]
down = down[:len(left)][:len(ur)][:len(dl)][:len(ul)]
dr = dr[:len(left)][:len(ur)][:len(dl)][:len(ul)]
up = up[:len(left)][:len(ur)][:len(dl)][:len(ul)]
left = left[:len(right)]
ur = ur[:len(right)]
dl = dl[:len(right)]
ul = ul[:len(right)]


finalData = up+left+right+down+ul+ur+dl+dr
shuffle(finalData)
print(len(finalData))
np.save("balancedData.npy", np.array(finalData, dtype = object))
    
