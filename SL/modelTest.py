import numpy as np
import pandas as pd
import time
import os
import random
import playerSeminar
from alexNet import alex
from PIL import Image

WIDTH = 9
HEIGHT = 9
LR = 1e-3
EPOCH = 10
modelName = './SL-collect-{}-{}-{}-epochs.model'.format(LR, 'SL', EPOCH)
#'C:\Users\Edene\Desktop\Seminar\SL\log\SL-collect-0.001-SL-10-epochs.model\events.out.tfevents.1668135489.PANZER'

model = alex(WIDTH, HEIGHT, LR)
model.load('./SL-collect-0.001-SL-10-epochs.model')


roomTemplate = [
        ['X','▲','▲','▲','▲','▲','▲','▲','X'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['◄','.','.','.','.','.','.','.','►'],
        ['X','▼','▼','▼','▼','▼','▼','▼','X']
        ]

p = playerSeminar.playerObj(4,4)
collect = ["W","I","N","N","E","R"]
savePoint = "$"
touchables = ["W","I","N","E","R","$"]
fire = []
nabbed = []
mapItems = []
firstStep = True
block = None
dynamic4 = 0


# clicking game button though PandP over and over again overflows hud due to threading issue
def load(pl, user):
    global block
    pl.x = 4
    pl.y = 4
    block = None
    f = open("playerSaves/"+user+"_collect.txt", "r")
    hm = f.read(1)
    try:
        hm = int(hm)
    except:
        hm = 0
        
    hm = int(hm)
    if hm in range(6):
        pl.prog = hm
        p.prog = hm

    i=0
    for item in range(int(pl.prog)):
        nabbed.append(str(collect[i]))
        i+=1
    f.close()
    return nabbed
    
#saving disabled for testing
def save(pl, user):
    #o = open("playerSaves/"+user+"_collect.txt", "w")
    #o.write(str(pl.prog))
    #o.close()
    print("\t\tCOLLECTION SAVED")

# where to keep the Crucial Coordinates when map is reprinted
def exclude(n):
    r = None
    while r == n or r == None:
        r = random.randint(1,7)
    return r

def chooseTile(pl):
    global mapItems
    r = random.randint(0,1)
    xSave = random.randint(1,7)
    ySave = random.randint(1,7)

    if r == 0:
        xColl = random.randint(1,7)
        yColl = random.randint(1,7)
        if xColl == xSave and yColl == ySave:
            axis = random.randint(0,1)
            if axis == 0:
                xColl = exclude(xSave)
            if axis == 1:
                yColl = exclude(ySave)
    else:
        xColl = None
        yColl = None

    mapItems = [xSave, ySave, xColl, yColl]
    return xSave, ySave, xColl, yColl

# goals for editing colect will be to create hazards or maze generation into randomizeRoom
def randomizeRoom(pl, room, newS, roll):
    global fire
    global hzd
    global mapItems
    global firstStep
    
    if  newS == True:
        firstStep = True
        roll = False
        rchoom = [i[:] for i in room]
        hzd, fire = hazards(pl)
        saveX, saveY, colX, colY = chooseTile(pl)

    if roll == True:
        firstStep = False
        rchoom = [i[:] for i in room]
        hazardRoller(pl, hzd, fire)
        saveX = mapItems[0]
        saveY = mapItems[1]
        colX = mapItems[2]
        colY = mapItems[3]
        
    if newS is False and roll is False:
        firstStep = True
        rchoom = [i[:] for i in roomTemplate]
        hzd, fire = hazards(pl)
        saveX, saveY, colX, colY = chooseTile(pl)

    posx = 0
    posy = 0
    checkCollect = False
    checkSave = False

    for y in rchoom:
        for x in y:
            if x == ".":
                    
                if posy == saveY and posx == saveX:
                    rchoom[posy][posx] = savePoint
                    posS = rchoom[posy][posx]
                    checkSave = True
                    
                if posy == colY and posx == colX:
                    rchoom[posy][posx] = collect[int(pl.prog)]
                    posC = rchoom[posy][posx]
                    checkCollect = True

                if [posx, posy] in fire and [posx,posy] != [colX,colY] and  [posx,posy] != [saveX,saveY]:
                    rchoom[posy][posx] = "F"

                if rchoom[posy][posx] == ".":
                    rchoom[posy][posx] = " "
                    
            posx+=1
        posy+=1
        posx=0
        
    return rchoom, hzd

def hazards(pl):
    hazardSet = random.randint(1,4)
    
    if hazardSet == 1:
        fire = [
            [1,2],[2,2],[3,2],[4,2],
            [4,-2],[5,-2],[6,-2],[7,-2],
           ]
    
    if hazardSet == 2:
        fire = [
            [random.randint(1,7),1],[random.randint(1,7)+3,2],
            [random.randint(1,7)+5,3],[random.randint(1,7)+6,4],
            [random.randint(1,7)+2,5],[random.randint(1,7)+1,6],
            [random.randint(1,7)+4,7]
            ]

    if hazardSet == 3:
       fire = statAndDynFire(pl)

    if hazardSet == 4:
       fire = statAndDynFire(pl)
    
    return hazardSet, fire


#moving logic for fire with fire array populated based on random screen choice
#1 is rolling logs rolling down vert
#2 is randomly placed logs rolling horiz
#3 is just randomly placed fire on ground static

def hazardRoller(pl, hzd, fire):
    index = 0
    global dynamic4
    if hzd == 1:
        for i in fire:
            fire[index]=[fire[index][0],fire[index][1]+1]
            if fire[index][1]>7:
                fire[index][1] = -(fire[index][1]-7)
            index+=1
    
    if hzd == 2:
        for i in fire:
            fire[index]=[fire[index][0]+1,fire[index][1]]
            if fire[index][0]>7:
                fire[index][0] = -(fire[index][0]-7)
            index+=1

    if hzd == 4:
        dynamic4+=1
        if dynamic4 > 4:
            statAndDynFire(pl)
            dynamic4 = 0
            
def statAndDynFire(pl):
    global fire
    fire = []
    index = 0
    px = pl.x
    py = pl.y
    avoid = [[px-1,py-1],[px,py-1],[px+1,py-1],
             [px-1,py],[px,py],[px+1,py],
             [px-1,py+1],[px,py+1],[px+1,py+1]]
    
    while len(fire)<8:
        x = random.randint(1,7)
        y = random.randint(1,7)
        if [x,y] not in avoid:
            fireball = [x,y]
            fire.append(fireball)
        index+=1
        
    return fire
            
def makeBounds(way, room, pl, roll):
    bounds = [i[:] for i in roomTemplate]
    global block
    
    if way in collect:
        newScr = False
        roll = True
    
    if way == "":
        newScr = False
        roll = True
    
    if way == "▲":
        newScr = True
        block = "▼"
        pl.x = 4
        pl.y = 7
        
    if way == "◄":
        newScr = True
        block = "►"
        pl.x = 7
        pl.y = 4
        
    if way == "▼":
        newScr = True
        block = "▲"
        pl.x = 4
        pl.y = 1 
        
    if way == "►":
        newScr = True
        block = "◄"
        pl.x = 1
        pl.y = 4

    posx = 0
    posy = 0
    for y in bounds:
        for x in y:
            if x == block:
                bounds[posy][posx] = "X"
            posx+=1
        posy+=1
        posx=0
            
    bounds, hzd = randomizeRoom(pl, bounds, newScr, roll)
    return bounds, hzd

def step(pl, move):
        pl.prevX = pl.x
        pl.prevY = pl.y
        running = True

        here = 0
        index = 0
        for i in move:
            print(i)
            if i == 1:
                here = index
                break
            index+=1
            

        if move == [1,0,0,0,0,0,0,0]:
            print("up left")
            pl.y -= 1
            pl.x -= 1
        if move == [0,1,0,0,0,0,0,0]:
            print("up")
            pl.y -= 1
        if move == [0,0,1,0,0,0,0,0]:
            print("up right")
            pl.y -= 1
            pl.x += 1
        if move == [0,0,0,1,0,0,0,0]:
            print("left")
            pl.x -= 1
        if move == [0,0,0,0,1,0,0,0]:
            print("right")
            pl.x += 1
        if move == [0,0,0,0,0,1,0,0]:
            print("down left")
            pl.y += 1
            pl.x -= 1
        if move == [0,0,0,0,0,0,1,0]:
            print("down")
            pl.y += 1
        if move == [0,0,0,0,0,0,0,1]:
            print("down right")
            pl.y += 1
            pl.x += 1
        
        return running

def checkSpot(pl, room, user, hzd):
    movePenalty = -1
    blockPenalty = -3
    firePenalty = -300
    screenReward = 3
    saveReward = 1
    colectReward = 50
    global mapItems
    newRoom = [i[:] for i in room]
    lost = False
    roll = False
    firePop(pl, hzd)

    if room[pl.y][pl.x] == " ":
        roll = True
        reward = movePenalty
        newRoom, hzd = makeBounds("", room, pl, roll)
        
    if room[pl.y][pl.x] ==  "F":
        reward = firePenalty
        lost = True

    if room[pl.y][pl.x] == "▲":
        roll = False
        reward = screenReward
        newRoom, hzd = makeBounds("▲", room, pl, roll)
        
    if room[pl.y][pl.x] == "◄":
        roll = False
        reward = screenReward
        newRoom, hzd = makeBounds("◄", room, pl, roll)
        
    if room[pl.y][pl.x] == "▼":
        roll = False
        reward = screenReward
        newRoom, hzd = makeBounds("▼", room, pl, roll)
        
    if room[pl.y][pl.x] == "►":
        roll = False
        reward = screenReward
        newRoom, hzd = makeBounds("►", room, pl, roll)

    if room[pl.y][pl.x] ==  "$":
        roll = True
        save(pl, user)
        reward = saveReward
        newRoom, hzd = makeBounds("", room, pl, roll)
        
    if room[pl.y][pl.x] == collect[pl.prog]:
        roll = True
        newRoom, hzd = makeBounds("", room, pl, roll)
        mapItems[2] = " "
        mapItems[3] = " "
        colStr = "YOU COLLECTED "+str(collect[pl.prog])
        colStr = colStr.center(43)
        reward = colectReward
        print(colStr)
        nabbed.append(collect[pl.prog])
        pl.prog+=1
        room[pl.x][pl.y] == " "
    
    if room[pl.y][pl.x] == "X":
        reward = blockPenalty
        pl.x = pl.prevX
        pl.y = pl.prevY
        
    else:
        reward = -1

    return newRoom, lost, reward

def roomPrint(pl, room):
    os.system("cls")
    global fire
    lets = ""
    hud = "COLLECTION: < "
    for i in nabbed:
        lets+=i+" "
    print("\n"+hud+lets+">\n")
    
    posx = 0
    posy = 0
    for y in room:
        for x in y:
            if [pl.x,pl.y] in fire and pl.x == posx and pl.y == posy :
                print("!",end="    ")
            elif pl.x == posx and pl.y == posy:
                print("P",end="    ")
            else:
                print(x,end="    ")
            posx+=1
        posy+=1
        posx=0
        print("\n")
    #print("x = "+str(pl.x)+"y = "+str(pl.y))
    #print("collect = "+str(pl.prog))

def placeP(pl, room):
    #get the room and then take each coordinate and place it in the env
    #P
    global collectSpawned
    global p
    posx = 0
    posy = 0
    for y in room:
        for x in y:
            if pl.x == posx and pl.y == posy:
                room[pl.x][pl.y] = "P"
            posx+=1
        posy+=1
        posx=0

    return room

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

def firePop(pl, hzd):
    global firstStep
    global fire
    global mapItems
    colectPair = [mapItems[2],mapItems[3]]
    savePair = [mapItems[0],mapItems[1]]

    index = 0
    for i in fire:
                
        if i == colectPair:
            fire.pop(index)

        if i == savePair:
            fire.pop(index)
                    
        if [pl.x,pl.y] in fire and firstStep == True:
            fire.pop(index)
                
        index+=1

def activeLoss(pl, hzd, room, reward):
    global firstStep
    global fire
    global mapItems
    done = False
    colectPair = [mapItems[2],mapItems[3]]
    savePair = [mapItems[0],mapItems[1]]
        
    if [pl.x,pl.y] in fire and [pl.x,pl.y] != colectPair and [pl.x,pl.y] != savePair and firstStep == False:
        reward = -300
        done = True
    return done, reward

def main(arg, user):
    global nabbed
    rewardCollected = 0
    rewardCount = 0
    os.system("cls")
    if arg == 1:
        load(p,user)
    random.seed()
    hzd = 0
    room, hzd = randomizeRoom(p, roomTemplate, False, False)
    
    running = True
    lost1 = False
    lost2 = False

    while running:

        roomPrint(p, room)
        oneS = placeP(p, room)
        twoS = getScreen(oneS)
        screen = np.array(twoS)
        
        prediction = model.predict([screen])[0]
        moves = list(np.around(prediction))
        print(moves)

        running = step(p, moves)
        room, lost1, reward = checkSpot(p, room, user, hzd)

        if reward == 50:
            rewardCollected += 1
            
        if reward == -1:
            lost2, reward = activeLoss(p, hzd, room, reward)

        rewardCount += reward

        if lost1 == True:
            os.system("cls")
            print("\n\n")
            print("You walked into fire!!!".center(43))
            roomPrint(p, room)
            print("You walked into fire!!!".center(43))
            print("\n\n")
            with open('rewardData.txt','w') as f:
                f.write('times collected: '+str(rewardCollected))
                f.write('\n reward: '+str(rewardCount))
                f.close()
            break
            
        if lost2 == True:
            os.system("cls")
            print("\n\n")
            print("Fire fell on you!!!".center(43))
            roomPrint(p, room)
            print("Fire fell on you!!!".center(43))
            print("\n\n")
            with open('rewardData.txt','w') as f:
                f.write('times collected: '+str(rewardCollected))
                f.write('\n reward: '+str(rewardCount))
                f.close()
            break
            
        if running == False:
            os.system("cls")
            break
        if p.prog == 6:
            print("Size of training data: "+str(len(trainingData)))
            np.save(trainFile, np.array(trainingData, dtype = object))
            os.system("cls")
            break
    
    if p.prog == 6:
        print("\n\n")
        print("You Win!!!".center(43))
        print("\n\n")
        with open('rewardData.txt','w') as f:
                f.write('times collected: '+str(rewardCollected))
                f.write('\n reward: '+str(rewardCount))
                f.close()
    else:
        print("\n\n")
        print("Thanks for Playing!!!".center(43))
        print("\n\n")
        
    os.system("pause")

main(1,"SL")
