import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as backend
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import time
from tqdm import tqdm
import cv2
import os
from PIL import Image
import random
import playerSeminar

LOAD_MODEL = None

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'RL_DQN'
MIN_REWARD = -300  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when training multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class	
class ModifiedTensorBoard(TensorBoard):	
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)	
    def __init__(self, **kwargs):	
        super().__init__(**kwargs)	
        self.step = 1	
        self.writer = tf.summary.create_file_writer(self.log_dir)	
        self._log_write_dir = self.log_dir	
    # Overriding this method to stop creating default log writer	
    def set_model(self, model):	
        self.model = model	
        self._train_dir = os.path.join(self._log_write_dir, 'train')	
        self._train_step = self.model._train_counter	
        self._val_dir = os.path.join(self._log_write_dir, 'validation')	
        self._val_step = self.model._test_counter	
        self._should_write_train_graph = False	
    # Overrided, saves logs with our step number	
    # (otherwise every .fit() will start writing from 0th step)	
    def on_epoch_end(self, epoch, logs=None):	
        self.update_stats(**logs)	
    # Overrided	
    # We train for one batch only, no need to save anything at epoch end	
    def on_batch_end(self, batch, logs=None):	
        pass	
    # Overrided, so won't close writer	
    def on_train_end(self, _):	
        pass	
    # Custom method for saving own metrics	
    # Creates writer, writes custom metrics and closes writer	
    def update_stats(self, **stats):	
        with self.writer.as_default():	
            for key, value in stats.items():	
                tf.summary.scalar(key, value, step = self.step)	
                self.writer.flush()

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")
        else:
            model = Sequential()
            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), padding = "same"))
            model.add(Dropout(0.2))
            
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))
            
            model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is alearning_rateeady saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)
        X = []
        y = []
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape)/255)[0]

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
    return
    #o = open("playerSaves/"+user+"_collect.txt", "w")
    #o.write(str(pl.prog))
    #o.close()
    #print("\t\tCOLLECTION SAVED")

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
    
    # have to place the P in the array for the agent to see where it is
    # currently the P is just printed
    # MUST BE ABLE TO SEND THE DATA WITH THE P ON THE SCREEN TO THE AGENT
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

class makeEnv():

    SIZE = 9
    movePenalty = -1
    blockPenalty = -3
    firePenalty = -300
    screenReward = 3
    saveReward = 1
    colectReward = 50
    boundList = []
    fireList = []
    xList = []
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    #may have to remove the save point after a save
    ACTION_SPACE_SIZE = 9
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
         6:(0, 0, 255)}#FIRE RED

    def getScreen(self, pl):
        global collectSpawned
        global saveSpawned
        env = np.zeros((9,9,3), dtype = np.uint8)
        posx = 0
        posy = 0
        for i in self.fireList:
            env[i.x][i.y] = self.d[self.fireConvert]
        for i in self.boundList:
            env[i.x][i.y] = self.d[self.boundConvert]
        for i in self.xList:
            env[i.x][i.y] = self.d[self.xConvert]
        if saveSpawned == True:
            env[self.save.x][self.save.y] = self.d[self.saveConvert]
        if collectSpawned == True:
            env[self.collect.x][self.collect.y] = self.d[self.colConvert]
        env[self.player.x][self.player.y] = self.d[self.pConvert]
        screenPic = Image.fromarray(env, 'RGB')
        return screenPic

    def reset(self, pl, room):
        #get the room and then take each coordinate and place it in the env
        #P
        global collectSpawned
        global saveSpawned
        global p
        global collect
        bounds = ['▲','◄','►','▼']
        collectSpawned = False
        saveSpawned = False
        self.boundList = []
        self.fireList = []
        self.xList = []
        posx = 0
        posy = 0
        for y in room:
            for x in y:
                if room[posx][posy] in collect:
                    collectSpawned = True
                    self.collect = playerSeminar.playerObj(posx,posy)
                if room[posx][posy] == "F":
                    self.fireList.append(playerSeminar.playerObj(posx,posy))
                if room[posx][posy] == "$":
                    saveSpawned = True
                    self.save = playerSeminar.playerObj(posx,posy)
                if room[posx][posy] in bounds:
                    self.boundList.append(playerSeminar.playerObj(posx,posy))
                if room[posx][posy] == "X":
                    self.xList.append(playerSeminar.playerObj(posx,posy))
                if pl.x == posx and pl.y == posy:
                    self.player = pl
                    room[posx][posy] = "P"
                posx+=1
            posy+=1
            posx=0
        
        observation = np.array(self.getScreen(pl))
           
        return observation, room

    def checkSpot(self, pl, room, user, hzd):
        global mapItems
        newRoom = [i[:] for i in room]
        lost = False
        roll = False
        firePop(pl, hzd)

        reward = self.movePenalty
        
        if room[pl.y][pl.x] == " ":
            roll = True
            #rewarding for agent
            reward = self.movePenalty
            newRoom, hzd = makeBounds("", room, pl, roll)
        
        if room[pl.y][pl.x] ==  "F":
            #rewarding for agent
            reward = self.firePenalty
            lost = True
        
        if room[pl.y][pl.x] == "▲":
            roll = False
            #rewarding for agent
            reward = self.screenReward
            newRoom, hzd = makeBounds("▲", room, pl, roll)
            
        if room[pl.y][pl.x] == "◄":
            roll = False
            #rewarding for agent
            reward = self.screenReward
            newRoom, hzd = makeBounds("◄", room, pl, roll)
            
        if room[pl.y][pl.x] == "▼":
            roll = False
            #rewarding for agent
            reward = self.screenReward
            newRoom, hzd = makeBounds("▼", room, pl, roll)
            
        if room[pl.y][pl.x] == "►":
            roll = False
            #rewarding for agent
            reward = self.screenReward
            newRoom, hzd = makeBounds("►", room, pl, roll)

        if room[pl.y][pl.x] ==  "$":
            roll = True
            save(pl, user)
            mapItems[0] = " "
            mapItems[1] = " "
            #rewarding for agent
            reward = self.saveReward
            room[pl.x][pl.y] == " "
            newRoom, hzd = makeBounds("", room, pl, roll)

        if room[pl.y][pl.x] == collect[pl.prog]:
            roll = True
            mapItems[2] = " "
            mapItems[3] = " "
            colStr = "YOU COLLECTED "+str(collect[pl.prog])
            colStr = colStr.center(43)
            #rewarding for agent
            reward = self.colectReward
            print(colStr)
            nabbed.append(collect[pl.prog])
            pl.prog+=1
            room[pl.x][pl.y] == " "
            newRoom, hzd = makeBounds("", room, pl, roll)

        if room[pl.y][pl.x] == "X":
            #rewarding for agent
            reward = self.blockPenalty
            pl.x = pl.prevX
            pl.y = pl.prevY

        #maybe handle this if collect doesnt spawn

        new_observation = np.array(self.getScreen(pl))

        #how to handle new_observation and done
        #if player got food or is player hit fire or if the step counter
        #is too high end then end the episode
        
        return newRoom, lost, new_observation, reward
    
    def step(self, pl, move):
        self.episode_step += 1
        pl.prevX = pl.x
        pl.prevY = pl.y
        running = True
        
        if move == 0:
            #print("up")
            pl.y -= 1
        if move == 1:
            #print("left")
            pl.x -= 1
        if move == 2:
            #print("down")
            pl.y += 1
        if move == 3:
            #print("right")
            pl.x += 1
        if move == 4:
            #print("up right")
            pl.y -= 1
            pl.x += 1
        if move == 5:
            #print("up left")
            pl.y -= 1
            pl.x -= 1
        if move == 6:
            #print("down left")
            pl.y += 1
            pl.x -= 1
        if move == 7:
            #print("down right")
            pl.y += 1
            pl.x += 1
        if move == 8:
            #print("no move")
            pl.y += 0
            pl.x += 0
        if move == "q":
            running = False

        return running

    def activeLoss(self, pl, reward):
        global firstStep
        global fire
        global mapItems
        colectPair = [mapItems[2],mapItems[3]]
        savePair = [mapItems[0],mapItems[1]]
        done = False
            
        if [pl.x,pl.y] in fire and [pl.x,pl.y] != colectPair and [pl.x,pl.y] != savePair and firstStep == False:
            reward = self.firePenalty
            done = True
        return done, reward
    
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
    
env = makeEnv()
gamer = DQNAgent()

def main(arg, user):
    global epsilon
    global nabbed
    
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #os.system("cls")
        if arg == 1:
            load(p,user)
        hzd = 0
        nabbed = []
        room, hzd = randomizeRoom(p, roomTemplate, False, False)
        env.episode_step = 0
        running = True
        lost1 = False
        lost2 = False
        
        # Update tensorboard step every episode
        gamer.tensorboard.step = episode
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        epiStep = 1
        # Reset environment and get initial state
        
        # Reset flag and start iterating until episode ends
        done = False
        while running:

            roomPrint(p, room) # WE WILL NEED TO HAVE THE AGENT LOOK AT THE GRID AND MAKE AN ACTION HERE\
            current_state, room = env.reset(p, room)

            action = None
            #agent is getting the action for the state
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(gamer.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            
            running = env.step(p, action) # THIS IS WHERE THE AGENT CHOOSES AN ACTION

            #needs to return new_state reward and done
            #done is true when all collectables are collected or fire is hit
            #done get set to true when the game is completed or a fireball is hit
            #each run will be calculated based on score of game so move will be a positive and fireball hits will remove from main highscore
            #the current iteration of an episode will most likely not work
            room, lost1, new_state, reward = env.checkSpot(p, room, user, hzd) # THIS IS WHERE THE REWARDS ARE GOING TO BE HANDLED

            if reward == -1:
                #lost2 changed to done for ai purposes
                done, reward = env.activeLoss(p, reward) # MAY HAVE TO RESTART THE GAME HERE IF NOT IN CHECKSPOT

            # print("Reward "+str(reward))
            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            if lost1 == True:
                done == True
            #Every step we update replay memory and train main network
            #gamer.model.summary()   
            gamer.update_replay_memory((current_state, action, reward, new_state, done))
            gamer.train(done, epiStep)

            if lost1 == True:
                os.system("cls")
                print("\n\n")
                print("You walked into fire!!!".center(43))
                roomPrint(p, room)
                print("You walked into fire!!!".center(43))
                print("\n\n")
                break
            
            if done == True:
                os.system("cls")
                print("\n\n")
                print("Fire fell on you!!!".center(43))
                roomPrint(p, room)
                print("Fire fell on you!!!".center(43))
                print("\n\n")
                break
            
            if running == False:
                os.system("cls")
                break
            
            if p.prog == 6:
                os.system("cls")
                break
            
        current_state = new_state
        epiStep += 1
        
        #gamer.model.summary()
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            gamer.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                gamer.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model', save_format = "h5")

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    if p.prog == 6:
        print("\n\n")
        print("You Win!!!".center(43))
        print("\n\n")
    else:
        print("\n\n")
        print("Thanks for Playing!!!".center(43))
        print("\n\n")
        
    os.system("pause")

main(1,"RL")
