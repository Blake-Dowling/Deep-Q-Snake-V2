import snake_game as s
from snake_game import *
import time, random
import tensorflow as tf
from tensorflow import keras

def ob(block):
    return (block.loc[0] < 0 or
        block.loc[0] >= WIDTH or
        block.loc[1] < 0 or
        block.loc[1] >= WIDTH)
def look(snake):
    #Remove old vision blocks
    global visionBlocks
    for b in visionBlocks:
        del b
    visionBlocks = []
    #Create new vision blocks
    headX = snake.blocks[0].loc[0]
    headY = snake.blocks[0].loc[1]

    vision = [
            (headX + (((snake.dir % 2) == 1)  * (1 + ((snake.dir >= 2) * -2))), headY + (((snake.dir % 2) == 0)  * (1 + ((snake.dir < 2) * -2)))), #left
            (headX + (((snake.dir % 2) == 0)  * (1 + ((snake.dir >= 2) * -2))), headY + (((snake.dir % 2) == 1)  * (1 + ((snake.dir >= 2) * -2)))), #forward
            (headX + (((snake.dir % 2) == 1)  * (1 + ((snake.dir < 2) * -2))), headY + (((snake.dir % 2) == 0)  * (1 + ((snake.dir >= 2) * -2)))) #right
              ]
    #Highlight vision blocks
    
    for pixel in vision:
        visionBlocks.append(s.Block(pixel[0], pixel[1], "", "lime"))
    visionInput = [ob(e) for e in visionBlocks]
    return visionInput
def train(visionInput, output):
    global model
    visionInput = (np.array(visionInput)).reshape((1,3))
    model.fit(np.array(visionInput), np.array([output]), epochs=1)
def decide(snake, visionInput):
    global model
    visionInput = (np.array(visionInput)).reshape((1,3))
    turn = model.predict(visionInput, verbose=0)[0][0]
    print(turn)
    print(round(turn))
    snake.dir = (snake.dir + round(turn)) % 4
    print(snake.dir)

model = keras.Sequential()
layer0 = keras.layers.Flatten(input_shape=(3,))
model.add(layer0)
layer1 = keras.layers.Dense(6, activation="relu")
model.add(layer1)
layer2 = keras.layers.Dense(1, activation="sigmoid")
model.add(layer2)

model.compile(optimizer="adam", loss="binary_crossentropy")

#Animation loop
visionBlocks = []
snake, apple = s.initGame()
window.bind("<Right>", lambda event: snake.setDir(0))
window.bind("<Down>", lambda event: snake.setDir(1))
window.bind("<Left>", lambda event: snake.setDir(2))
window.bind("<Up>", lambda event: snake.setDir(3))
while True:
    time.sleep(.1)
    #Snake's vision
    visionInput = look(snake)
    #Decide
    decide(snake, visionInput)
    #Ate apple
    ate = s.checkAte(snake, apple) #Check if head on apple
    if ate:
        del apple 
        apple = s.Block(random.randint(0, s.WIDTH-1), random.randint(0, s.WIDTH-1), "red", "black") #Move apple
    s.window.update()
    #Move snake
    snake.move(ate) #Move snake
    #Out of bounds
    if s.checkOB(snake):
        train(visionInput, 1)
        snake, apple = s.initGame()
    else:
        train(visionInput, 0)
    

    
    
    