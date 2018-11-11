from Neural import NeuralNet as nn
import pygame
import random
import time

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Ball():
    def __init__(self, r, paddle1, paddle2, x=250, y=250, xS=0, yS=0):
        self.r=r
        self.x=x
        self.y=y
        self.xSpeed = xS
        self.ySpeed = yS
        self.paddle1 = paddle1
        self.paddle2 = paddle2

    def changeSpeed(self, dx, dy):
        self.xSpeed = dx
        self.ySpeed = dy

    def getXSpeed(self):
        return self.xSpeed

    def getYSpeed(self):
        return self.ySpeed

    def move(self,dx,dy):
        self.x+=dx
        self.y+=dy

    def step(self):
        if self.x - self.r<=0:
            self.xSpeed=abs(self.xSpeed)
        if self.x + self.r>=500:
            self.xSpeed=-abs(self.xSpeed)
        if self.y - self.r<=0:
            self.ySpeed=abs(self.ySpeed)
        if self.y + self.r>=600:
            self.ySpeed=-abs(self.ySpeed)
        if 550<=self.y + self.r<=560:
            if self.paddle1.x-self.paddle1.getWidth()/2<self.x+self.r and self.paddle1.x+self.paddle1.getWidth()/2>self.x-self.r:
                self.ySpeed=-abs(self.ySpeed)
                self.xSpeed+=(self.x-self.paddle1.x)/5
        if 40<=self.y - self.r<=50:
            if self.paddle2.x-self.paddle2.getWidth()/2<self.x+self.r and self.paddle2.x+self.paddle2.getWidth()/2>self.x-self.r:
                self.ySpeed=abs(self.ySpeed)
                self.xSpeed+=(self.x-self.paddle2.x)/5
        self.move(self.xSpeed, self.ySpeed)

class Paddle():
    def __init__(self,x,y,net,width=100,height=10):
        self.x=x
        self.y=y
        self.width=width
        self.height=height
        self.net=net

    def getWidth(self):
        return self.width

    def act(self, inputs):
        move = self.net.propagate(inputs)
        move = move.index(max(move))
        if move == 0:
            self.left()
        elif move == 2:
            self.right()

    def left(self):
        if self.x>self.width/2:
            self.x-=10

    def right(self):
        if self.x<500-self.width/2:
            self.x+=10

def resetPlayers(new=[]):
    players=[]
    for i in range(numPlayers):

        if len(new)==0:
            net1 = nn.NeuralNetwork(5,6,5,3)
            net1.randomizeWeights()
            net2 = nn.NeuralNetwork(5,6,5,3)
            net2.randomizeWeights()
        else:
            net1 = new[i*2]
            net2 = new[i*2+1]

        paddle1 = Paddle(250,555,net1)
        paddle2 = Paddle(250,45,net2)


        ball = Ball(10,paddle1,paddle2,255,300,random.randint(-10,10),random.choice([10,-10]))

        players.append([paddle1,paddle2,ball])
    return players

pygame.init()
size = [500, 600]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Play Pong")
done = False
clock = pygame.time.Clock()

numPlayers=100

players=resetPlayers()
enemyPaddle= nn.NeuralNetwork(1,1,1,1)
n=0
while not done:

    remaning = [x for x in range(numPlayers)]
    score=0
    nextGen=[]
    while len(remaning)>0:
        screen.fill(BLACK)
        for i in remaning:
            player = players[i]
            paddle1=player[0]
            paddle2=player[1]
            ball=player[2]
            inputs1 = [
                ball.y/300-1,
                ball.getXSpeed()/20,
                #paddle2.y/300-1,
                #paddle2.x/250-1,
                paddle1.y/300-1,
                paddle1.x/250-1,
                ball.x/250-1
            ]
            paddle1.act(inputs1)
            inputs2 = [
                ball.y/300-1,
                ball.getXSpeed()/20,
                #paddle1.y/300-1,
                #paddle1.x/250-1,
                paddle2.y/300-1,
                paddle2.x/250-1,
                ball.x/250-1
            ]
            paddle2.act(inputs2)
            ball.step()
            if score > 1000:
                remaning.remove(i)
                nextGen.append(random.choice([paddle1.net,paddle2.net]))
            elif (ball.y+10>570 or ball.y-10<30) and len(remaning)>0:
                if ball.y+10>570:
                    nextGen.append(paddle2.net)
                else:
                    nextGen.append(paddle1.net)
                remaning.remove(i)
            w=paddle1.width/2
            h=paddle1.height/2
            pygame.draw.circle(screen, WHITE, [round(ball.x), round(ball.y)], ball.r)
            pygame.draw.rect(screen, WHITE, [round(paddle1.x-w), round(paddle1.y-h), paddle1.width, paddle1.height])
            pygame.draw.rect(screen, WHITE, [round(paddle2.x-w), round(paddle2.y-h), paddle2.width, paddle2.height])
        score+=1
        if score > 50000:
            pygame.display.flip()
            #clock.tick(100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                remaining=[]
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    done = True
                    remaining=[]
    if not done:
        mutationChance=1
        mutationScale=1
        if score>100:
            mutationChance=0.5
            mutationScale=0.5
        if score>500:
            mutationChance=0.35
            mutationScale=0.4
        if score > 900:
            mutationChance=0.2
            mutationScale=0.3
        for org in nextGen:
            org.addScore(1)
        for _ in range(numPlayers*2-len(nextGen)):
            newOrg=random.choice(nextGen).clone()
            newOrg.mutate(mutationChance,mutationScale)
            nextGen.append(newOrg)
        print(nextGen[4].inWeights,nextGen[4].hiddenWeights,nextGen[4].outWeights)
        print(n+1)
        players=resetPlayers(nextGen)
        n+=1
    else:
        for org in nextGen:
            if org.getScore()>enemyPaddle.getScore():
                enemyPaddle=org
        print(enemyPaddle.getScore())


playerPaddle = Paddle(250,555,nn.NeuralNetwork(1,1,1,1))
enemyPaddle = Paddle(250,45,enemyPaddle)
ball = Ball(10,playerPaddle,enemyPaddle,250,300,random.randint(-10,10),random.choice([10,-10]))
left=False
right=False
while done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:        # left arrow turns left
                left = True
            elif event.key == pygame.K_RIGHT:
                right = True
            elif event.key == pygame.K_SPACE:
                done = False
        elif event.type == pygame.KEYUP:            # check for key releases
            if event.key == pygame.K_LEFT:        # left arrow turns left
                left = False
            elif event.key == pygame.K_RIGHT:     # right arrow turns right
                right = False
    screen.fill(BLACK)
    inputs = [
        ball.y/300-1,
        ball.getXSpeed()/20,
        #playerPaddle.y/300-1,
        #playerPaddle.x/250-1,
        enemyPaddle.y/300-1,
        enemyPaddle.x/250-1,
        ball.x/250-1
    ]
    enemyPaddle.act(inputs)
    ball.step()
    if left:
        playerPaddle.left()
    elif right:
        playerPaddle.right()
    w=playerPaddle.width/2
    h=playerPaddle.height/2
    pygame.draw.circle(screen, WHITE, [round(ball.x), round(ball.y)], ball.r)
    pygame.draw.rect(screen, WHITE, [round(playerPaddle.x-w), round(playerPaddle.y-h), 2*w, 2*h])
    pygame.draw.rect(screen, WHITE, [round(enemyPaddle.x-w), round(enemyPaddle.y-h), enemyPaddle.width, enemyPaddle.height])
    if abs(ball.getXSpeed())>20:
        ball.xSpeed=abs(ball.getXSpeed())/ball.getXSpeed()*20
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
