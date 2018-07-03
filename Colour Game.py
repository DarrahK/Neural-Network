
# Standard lib
import random
import csv
import copy
import time

# External lib
import numpy as np
import pygame



class ColorBlob:

    def __init__(self, r=False, g=False, b=False):
        if not r:
            self.r = random.randrange(0, 255)
            self.g = random.randrange(0, 255)
            self.b = random.randrange(0, 255)
        else:
            self.r = r
            self.g = g
            self.b = b
        self.colour_text = []

    def __str__(self):
        return "R: {}, G: {}, B: {}".format(self.r, self.g, self.b)

    def randomSuffle(self):
        self.r = random.randrange(0, 255)
        self.g = random.randrange(0, 255)
        self.b = random.randrange(0, 255)
        return

    def textcolour(self, colour):
        self.colour_text = np.array(colour)
        return

    def datacolour(self):
        my_file = open("Colour data.csv", "a")
        my_file.write(str(self.r) + ", " + str(self.g) + ", " + str(self.b) + ", " + str(self.colour_text) + "\n")
        return


class Network:

    def __init__(self, sizes):
        """
        This function creates the neural network. Sizes is a list that contains the input of how many nodes there is on
        each layer of the network.
        """
        self.sizes = sizes
        self.layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]
        self.zs = []

        # Collecting structure data.
        my_file = open("Starting Data.csv", "w")
        my_file.write("Network structure")
        for size in self.sizes:
            my_file.writelines( ", " + str(size) + " ")
        # Collecting baises.
        my_file.write("\nBaises")
        for layer in range(self.layers-1):
            # Reshaping data
            B = np.reshape(self.biases[layer], (1, (np.shape(self.biases[layer])[0])))
            my_file.writelines(", " + str(B[0]) + " ")
        # Collecting weights
        for layer in range(self.layers-1):
            my_file.write("\nWeights for layer " + str(layer))
            size = self.sizes[layer + 1]
            for i in range(size):
                my_file.writelines(", " + str(self.weights[layer][i]) + " ")

    def feedForward(self, a_data):
        """
        This feeds the A_data through the neural network.
        """
        # This converts it to an np array so we are able to use np matrix mul
        self.a_data = []
        self.nodes = []
        B = np.matrix(a_data)
        self.a_data = np.reshape(B, (np.shape(B)[1], 1))
        self.nodes.append(self.a_data)
        for x in range(self.layers - 1):
            self.zs.append(np.add(np.dot(self.weights[x], self.nodes[x]), self.biases[x]))
            self.nodes.append(sigmoid(np.add(np.dot(self.weights[x], self.nodes[x]), self.biases[x])))
        # Collecting node data
        my_file = open("Starting Data.csv", "a")
        my_file.write("\nNodes")
        for layer in range(self.layers):
            # Reshaping data
            B = np.reshape(self.nodes[layer], (1, (np.shape(self.nodes[layer])[0])))
            my_file.writelines(", " + str(B[0]) + " ")
        return

    def back_prop(self, x, y):
        self.feedForward(x)
        y_tran = np.matrix(y).transpose()
        delta = np.multiply(cost_prime(self, y_tran), sigmoid_prime(self.zs[-1]))
        biases_copy = copy.deepcopy(self.biases)
        weights_copy = copy.deepcopy(self.weights)
        biases_copy[-1] = delta
        nodes_2 = np.matrix(self.nodes[-2])
        weights_copy[-1] = np.dot(delta, nodes_2.transpose())
        for back in range(1, self.layers - 1):
            weight = np.matrix(self.weights[-back])
            delta = np.multiply(np.dot(weight, delta), sigmoid_prime(self.zs[-back-1]))
            biases_copy[-back] = delta
            nodes = np.matrix(self.nodes[-back-1])
            weights_copy[-back] = np.dot(delta, nodes.transpose())

        return biases_copy, weights_copy


def sigmoid(x):
    """
    Sigmoid function for a matrix applied to every element.
    """
    x = np.matrix(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """
    Sigmoid prime function for a matrix applied to every element.
    """
    x = np.matrix(x)
    out = np.multiply(sigmoid(x),(1-sigmoid(x)))
    return out

def costFunction(network, correct_data):
    """
    Works for out the cost function
    """
    return np.sum(np.square(np.subtract(network.nodes[network.layers - 1], correct_data)))

def cost_prime(self, y):
    return np.subtract(self.nodes[-1], y)

# PyGame

# Setting width and height.
game_display = pygame.display.set_mode((800, 600))
# Setting up window name.
pygame.display.set_caption("Colour Blob Game")
# Setting up game clock.
clock = pygame.time.Clock()

def text_objects(text, font, colour):
    textSurface = font.render(text, True, colour)
    return textSurface, textSurface.get_rect()

def button(msg,x,y,w,h,ic,ac,colour):
    mouse = pygame.mouse.get_pos()
    click =  pygame.mouse.get_pressed()

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(game_display, ac,(x,y,w,h))
        if click[0] == 1:
            if msg == "White":
                return 1
            elif msg == "Black":
                return 0
    else:
        pygame.draw.rect(game_display, ic,(x,y,w,h))
    pygame.font.init()
    smallText = pygame.font.SysFont("comicsansms",30)
    textSurf, textRect = text_objects(msg, smallText,colour)
    textRect.center = ( (x+(w/2)), (y+(h/2)) )
    game_display.blit(textSurf, textRect)
    pygame.font.quit()

def draw_enviroment(blob, network, colour):
    # Able to customise the background colour.
    game_display.fill(colour)
    # Adds the two circles with the colour of the ColourBlob class.
    pygame.draw.circle(game_display, (100, 100, 100), [200, 300], 130)
    pygame.draw.circle(game_display, (100, 100, 100), [600, 300], 130)
    pygame.draw.circle(game_display, (blob.r, blob.g, blob.b), [200,300], 125)
    pygame.draw.circle(game_display, (blob.r, blob.g, blob.b), [600, 300], 125)
    # Adds black and white text on top of the circles.

    # Networks guess

    if network.nodes[-1] < 0.5:
        pygame.draw.circle(game_display, (100, 100, 100), [200, 100], 30)
        pygame.draw.circle(game_display, (0, 0, 0), [200, 100], 25)
    else:
        pygame.draw.circle(game_display, (100, 100, 100), [600, 100], 30)
        pygame.draw.circle(game_display, (255, 255, 255), [600, 100], 25)

    button("Black", 150, 275, 100, 50, (blob.r, blob.g, blob.b), (175, 175, 175),(0,0,0))
    button("White", 550, 275, 100, 50, (blob.r, blob.g, blob.b), (175, 175, 175),(255,255,255))
    network.feedForward([blob.r,blob.g,blob.b])
    pygame.display.update()


def main():
    blob = ColorBlob()
    # You are able to change the layout of the network here
    # REMEBER EVERTIME YOU RUN THE PROGRAM IT HAS A DIFFERENT WEIGHTS AND BIASES
    network = Network([3, 5, 1])
    # Feeding the data into the network
    network.feedForward([blob.r, blob.g, blob.b])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        network.feedForward([blob.r, blob.g, blob.b])

        draw_enviroment(blob, network, (195,195,195))
        clock.tick(60)
        if button("Black", 150, 275, 100, 50, (blob.r, blob.g, blob.b), (175, 175, 175), (0,0,0)) == 0:
            my_file = open("Data.csv", "a")
            my_file.write("\n" + str(blob.r) + ", " + str(blob.g) + ", " + str(blob.g) + ", " + str(0))
            blob = ColorBlob()
            network.feedForward([blob.r, blob.g, blob.b])
            time.sleep(0.2)
        elif button("White", 550, 275, 100, 50, (blob.r, blob.g, blob.b), (175, 175, 175), (255,255,255)) == 1:
            my_file = open("Data.csv", "a")
            my_file.write("\n" + str(blob.r) + ", " + str(blob.g) + ", " + str(blob.g) + ", " + str(1))
            blob = ColorBlob()
            time.sleep(0.2)

if __name__ == "__main__":
    main()