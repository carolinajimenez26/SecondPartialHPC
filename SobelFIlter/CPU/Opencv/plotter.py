import numpy as np
import matplotlib.pyplot as plt
import sys

def readData(fileName):
    N = 0
    f = open(fileName,"r")
    data = f.readlines()

    dic = {} # dic[image] = [size, time]
    for i in range(1,len(data)):
        words = data[i].split()
        # print (words)
        key = words[1]
        if (i == 1):
            key_aux = words[1]
        if (key_aux == words[1] or i == 1):
            N += 1
        if (key in dic):
            dic[key] = [float(words[0]), dic[key][1] + float(words[2])]
        else:
            dic[key] = [float(words[0]), float(words[2])]

    return dic, N

def getAxis(x, y, dic, N):
    for e in dic: # get the average of the times
        dic[e][1] = round(dic[e][1]/N, 3)
        print (e + " -> ")
        print (dic[e])
        x_aux.append(dic[e][1])
        y_aux.append(dic[e][0])

if __name__ == "__main__":

    # print (sys.argv)
    # print (len(sys.argv))
    if (len(sys.argv) != 3):
        print ("Usage: <times fileName> <plot title>")
        sys.exit()

    dic, N = readData(sys.argv[1])
    print ("N: ", N)
    x_aux = []
    y_aux = []
    getAxis(x_aux, y_aux, dic, N)

    x = np.array(x_aux)
    y = np.array(y_aux)
    plt.subplot(111)
    plt.xlabel('time (seconds)')
    plt.ylabel('image size (height*width)')
    plt.scatter(x, y)
    # plt.yscale('linear')
    plt.title('Sobel Filter, ' + sys.argv[2])
    plt.grid(True)
    # plt.plot(x, y)
    # plt.show()
    plt.savefig(sys.argv[2] + '.png')
