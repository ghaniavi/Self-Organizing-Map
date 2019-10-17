import csv
import random
import math
import matplotlib.pyplot as plt

# for save the data set
dataset_X = []
dataset_Y = []

lr = 0.8
lr0 = 0.8

# for save the class
unit = []

m_iNumIterations = 20000
m_dMapRadius = 8.5

clas = {}

# set the dataset into X and Y
# dataset_X is for the first column
# dataset_Y is for the second column
def openDataset():
    with open('TanpaLabel.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            dataset_X.append(float(row[0]))
            dataset_Y.append(float(row[1]))
    csvfile.close()

# STEP 1: Randomize the map’s nodes weight
# randomize 15 class
def setUnit():
    for i in range(15):
        a=[]
        a.append(float(random.uniform(3, 17)))
        a.append(float(random.uniform(3, 17)))
        unit.append(a)
    return unit

# STEP 2: Select randomly one instance
# select one data from dataset
def choiceranddata():
    choice = []
    x = []
    y = []
    x = random.choice(dataset_X)
    y = dataset_Y[dataset_X.index(x)]
    choice.append(x)
    choice.append(y)
    return choice

# STEP 3: Find the closest node: best matching unit
# find the distance for every 15 class with the choosen dataset
def distance():
    array = []
    for i in range(15):
        a = ((unit[i][0]-choice[0])**2)
        b = ((unit[i][1]-choice[1])**2)
        c = math.sqrt(a+b)
        array.append(c)
    return array

# STEP 4: The codebook of this node is updated
# Update the value of the class which has the nearest distance
def weight(a):
    for i in range(2):
        unit[a][i] = (unit[a][i] + (lr * (choice[i] - unit[a][i])))

# STEP 6: The codebooks of the adjacent
# nodes are also updated, by not
# to the same degree
def update_tetangga(w, neighbourhoodradius):
    for i in range(15):
        a = ((unit[w][0] - unit[i][0]) ** 2)
        b = ((unit[w][1] - unit[i][1]) ** 2)
        c = math.sqrt(a+b)
        if ((c > 0) and (c < neighbourhoodradius)):
            # h() is a neighborhood function. Its amplitude (spatial
            # width of the kernel) decreases according to the step
            # index (t)
            h = math.exp(-((c**2)/(2*neighbourhoodradius**2)))
            for j in range(2):
                unit[i][j] = unit[i][j] + lr * h * (choice[j] - unit[i][j])

# Classification for every dataset
def classification():
    for i in range(15):
        clas[i] = []
    for k in range(600):
        data = []
        for j in range(15):
            a = ((dataset_X[k] - unit[j][0]) ** 2)
            b = ((dataset_Y[k] - unit[j][1]) ** 2)
            c = math.sqrt(a+b)
            data.append(c)
        clas[data.index(min(data))].append([dataset_X[k], dataset_Y[k]])

# for show the visual representation of the data and the program
def gambar():
    colors = ['maroon', 'orange', 'yellow', 'olive', 'green', 'purple', 'salmon', 'lime', 'teal', 'aqua',
              'pink', 'navy', 'black', 'darkslategrey', 'chocolate']
    for i in range(15):
        for j in range(len(clas[i])):
            plt.scatter(clas[i][j][0], clas[i][j][1], color=colors[i])
    for i in range(15):
        plt.scatter(unit[i][0],unit[i][1],color='red')
    plt.show()

if __name__ == '__main__':
    print("===============================   Running process ...   ===============================")
    openDataset()

    # train
    for iterasi in range(m_iNumIterations):
        setUnit()
        choice = choiceranddata()
        a = distance()
        min_a = min(a)
        b = a.index(min(a))
        weight(b)

        m_dMapRadius = max(dataset_X[0], dataset_Y[1]) / 2
        m_dTimeConstant = m_iNumIterations / math.log(m_dMapRadius)
        if (iterasi < m_iNumIterations/6):
            neighbourhoodradius = m_dMapRadius * math.exp(iterasi/ m_dTimeConstant)
            update_tetangga(b, neighbourhoodradius)

        # learning rate decreases according the step index
        lr = lr0 * math.exp(-iterasi/m_iNumIterations)

    # classification process
    classification()
    # show the graph
    gambar()
