import multiprocessing as mp
import time
import random
import collections
import sys
import numpy as np

class ComputePath:
    def __init__(self, distance, ID2Index, lives,num):
        self.distance = distance
        self.ID2Index = ID2Index
        self.crossRate = 0.7  # cross rate
        self.mutationRate = 0.02  # mutation rate
        self.num_of_life = num # population number, number of series
        self.geneLength = 0  # number of items
        self.lives = lives  # population
        self.best = lives[-1]  # optimal individuals
        self.generation = 1
        self.crossCount = 0  # cross time
        self.mutationCount = 0  # mutation time
        self.order=[]

    def evaluation(self):
        #print(self.lives)
        #self.best = self.lives[0]
        #print(self.generation)
        best_score=self.Fitness(self.best)
        for life in self.lives:
            #print(life)
            score = self.Fitness(life)
            #print(best_score)
            if score > best_score:
                self.best = life
                best_score = score

    def crossover(self, parent1, parent2):
        left = random.randint(1, len(parent1) - 2)
        right = random.randint(left, len(parent1) - 2)
        newgene = collections.deque()
        newgene.extend(parent1[left:right])
        point = 0
        for g in parent2[1:-1]:
            if g not in parent1[left:right]:
                if point<left:
                    newgene.appendleft(g)
                    point += 1
                else:
                    newgene.append(g)
        newgene.appendleft(0)
        newgene.append(len(parent1)-1)
        self.crossCount += 1
        return list(newgene)

    def Mutation(self, gene):
        newg = gene
        left = random.randint(1, len(gene) - 2)
        right = random.randint(1, len(gene) - 2)
        newg[left], newg[right] = newg[right], newg[left]
        self.mutationCount += 1
        #print(gene)
        return list(newg)

    def select(self):
        index = random.randint(0, len(self.lives)-1)
        life = self.lives[index]
        return life

    def Child(self):
        parent1 = self.select()
        rate = random.random()
        if rate < self.crossRate:
            parent2 = self.select()
            gene = self.crossover(parent1, parent2)
        else:
            gene = parent1
        # ra = random.random()
        # if ra < self.mutationRate:
        #     gene = self.Mutation(gene)
        return gene

    def next_generation(self):
        self.evaluation()
        newLives = []
        newLives.append(self.best)
        #print(newLives[0],self.New_distance(newLives[0]))
        while len(newLives) < self.num_of_life:
            newLives.append(self.Child())
        self.lives[:] = newLives
        self.generation += 1
        #print(self.lives)

    def New_distance(self,lifes):
        new_distance = 0.0
        #print(lifes)
        for i in range(0, self.geneLength - 1):
            index1, index2 = self.ID2Index[self.order[lifes[i]]], self.ID2Index[self.order[lifes[i + 1]]]
            new_distance+=self.distance[index1][index2]
        return new_distance

    def Fitness(self, life):
        return 1.0/self.New_distance(life)

    def GA(self,order_list,q):
        dis=0
        n=2000
        self.order = ['000'] + order_list + ['-1']
        self.geneLength=len(self.order)
        while n > 0:
            self.next_generation()
            dis = self.New_distance(self.best)
            n -= 1
        #print(dis)
        res=[]
        for i in self.best:
            res.append(self.order[i])
        #print(res[1:-1])
        q.put(res[1:-1])
        q.put(dis)



def initialpopulation(num,geneLength,bl):
    lives=[]
    while len(lives)<num:
        temp = [x for x in range(1,geneLength+1)]
        random.shuffle(temp)
        gene=[0]+temp+[geneLength+1]
        if gene not in lives:
            lives.append(gene)
    bl.append(lives)
    #q.put(lives)
 

def greedy(orderlist,ID2Index,distance,al):
    new_dis = convert(orderlist,ID2Index,distance)
    N = len(new_dis)
    path_distance = 0  # least distance of current path
    s = []  # items has been picked up
    s.append(0)  # 0 is the starting location
    i, j = 1, 0
    for i in range(1, N-1):  # N-1 because we don't care about end point
        k = 1
        shortest = sys.maxsize  # shortest distance of current point
        for k in range(1, N-1):  # Greedy algo doesnt care about end
            picked = 0  # whether the item is picked or not
            if k in s:
                picked = 1
            if (picked == 0) and (new_dis[k][s[i - 1]] < shortest):
                j = k
                shortest = new_dis[k][s[i - 1]]
        s.append(j)
        path_distance += shortest
    path_distance += new_dis[-1][j]  # Distance to the end point
    res = []
    for item in s[1:]:
        res.append(orderlist[item-1])
    al.append(s+[len(orderlist)+1])
    #print(al)

    #q.put(s+[len(orderlist)+1])
    # print(q)

def convert(orderlist,ID2Index,distance):
    length = len(orderlist)+2
    start_pt = '000'  # By convention start_pt id is '000'
    end_pt = '-1'  # Similarly end_pt id is '-1'
    new_list = [start_pt]+orderlist+[end_pt]
    fidistance = []
    for i in range(length):
        temp = [0]*length
        for j in range(length):
            sta = ID2Index[new_list[i]]
            end = ID2Index[new_list[j]]
            temp[j] = distance[sta][end]
        fidistance.append(temp)
        #print(distance)
    return fidistance

def parallel(order_list,distance_test_1,ID2Index):
    length = len(order_list)
    num = 1
    if len(order_list) <= 5:
        for i in range(1, len(order_list) + 1):
            num = num * i
    else:
        num = 800
  
    # s1 = mp.Queue()
    # s2 = mp.Queue()
    ary=mp.Manager()
    a_L=ary.list()
    bry=mp.Manager()
    b_L=bry.list()
    st1 = mp.Process(target=initialpopulation, args=(num, length,b_L,))
    st2 = mp.Process(target=greedy, args=(order_list,ID2Index,distance_test_1,a_L,))
    
    st1.start()
    st2.start()
    st1.join()
    st2.join()
    st1.terminate()
    st2.terminate()
    
    # lives = list(s1.get())
    # best_gre=list(s2.get())
    lives=b_L[0]
    best_gre=a_L[0]
    # print(list(best_gre))
    #lives = initialpopulation(num, length)
    a = int(len(lives) / 4)
    l1 = lives[:a]+[best_gre]
    l2 = lives[a:2*a]+[best_gre]
    l3 = lives[2*a:3*a]+[best_gre]
    l4 = lives[3*a:]+[best_gre]
    com1 = ComputePath(distance_test_1, ID2Index, l1, num / 4)
    com2 = ComputePath(distance_test_1, ID2Index, l2, num / 4)
    com3 = ComputePath(distance_test_1, ID2Index, l3, num / 4)
    com4 = ComputePath(distance_test_1, ID2Index, l4, num / 4)
    q1 = mp.Queue()
    q2 = mp.Queue()
    q3 = mp.Queue()
    q4 = mp.Queue()
    p1 = mp.Process(target=com1.GA, args=(order_list, q1,))
    p2 = mp.Process(target=com2.GA, args=(order_list, q2,))
    p3 = mp.Process(target=com3.GA, args=(order_list, q3,))
    p4 = mp.Process(target=com4.GA, args=(order_list, q4,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    res1 = q1.get()
    dis_1= q1.get()
    res2 = q2.get()
    dis_2 =q2.get()
    res3 = q3.get()
    dis_3= q3.get()
    res4 = q4.get()
    dis_4 =q4.get()
    reslist=[res1,res2,res3,res4]
    dislist=[dis_1,dis_2,dis_3,dis_4]
    ind = dislist.index(min(dislist))
    print(reslist[ind],dislist[ind])

def distance_array(user_input):
    end = user_input+1
    #order_list
    order_list = []
    for i in range (1,end):
        order_list.append(str(i))
    # index dictionary
    ID2Index = {}
    ID2Index['000']=0
    ID2Index['-1']=end
    for i in range (1,end):
        ID2Index[str(i)] = i
    #distance array
    l=np.loadtxt('list.txt').astype(int) #'list' is the name of the txt file of city list
    c = np.insert(l, 0, np.array([[0,0,0]]), axis=0) #add start (0,0,0) and terminal (0,0,0)
    c = np.insert(c, user_input+1, np.array([[0,0,0]]), axis=0)
    # distance=np.zeros((user_input+2,user_input+2))
    distance = []
    for a in range(user_input+2):
        temp = [0] * (user_input+2)
        for b in range(user_input+2):
            temp[b] = int(np.sqrt(np.square((c.T[1][a]-c.T[1][b]))+np.square((c.T[-1][a]-c.T[-1][b]))))
        distance.append(temp)
    return order_list, ID2Index, distance

def main():
    # order_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
    # ID2Index = {}
    # ID2Index['000'] = 0
    # ID2Index['-1'] = 31
    # ID2Index['1'] = 1
    # ID2Index['2'] = 2
    # ID2Index['3'] = 3
    # ID2Index['4'] = 4
    # ID2Index['5'] = 5
    # ID2Index['6'] = 6
    # ID2Index['7'] = 7
    # ID2Index['8'] = 8
    # ID2Index['9'] = 9
    # ID2Index['10'] = 10
    # ID2Index['11'] = 11
    # ID2Index['12'] = 12
    # ID2Index['13'] = 13
    # ID2Index['14'] = 14
    # ID2Index['15'] = 15
    # ID2Index['16'] = 16
    # ID2Index['17'] = 17
    # ID2Index['18'] = 18
    # ID2Index['19'] = 19
    # ID2Index['20'] = 20
    # ID2Index['21'] = 21
    # ID2Index['22'] = 22
    # ID2Index['23'] = 23
    # ID2Index['24'] = 24
    # ID2Index['25'] = 25
    # ID2Index['26'] = 26
    # ID2Index['27'] = 27
    # ID2Index['28'] = 28
    # ID2Index['29'] = 29
    # ID2Index['30'] = 30
    # distance = [[11, 17, 5, 25, 13, 13, 13, 27, 9, 23, 4, 4, 16, 23, 17, 13, 18, 14, 4, 37, 37, 23, 19, 5, 12, 17, 17, 11, 12, 5, 37, 11],
    #                    [17, 0, 9, 28, 16, 7, 21, 24, 16, 30, 8, 8, 13, 11, 12, 16, 26, 24, 8, 16, 16, 11, 14, 9, 13, 0, 0, 11, 16, 14, 16, 17],
    #                    [5, 9, 0, 8, 6, 19, 13, 10, 6, 20, 8, 8, 12, 2, 13, 6, 10, 10, 8, 7, 7, 2, 13, 0, 8, 9, 9, 12, 7, 8, 7, 5],
    #                    [25, 28, 8, 0, 10, 23, 18, 10, 6, 12, 16, 16, 20, 10, 19, 10, 18, 10, 16, 12, 12, 10, 18, 8, 16, 28, 28, 26, 15, 16, 12, 25],
    #                    [13, 16, 6, 10, 0, 5, 14, 4, 6, 18, 12, 12, 12, 4, 9, 0, 18, 10, 12, 16, 16, 4, 8, 6, 16, 16, 16, 2, 7, 12, 16, 13],
    #                    [13, 7, 19, 23, 5, 0, 12, 7, 6, 17, 17, 17, 15, 6, 8, 5, 12, 13, 17, 13, 13, 6, 7, 19, 11, 7, 7, 9, 11, 9, 13, 13],
    #                    [13, 21, 13, 18, 14, 12, 0, 14, 26, 6, 22, 22, 10, 21, 13, 14, 1, 10, 22, 16, 16, 21, 10, 13, 6, 21, 21, 22, 26, 12, 16, 13],
    #                    [27, 24, 10, 10, 4, 7, 14, 0, 4, 8, 16, 16, 14, 8, 7, 4, 16, 10, 16, 6, 6, 8, 4, 10, 12, 24, 24, 10, 9, 12, 6, 27],
    #                    [9, 16, 6, 6, 6, 6, 26, 4, 0, 12, 12, 12, 6, 6, 11, 6, 14, 6, 12, 26, 26, 6, 24, 6, 4, 16, 16, 18, 11, 12, 26, 9],
    #                    [23, 30, 20, 12, 18, 17, 6, 8, 12, 0, 13, 13, 24, 15, 17, 18, 6, 8, 13, 15, 15, 15, 20, 20, 20, 30, 30, 12, 20, 6, 15, 23],
    #                    [4, 8, 8, 16, 12, 17, 22, 16, 12, 13, 0, 0, 20, 8, 1, 12, 10, 16, 0, 10, 10, 8, 13, 8, 16, 8, 8, 6, 11, 4, 10, 4],
    #                    [4, 8, 8, 16, 12, 17, 22, 16, 12, 13, 0, 0, 20, 8, 1, 12, 10, 16, 0, 10, 10, 8, 13, 8, 16, 8, 8, 6, 11, 4, 10, 4],
    #                    [16, 13, 12, 20, 12, 15, 10, 14, 6, 24, 20, 20, 0, 23, 3, 12, 8, 21, 20, 16, 16, 23, 22, 12, 4, 13, 13, 30, 28, 14, 16, 16],
    #                    [23, 11, 2, 10, 4, 6, 21, 8, 6, 15, 8, 8, 23, 0, 2, 4, 9, 10, 8, 11, 11, 0, 14, 2, 19, 11, 11, 9, 5, 8, 11, 23],
    #                    [17, 12, 13, 19, 9, 8, 13, 7, 11, 17, 1, 1, 3, 2, 0, 9, 18, 16, 1, 9, 9, 2, 18, 13, 7, 12, 12, 16, 11, 9, 9, 17],
    #                    [13, 16, 6, 10, 0, 5, 14, 4, 6, 18, 12, 12, 12, 4, 9, 0, 18, 10, 12, 16, 16, 4, 8, 6, 16, 16, 16, 2, 7, 12, 16, 13],
    #                    [18, 26, 10, 18, 18, 12, 1, 16, 14, 6, 10, 10, 8, 9, 18, 18, 0, 17, 10, 8, 8, 9, 8, 10, 6, 26, 26, 11, 12, 10, 8, 18],
    #                    [14, 24, 10, 10, 10, 13, 10, 10, 6, 8, 16, 16, 21, 10, 16, 10, 17, 0, 16, 12, 12, 10, 9, 10, 25, 24, 24, 11, 13, 16, 12, 14],
    #                    [4, 8, 8, 16, 12, 17, 22, 16, 12, 13, 0, 0, 20, 8, 1, 12, 10, 16, 0, 10, 10, 8, 13, 8, 16, 8, 8, 6, 11, 4, 10, 4],
    #                    [37, 16, 7, 12, 16, 13, 16, 6, 26, 15, 10, 10, 16, 11, 9, 16, 8, 12, 10, 0, 0, 11, 10, 7, 12, 16, 16, 14, 18, 4, 0, 37],
    #                    [37, 16, 7, 12, 16, 13, 16, 6, 26, 15, 10, 10, 16, 11, 9, 16, 8, 12, 10, 0, 0, 11, 10, 7, 12, 16, 16, 14, 18, 4, 0, 37],
    #                    [23, 11, 2, 10, 4, 6, 21, 8, 6, 15, 8, 8, 23, 0, 2, 4, 9, 10, 8, 11, 11, 0, 14, 2, 19, 11, 11, 9, 5, 8, 11, 23],
    #                    [19, 14, 13, 18, 8, 7, 10, 4, 24, 20, 13, 13, 22, 14, 18, 8, 8, 9, 13, 10, 10, 14, 0, 13, 26, 14, 14, 12, 2, 10, 10, 19],
    #                    [5, 9, 0, 8, 6, 19, 13, 10, 6, 20, 8, 8, 12, 2, 13, 6, 10, 10, 8, 7, 7, 2, 13, 0, 8, 9, 9, 12, 7, 8, 7, 5],
    #                    [12, 13, 8, 16, 16, 11, 6, 12, 4, 20, 16, 16, 4, 19, 7, 16, 6, 25, 16, 12, 12, 19, 26, 8, 0, 13, 13, 26, 24, 10, 12, 12],
    #                    [17, 0, 9, 28, 16, 7, 21, 24, 16, 30, 8, 8, 13, 11, 12, 16, 26, 24, 8, 16, 16, 11, 14, 9, 13, 0, 0, 11, 16, 14, 16, 17],
    #                    [17, 0, 9, 28, 16, 7, 21, 24, 16, 30, 8, 8, 13, 11, 12, 16, 26, 24, 8, 16, 16, 11, 14, 9, 13, 0, 0, 11, 16, 14, 16, 17],
    #                    [11, 11, 12, 26, 2, 9, 22, 10, 18, 12, 6, 6, 30, 9, 16, 2, 11, 11, 6, 14, 14, 9, 12, 12, 26, 11, 11, 0, 14, 12, 14, 11],
    #                    [12, 16, 7, 15, 7, 11, 26, 9, 11, 20, 11, 11, 28, 5, 11, 7, 12, 13, 11, 18, 18, 5, 2, 7, 24, 16, 16, 14, 0, 7, 18, 12],
    #                    [5, 14, 8, 16, 12, 9, 12, 12, 12, 6, 4, 4, 14, 8, 9, 12, 10, 16, 4, 4, 4, 8, 10, 8, 10, 14, 14, 12, 7, 0, 4, 5],
    #                    [37, 16, 7, 12, 16, 13, 16, 6, 26, 15, 10, 10, 16, 11, 9, 16, 8, 12, 10, 0, 0, 11, 10, 7, 12, 16, 16, 14, 18, 4, 0, 37],
    #                    [11, 17, 5, 25, 13, 13, 13, 27, 9, 23, 4, 4, 16, 23, 17, 13, 18, 14, 4, 37, 37, 23, 19, 5, 12, 17, 17, 11, 12, 5, 37, 11]]

    print('Please input the order size:')
    user_input = int(input())
    order_list, ID2Index, distance = distance_array(user_input)
    start = time.time()
    parallel(order_list, distance, ID2Index)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
