import numpy as np
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
    l=np.loadtxt('list').astype(int) #'list' is the name of the txt file of city list
    c = np.insert(l, 0, np.array([[0,0,0]]), axis=0) #add start (0,0,0) and terminal (0,0,0)
    c = np.insert(c, user_input+1, np.array([[0,0,0]]), axis=0)
    distance=np.zeros((user_input+2,user_input+2))
    for a in range(user_input+2):
        for b in range(user_input+2):
            distance[a][b]=np.sqrt(np.square((c.T[1][a]-c.T[1][b]))+np.square((c.T[-1][a]-c.T[-1][b])))
    return order_list, ID2Index, distance

