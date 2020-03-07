
import numpy as np

#Creating a directory called outputs in the current folder
import os
os.mkdir("./ouputs")

stamina = [0, 50, 100]
health = [0, 25, 50, 75, 100]
ACTIONS = ["SHOOT", "DODGE", "RECHARGE"]
arrow = [0, 1, 2, 3]
team = 2  # team number
step_cost = -20
bellman = 1e-3
states = [(h, a, s) for h in range( 5 ) for a in range( 4 ) for s in range( 3 )]
discount = 0.99
state_v = np.zeros((5, 4, 3))

######################################## PRINT FUNCTION FOR TASK1 IS HERE ###################################################################################

def Print1(itr, action_index):
    
    f=open("./ouputs/task_1_trace.txt","a")
    f.write(('iteration=' + str(itr)+'\n'))
    for h in range(5):
        for a in range(4):
            for s in range(3):
                string = ACTIONS[action_index[h][a][s]]
                if(h==0):
                    string=-1 #because the game is over
                finalstring= "("+ str(h) + "," + str(a) + "," + str(s) + "):" + str(string) + "=[" + str(round(state_v[h,a,s],3)) + "]\n"
                f.write(finalstring)
                finalstring=""
    f.write( "\n\n" )

def Print2(itr, action_index):
    
    f=open("./ouputs/task_2_part_1_trace.txt","a")
    f.write(('iteration=' + str(itr)+'\n'))
    for h in range(5):
        for a in range(4):
            for s in range(3):
                string = ACTIONS[action_index[h][a][s]]
                if(h==0):
                    string=-1 #because the game is over
                finalstring= "("+ str(h) + "," + str(a) + "," + str(s) + "):" + str(string) + "=[" + str(round(state_v[h,a,s],3)) + "]\n"
                f.write(finalstring)
                finalstring=""
    f.write( "\n\n" )

def Print3(itr, action_index):
    
    f=open("./ouputs/task_2_part_2_trace.txt","a")
    f.write(('iteration=' + str(itr)+'\n'))
    for h in range(5):
        for a in range(4):
            for s in range(3):
                string = ACTIONS[action_index[h][a][s]]
                if(h==0):
                    string=-1 #because the game is over
                finalstring= "("+ str(h) + "," + str(a) + "," + str(s) + "):" + str(string) + "=[" + str(round(state_v[h,a,s],3)) + "]\n"
                f.write(finalstring)
                finalstring=""
    f.write( "\n\n" )

def Print4(itr, action_index):

    f=open("./ouputs/task_2_part_3_trace.txt","a")
    f.write(('iteration=' + str(itr)+'\n'))
    for h in range(5):
        for a in range(4):
            for s in range(3):
                string = ACTIONS[action_index[h][a][s]]
                if(h==0):
                    string=-1 #because the game is over
                finalstring= "("+ str(h) + "," + str(a) + "," + str(s) + "):" + str(string) + "=[" + str(round(state_v[h,a,s],3)) + "]\n"
                f.write(finalstring)
                finalstring=""
    f.write( "\n\n" )

############################################################ UTILILTY FUNCTION HERE ##############################################################################

def Utility(s, a, h, action):

    #Default value of utility is very high so that if none of the allowd cases occur this can take place and not get selected during the max of v
    utility=-100000
    #for action=="SHOOT"
    if action==0:

        #To shoot, conditions are as follows:
        #Number of arrows >0
        
        if(a > 0):

            #To shoot, also the stamina must be either 50 or 100, so

            if(s>0):

                #The opponents heatlth can either be 25 or >25 so we have 2 cases( The case where h==0 is handled in the task functions itself)

                if(h==1):

                    #The opponent has health=25 so if this arrow hits then the game is over with a reward of 10 extra
                    prob=0.5
                    utility= (1-prob)* state_v[h,a-1,s-1] + (prob)*10

                else:

                    #The opponent cannot die hence getting reward is out of question

                    prob=0.5
                    utility=(1-prob)*state_v[h,a-1,s-1] + prob*state_v[h-1,a-1,s-1]


    #for action=="DODGE"
    if action==1:

        #To dodge the conditions are as follows:
        #if Stamina==50, then it decreases by 50
        #if Stamina==100, then it decrease by 50 or 100 with prob=0.5

        if(s>0):

            if(s==1 and a==3):
                prob=1
                utility= prob*state_v[h,a,0]
            elif(s==1 and a!=3):
                prob=0.8
                utility= prob*state_v[h,a+1,0]+(1-prob)*state_v[h,a,0]
            elif(s!=1 and a==3):
                prob1=0.8
                prob2=1
                utility=prob1*prob2*state_v[h,a,s-1]+(1-prob1)*prob2*state_v[h,a,s-2]
            elif(s!=1 and a!=3):
                prob1=0.8 
                prob2=0.8
                utility=prob1*prob2*state_v[h,a,s-1]+(1-prob1)*prob2*state_v[h,a,s-2]+prob1*(1-prob2)*state_v[h,a+1,s-1]+(1-prob1)*(1-prob2)*state_v[h,a+1,s-2]       

    if action==2:

        #To recharge the stamina increased with prob=0.8, and if the state is 100 already, then no increase in the stamina
        if(s!=2):
            prob=0.8            
            utility= prob*state_v[h,a,s+1] + (1-prob)*state_v[h,a,s]
        else:
            utility=state_v[h,a,s]


    return utility


############################################################### ALGORITHM FOR TASK 1 IS HERE ###############################################################################

def task1():
    
    global step_cost, discount, bellman, state_v
    state_v = np.zeros((5, 4, 3))

    Temp_table = np.zeros( shape=(5, 4, 3, 3) )  # 4th dimension is the actions

    for itr in range(1000 ):
        for h, a, s in states:
            for action in range( 3 ):  # produces 0,1,2 for the corresponding actions
                if(h == 0):  # skip terminal states
                    continue
                if(h!=0):  # the usual cases to handle
                    Temp_table[h, a, s, action] = step_cost + discount * Utility( s, a, h, action )

        max_table = np.max( Temp_table, axis=3 )  # getting the max utility as per actions

        action_index = np.argmax( Temp_table, axis=3 )  # getting the corresponding index numbers of the actions with max utility

        val = np.max( np.abs( max_table - state_v ) )

        state_v = max_table  # updated value table
        Print1( itr, action_index)

        if val < bellman:  # end condition for algorithm
            break


################################################# ALGORITHM FOR TASK 2 Part 1 IS HERE ##############################################################
def task2():
    
    global step_cost, discount, bellman, state_v
    step_cost = -2.5
    state_v = np.zeros((5, 4, 3))

    Answers = np.zeros( shape=(5, 4, 3, 3) )  

    for itr in range(1000 ):
        for h, a, s in states:
            for action in range( 3 ):  
                if(h == 0):
                    continue
                if(action==0 and h!=0):
                    Answers[h, a, s, action] = discount * Utility(s, a, h, action)-0.25
                if(action!=0 and h!=0): 
                    Answers[h, a, s, action] =discount * Utility( s, a, h, action ) -2.5

        action_index = np.argmax( Answers, axis=3 )  
             
        max_table = np.max( Answers, axis=3 )  
        val = np.max( np.abs( max_table - state_v ) )

        state_v = max_table  
        Print2( itr, action_index)

        if val < bellman:  
            break


################################################# ALGORITHM FOR TASK 2 Part 2 IS HERE ##############################################################
def task3():
    
    global step_cost, discount, bellman, state_v
    step_cost = -2.5
    discount = 0.1
    state_v = np.zeros((5, 4, 3))

    Temp_table = np.zeros( shape=(5, 4, 3, 3) )  # 4th dimension is the actions

    for itr in range(1000):
        for h, a, s in states:
            for action in range( 3 ):  # produces 0,1,2 for the corresponding actions
                if(h == 0):  # skip terminal states
                    continue
                if(h!=0):  # the usual cases to handle
                    Temp_table[h, a, s, action] = step_cost + discount * Utility( s, a, h, action )

        max_table = np.max( Temp_table, axis=3 )  # getting the max utility as per actions
        action_index = np.argmax( Temp_table, axis=3 )  # getting the corresponding index numbers of the actions with max utility

        val = np.max( np.abs( max_table - state_v ) )

        state_v = max_table  # updated value table
        Print3( itr, action_index)

        if val < bellman:  # end condition for algorithm
            break

################################################# ALGORITHM FOR TASK 2 Part 3 IS HERE ##############################################################
def task4():
    
    global step_cost, discount, bellman, state_v
    step_cost = -2.5
    discount = 0.1
    bellman=1e-10
    state_v = np.zeros((5, 4, 3))

    Temp_table = np.zeros( shape=(5, 4, 3, 3) )  # 4th dimension is the actions

    for itr in range(1000):
        for h, a, s in states:
            for action in range( 3 ):  # produces 0,1,2 for the corresponding actions
                if(h == 0):  # skip terminal states
                    continue
                if(h!=0):  # the usual cases to handle
                    Temp_table[h, a, s, action] = step_cost + discount * Utility( s, a, h, action )

        max_table = np.max( Temp_table, axis=3 )  # getting the max utility as per actions
        action_index = np.argmax( Temp_table, axis=3 )  # getting the corresponding index numbers of the actions with max utility

        val = np.max( np.abs( max_table - state_v ) )

        state_v = max_table  # updated value table
        Print4( itr, action_index)

        if val < bellman:  # end condition for algorithm
            break

################################################################### TASKS ARE CALLED HERE #########################################################################
task1()
task2()
task3()
task4()