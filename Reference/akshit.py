import numpy as np
import os

# U(t+1)(i) = max_A[R(i,A) + gamma * SIGMA [P(j| i,A) * U(t)(j)]]
# P(t+1)(i) = argmax_A[R(i,A) + gamma * SIGMA [P(j| i,A) * U(t+1)(j)]]


if(os.path.exists("./outputs")):

    open('./outputs/task_1_trace.txt', 'w').close()
    open('./outputs/task_2_part_1_trace.txt', 'w').close()
    open('./outputs/task_2_part_2_trace.txt', 'w').close()
    open('./outputs/task_2_part_3_trace.txt', 'w').close()

else:

    os.mkdir("./outputs")

Penalty = -20
Penalty_shoot = -20
Gamma = 0.99
Delta = 1e-3
Final_reward = 10

max_MD_health = 4
max_arrows_cnt = 3
max_hero_stamina = 2 


Ucurr = np.zeros((5,4,3))
Uprev = np.zeros((5,4,3))
Reward = np.zeros((5,4,3))
Reward_shoot = np.zeros((5,4,3))
Policy = np.array([[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']]],dtype = 'object')

for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward[i][j][k] = Penalty
            else:
                Reward[i][j][k] = Penalty + Final_reward

for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward_shoot[i][j][k] = Penalty_shoot
            else:
                Reward_shoot[i][j][k] = Penalty_shoot + Final_reward

def reward_shoot(MD_health , arrows_cnt , hero_stamina):
    
    prob_hit = 0.5
    return (prob_hit*Reward_shoot[MD_health-1][arrows_cnt-1][hero_stamina-1] + (1-prob_hit)*Reward_shoot[MD_health][arrows_cnt-1][hero_stamina-1])

def reward_dodge(MD_health , arrows_cnt , hero_stamina):
    
    if (hero_stamina == 2 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 0.8
        prob_pick_arrow = 0.8
        return (prob_stamina_reduce_by_50*prob_pick_arrow*Reward[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*prob_pick_arrow*Reward[MD_health][arrows_cnt+1][hero_stamina-2] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-2])          

    elif (hero_stamina == 2 and arrows_cnt == 3):
        prob_pick_arrow = 0
        prob_stamina_reduce_by_50 = 0.8
        return (prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-2])          

    elif (hero_stamina == 1 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0.8
        return (prob_stamina_reduce_by_50*prob_pick_arrow*Reward[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1])          

    elif (hero_stamina ==1 and arrows_cnt == 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0  
        return (prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1])

def reward_recharge(MD_health , arrows_cnt , hero_stamina):

    if (hero_stamina == 2):
        prob_recharge = 0
        return ((1-prob_recharge)*Reward[MD_health][arrows_cnt][hero_stamina])
    
    elif (hero_stamina < 2):
        prob_recharge = 0.8
        return (prob_recharge*Reward[MD_health][arrows_cnt][hero_stamina+1] + (1-prob_recharge)*Reward[MD_health][arrows_cnt][hero_stamina])


def utility_shoot(Uprev , MD_health , arrows_cnt , hero_stamina):

    prob_hit = 0.5

    if (arrows_cnt > 0 and hero_stamina > 0):
        return (reward_shoot(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_hit*Uprev[MD_health-1][arrows_cnt-1][hero_stamina-1] + (1-prob_hit)*Uprev[MD_health][arrows_cnt-1][hero_stamina-1]))

def utility_dodge(Uprev, MD_health , arrows_cnt , hero_stamina):
    
    if (hero_stamina == 2 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 0.8
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Uprev[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*prob_pick_arrow*Uprev[MD_health][arrows_cnt+1][hero_stamina-2] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 2 and arrows_cnt == 3):
        prob_pick_arrow = 0
        prob_stamina_reduce_by_50 = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 1 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Uprev[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1]))           

    elif (hero_stamina ==1 and arrows_cnt == 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0  
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1]))


def utility_recharge(Uprev, MD_health , arrows_cnt , hero_stamina):

    if (hero_stamina == 2):
        prob_recharge = 0
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*((1-prob_recharge)*Uprev[MD_health][arrows_cnt][hero_stamina]))
    
    elif (hero_stamina < 2):
        prob_recharge = 0.8
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_recharge*Uprev[MD_health][arrows_cnt][hero_stamina+1] + (1-prob_recharge)*Uprev[MD_health][arrows_cnt][hero_stamina]))

def policy_shoot(Ucurr, MD_health , arrows_cnt , hero_stamina):

    prob_hit = 0.5

    if (arrows_cnt > 0 and hero_stamina > 0):
        return (reward_shoot(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_hit*Ucurr[MD_health-1][arrows_cnt-1][hero_stamina-1] + (1-prob_hit)*Ucurr[MD_health][arrows_cnt-1][hero_stamina-1]))

def policy_dodge(Ucurr , MD_health , arrows_cnt , hero_stamina):
    
    if (hero_stamina == 2 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 0.8
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Ucurr[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*prob_pick_arrow*Ucurr[MD_health][arrows_cnt+1][hero_stamina-2] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 2 and arrows_cnt == 3):
        prob_pick_arrow = 0
        prob_stamina_reduce_by_50 = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 1 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Ucurr[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1]))           

    elif (hero_stamina ==1 and arrows_cnt == 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0  
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1]))          


def policy_recharge(Ucurr , MD_health , arrows_cnt , hero_stamina):

    if (hero_stamina == 2):
        prob_recharge = 0
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*((1-prob_recharge)*Ucurr[MD_health][arrows_cnt][hero_stamina]))
    
    elif (hero_stamina < 2):
        prob_recharge = 0.8
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_recharge*Ucurr[MD_health][arrows_cnt][hero_stamina+1] + (1-prob_recharge)*Ucurr[MD_health][arrows_cnt][hero_stamina]))

def can_shoot(MD_health , arrows_cnt , hero_stamina):
    if (arrows_cnt > 0 and hero_stamina > 0):
        return 1
    else:
        return 0

def can_dodge(MD_health , arrows_cnt , hero_stamina):
    if (hero_stamina > 0):
        return 1
    else:
        return 0

def print_iteration(Ucurr , Policy , iteration_cnt,num):

    # sr1="\n\n"
    sr=('iteration=' + str(iteration_cnt)+'\n')
    if(num==0):
        f=open("./outputs/task_1_trace.txt","a")
        f.write(sr)
        f.close()
    elif(num==1):
        f=open("./outputs/task_2_part_1_trace.txt","a")
        f.write(sr)
        f.close()                
    elif(num==2):
        f=open("./outputs/task_2_part_2_trace.txt","a")
        f.write(sr)
        f.close()
    else:
        f=open("./outputs/task_2_part_3_trace.txt","a")
        f.write(sr)
        f.close()

    iteration_cnt += 1
    for i in range(max_MD_health + 1):
        for j in range(max_arrows_cnt + 1):
            for k in range(max_hero_stamina + 1):
                sr=('(' + str(i) + ',' + str(j) + ',' + str(k) + ')' + ':' + str(Policy[i][j][k]) + '=' + '[' + str(round(Ucurr[i][j][k],3)) + ']\n')     

                if(num==0):
                    f=open("./outputs/task_1_trace.txt","a")
                    f.write(sr)
                    f.close()
                elif(num==1):
                    f=open("./outputs/task_2_part_1_trace.txt","a")
                    f.write(sr)
                    f.close()                
                elif(num==2):
                    f=open("./outputs/task_2_part_2_trace.txt","a")
                    f.write(sr)
                    f.close()
                else:
                    f=open("./outputs/task_2_part_3_trace.txt","a")
                    f.write(sr)
                    f.close()

    sr=('\n\n')
    if(num==0):
        f=open("./outputs/task_1_trace.txt","a")
        f.write(sr)
        f.close()
    elif(num==1):
        f=open("./outputs/task_2_part_1_trace.txt","a")
        f.write(sr)
        f.close()                
    elif(num==2):
        f=open("./outputs/task_2_part_2_trace.txt","a")
        f.write(sr)
        f.close()
    else:
        f=open("./outputs/task_2_part_3_trace.txt","a")
        f.write(sr)
        f.close()

def value_iteration(Uprev , Ucurr , iteration_cnt,num):
    
    for i in range (max_MD_health + 1):
        for j in range(max_arrows_cnt + 1):
            for k in range(max_hero_stamina + 1):
                if (i > 0):
                    if(can_dodge(i,j,k) and can_shoot(i,j,k)):
                        Ucurr[i][j][k] = max(utility_shoot(Uprev,i,j,k) , utility_dodge(Uprev,i,j,k) , utility_recharge(Uprev,i,j,k))

                    elif(can_shoot(i,j,k)):
                        Ucurr[i][j][k] = max(utility_shoot(Uprev,i,j,k) , utility_recharge(Uprev,i,j,k))

                    elif(can_dodge(i,j,k)):
                        Ucurr[i][j][k] = max(utility_dodge(Uprev,i,j,k) , utility_recharge(Uprev,i,j,k))

                    else:
                        Ucurr[i][j][k] = utility_recharge(Uprev,i,j,k)
                else:
                    Ucurr[i][j][k] = 0

    for i in range (max_MD_health + 1):
        for j in range(max_arrows_cnt + 1):
            for k in range(max_hero_stamina + 1):
                if (i > 0):
                    if(can_dodge(i,j,k) and can_shoot(i,j,k)):
                        policy = [policy_shoot(Ucurr,i,j,k),policy_dodge(Ucurr,i,j,k),policy_recharge(Ucurr,i,j,k)]
                        index = policy.index(max(policy))
                        if(index == 0):
                            Policy[i][j][k] = 'SHOOT'
                        elif(index == 1):
                            Policy[i][j][k] = 'DODGE'
                        elif(index == 2):
                            Policy[i][j][k] = 'RECHARGE'

                    elif(can_shoot(i,j,k)):
                        policy = [policy_shoot(Ucurr,i,j,k),policy_recharge(Ucurr,i,j,k)]
                        index = policy.index(max(policy))
                        if(index == 0):
                            Policy[i][j][k] = 'SHOOT'
                        if(index == 1):
                            Policy[i][j][k] = 'RECHARGE'

                    elif(can_dodge(i,j,k)):
                        policy = [policy_dodge(Ucurr,i,j,k),policy_recharge(Ucurr,i,j,k)]
                        index = policy.index(max(policy))
                        if(index == 0):
                            Policy[i][j][k] = 'DODGE'
                        elif(index == 1):
                            Policy[i][j][k] = 'RECHARGE'
                    else:
                        Policy[i][j][k] = 'RECHARGE'    
                else:
                    Policy[i][j][k] = '-1'                

    print_iteration(Ucurr , Policy , iteration_cnt,num)


def check(Uprev, Ucurr):

    cnt = 0
    for i in range(max_MD_health + 1):
        for j in range(max_arrows_cnt + 1):
            for k in range(max_hero_stamina + 1):
                if (abs(Ucurr[i][j][k] - Uprev[i][j][k]) < Delta):
                    cnt += 1

    return(cnt)

def update(Uprev, Ucurr):
    for i in range(max_MD_health + 1):
        for j in range(max_arrows_cnt + 1):
            for k in range(max_hero_stamina + 1):
                Uprev[i][j][k] = Ucurr[i][j][k]    


iteration_cnt = 0
value_iteration(Uprev,Ucurr,iteration_cnt,0)

while(1):
    
    cnt = check(Uprev,Ucurr)

    if(cnt == 60):
        break
    else:
        iteration_cnt += 1
        update(Uprev,Ucurr)
        value_iteration(Uprev,Ucurr,iteration_cnt,0)    

# print("*********************************************************NEXT***********************************************")


Penalty=-2.5
Penalty_shoot=-0.25

Ucurr = np.zeros((5,4,3))
Uprev = np.zeros((5,4,3))
Reward = np.zeros((5,4,3))
Reward_shoot = np.zeros((5,4,3))
Policy = np.array([[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']]],dtype = 'object')


for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward[i][j][k] = Penalty
            else:
                Reward[i][j][k] = Penalty + Final_reward

for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward_shoot[i][j][k] = Penalty_shoot
            else:
                Reward_shoot[i][j][k] = Penalty_shoot + Final_reward

iteration_cnt = 0
value_iteration(Uprev,Ucurr,iteration_cnt,1)

while(1):
    
    cnt = check(Uprev,Ucurr)

    if(cnt == 60):
        break
    else:
        iteration_cnt += 1
        update(Uprev,Ucurr)
        value_iteration(Uprev,Ucurr,iteration_cnt,1)    

# print("*********************************************************NEXT***********************************************")

Penalty=-2.5
Penalty_shoot=-2.5
Gamma=0.1

Ucurr = np.zeros((5,4,3))
Uprev = np.zeros((5,4,3))
Reward = np.zeros((5,4,3))
Reward_shoot = np.zeros((5,4,3))
Policy = np.array([[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']]],dtype = 'object')


for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward[i][j][k] = Penalty
            else:
                Reward[i][j][k] = Penalty + Final_reward

for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward_shoot[i][j][k] = Penalty_shoot
            else:
                Reward_shoot[i][j][k] = Penalty_shoot + Final_reward

iteration_cnt = 0
value_iteration(Uprev,Ucurr,iteration_cnt,2)

while(1):
    
    cnt = check(Uprev,Ucurr)

    if(cnt == 60):
        break
    else:
        iteration_cnt += 1
        update(Uprev,Ucurr)
        value_iteration(Uprev,Ucurr,iteration_cnt,2)    

# print("*********************************************************NEXT***********************************************")

Delta=1e-10 

Ucurr = np.zeros((5,4,3))
Uprev = np.zeros((5,4,3))
Reward = np.zeros((5,4,3))
Reward_shoot = np.zeros((5,4,3))
Policy = np.array([[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']]],dtype = 'object')


for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward[i][j][k] = Penalty
            else:
                Reward[i][j][k] = Penalty + Final_reward

for i in range(max_MD_health + 1):
    for j in range(max_arrows_cnt + 1):
        for k in range(max_hero_stamina + 1):
            if(i != 0):
                Reward_shoot[i][j][k] = Penalty_shoot
            else:
                Reward_shoot[i][j][k] = Penalty_shoot + Final_reward

iteration_cnt = 0
value_iteration(Uprev,Ucurr,iteration_cnt,3)

while(1):
    
    cnt = check(Uprev,Ucurr)

    if(cnt == 60):
        break
    else:
        iteration_cnt += 1
        update(Uprev,Ucurr)
        value_iteration(Uprev,Ucurr,iteration_cnt,3)    

# print("*********************************************************NEXT***********************************************")


