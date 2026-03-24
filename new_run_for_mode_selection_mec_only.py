import numpy as np
import matplotlib.pyplot as plt
from wolf_agent import WoLFAgent 
# from matrix_game import MatrixGame
import pandas as pd
from matrix_game_mec_only import MatrixGame_mec
from queue_relay import QueueRelay

from dataToExcel import DTE    ##  TLIU
from gpd import GPD     ##  TLIU
import xlrd      ##  TLIU
import xlsxwriter    ##  TLIU

if __name__ == '__main__':
    OUTPUT = []  #
    gpdaa = GPD()
    nb_episode = 2000
    actions_set = [
                   [1, 0, 0.1],
                   [1, 0, 0.2],
                   [1, 0, 0.3],
                   [1, 0, 0.4]]
    actions = np.arange(len(actions_set))
    user_num = 10
    lambda_n = np.zeros(user_num)
    for i in range(user_num):                           # 每比特需要周期量 70~800 cycles/bits
        if i % 5 == 0:
            lambda_n[i] = 0.001
        if i % 5 == 1:
            lambda_n[i] = 0.01
        if i % 5 == 2:
            lambda_n[i] = 0.1
        if i % 5 == 3:
            lambda_n[i] = 0.001
        if i % 5 == 4:
            lambda_n[i] = 0.01
    # actions_set = [[0, 5 * pow(10, 6), 0.4],
    #      [0, 5 * pow(10, 6), 0.4],
    #      [0, 5 * pow(10, 6), 0.4],
    #      [0, 5 * pow(10, 6), 0.4],
    #      [1, 0, 0.4],
    #      [1,0, 0.4],
    #      [1, 0, 0.4],
    #      [1, 0, 0.4]]

    GPD1_array = [4 * pow(10, 6) for _ in range(user_num)]
    GPD2_array = [0.3 for _ in range(user_num)]

    #init wolf agent 
    wolf_agent_array = []
    for i in range(user_num):
        wolf_agent_array.append(WoLFAgent(alpha=0.1, actions=actions, high_delta=0.004, low_delta=0.002))

    
    queue_relay_array = []

    for i in range(user_num):
        queue_relay_array.append(QueueRelay(lambda_n[i], GPD1_array[i], GPD2_array[i]))
    
    #set reward functio

    # reward = Reward()
    reward_history  = []
    #init_Queue_relay
    
    Q_array_histroy = [  [10] for i in range(user_num)  ]     ##  TLIU

    for episode in range(nb_episode):
        print("episode", episode)
        Q_array = []
        Qx_array = []
        Qy_array = []
        Qz_array = []
        M1_array = []
        M2_array = []

        for i in range(user_num):
            Q_array.append(queue_relay_array[i].Q)
            Qx_array.append(queue_relay_array[i].Qx)
            Qy_array.append(queue_relay_array[i].Qy)
            Qz_array.append(queue_relay_array[i].Qz)
            M1_array.append(queue_relay_array[i].M1)
            M2_array.append(queue_relay_array[i].M2)

        ##  TLIU
        for i in range(user_num):
            Q_array_histroy[i].append(Q_array[i])
        if episode % 50 == 0  and episode != 0  :
            for i in range(user_num):

                data = Q_array_histroy[i]
                # data = [10000000000000 for i in range(200) ]
                # res = aa.gpd(  data  , 3.96*pow(10,5)  )
                res = gpdaa.gpd(  data  , 3.96 * pow(10, 6) ,i   )
                if res :
                    queue_relay_array[i].GPD1 = res[0][0]
                    queue_relay_array[i].GPD2 = res[0][1]
                    queue_relay_array[i].updateM1()
                    queue_relay_array[i].updateM2()
        ##  TLIU


        iteration_actions = []
        for i in range(user_num):
            iteration_actions.append(wolf_agent_array[i].act())
        game = MatrixGame_mec(actions=iteration_actions, Q=Q_array,
                  Qx=Qx_array, Qy=Qy_array, Qz=Qz_array,
                  M1=M1_array,
                  M2=M2_array,BW = 10 * pow(10, 6) )

        #print('Q value :'+str(Q_array)+str(Qx_array)+str(Qy_array)+str(Qz_array))

        reward, bn, lumbda, rff = game.step(actions=iteration_actions)
        print("episode", episode, "reward", sum(reward))
        OUTPUT.append(sum(reward))

        for i in range(user_num):
            #wolf agent act
            # update_Queue_relay
            queue_relay_array[i].lumbda = lumbda[i]
            queue_relay_array[i].updateQ(bn[i], actions_set[iteration_actions[i]][0], rff[i])
            queue_relay_array[i].updateQx()
            queue_relay_array[i].updateQy()
            queue_relay_array[i].updateQz()
                
        # reward step
        reward_history.append(sum(reward))
        for i in range(user_num):
            wolf_agent_array[i].observe(reward=reward[i])

    for i in range(user_num):
        print(wolf_agent_array[i].pi_average)
    plt.plot(np.arange(len(reward_history)), reward_history, label="only mec")
    plt.show()

    data = DTE("./picture/pic1/mec")   ##  TLIU
    print(OUTPUT)
    data.write(OUTPUT)










