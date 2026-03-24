import numpy as np
import random

class MatrixGame():

    def __init__(self, actions, Q ,Qx ,Qy, Qz, M1, M2, BW, reward_mode="energy"):
        self.Q = Q  # 任务等待队列
        self.Qx = Qx  # 虚拟队列Qx
        self.Qy = Qy  # 虚拟队列Qy
        self.Qz = Qz  # 虚拟队列Qz
        self.M1 = M1
        self.M2 = M2

        self.num_ue = len(actions)
        self.BW = BW                  # BW =10MHz    带宽
        self.reward_mode = reward_mode
        self.F = 10 * pow(10, 9)                    # F =10Ghz     MEC 计算能力
        self.V = 4 * pow(10, 27)                    # V =4 (Mbits)^2/J   李雅普诺夫权衡参数
        self.tau = pow(10, -2)                      # 时隙τ，1ms
        self.kmob = pow(10, -21)                    # 本地cpu核有效电容参数
        self.kser = 0.08 * pow(10, -27)             # 服务器cpu核有效电容参数 参考【9】0.08 * pow(10,-27)
        self.Li = 737.5                       # 737.5 ,mecCPU处理密度 单位 cycles/bits
        self.t = [[] for _ in  range(self.num_ue)]                                 #记录时延
        self.delta_t = [[] for _ in  range(self.num_ue)]                            #记录时延和时延门限值的差值t_max-t
        self.g0 = pow(10, 4)                        # 信道增益 g0= -40dB
        self.N0 = pow(10, -204 / 10)                # 噪声方差 N0= -174 dBm/Hz
        self.q0 = 3.96 * pow(10, 7)                   # 队列阈值设定,单位 bits ,参考dymatic文章

        # 动作含义定义0~7
        self.action_space = [[0, 5 * pow(10, 5), 0],
                             [0, 10 * pow(10, 5), 0],
                             [0, 20 * pow(10, 5), 0],
                             [0, 30 * pow(10,5), 0],
                             [1, 0, 0.1],
                             [1, 0, 0.5],
                             [1, 0, 1],
                             [1, 0, 2]]
        # self.bn = np.random.uniform(3000, 5000, size=self.num_ue)   # 输入量 kbits    #生成num_ue个数的【300，500】浮点数的数值
        # self.dn = np.random.uniform(70, 800, size=self.num_ue)

        self.bn = np.zeros(self.num_ue)
        for i in range(len(self.bn)):                           # 每比特需要周期量 70~800 cycles/bits
            if i % 5 == 0:
                self.bn[i] = random.randint(3000,3100)*100
            if i % 5 == 1:
                self.bn[i] = random.randint(4300, 4400)*100
            if i % 5 == 2:
                self.bn[i] = random.randint(3200,3300)*100
            if i % 5 == 3:
                self.bn[i] = random.randint(4500,4600)*100
            if i % 5 == 4:
                self.bn[i] = random.randint(4900, 5000)*100

        self.dn = np.zeros(self.num_ue)
        for i in range(len(self.dn)):  # 每比特需要周期量 70~800 cycles/bits
            if i % 5 == 0:
                self.dn[i] = random.randint(70,100)
            if i % 5 == 1:
                self.dn[i] = random.randint(340, 390)
            if i % 5 == 2:
                self.dn[i] = random.randint(240,290)
            if i % 5 == 3:
                self.dn[i] = random.randint(680,730)
            if i % 5 == 4:
                self.dn[i] = random.randint(520,570)



        self.t_max = np.zeros(self.num_ue)
        for i in range(len(self.t_max)):                           # 每比特需要周期量 70~800 cycles/bits
            if i % 5 == 0:
                self.t_max[i] = 0.05
            if i % 5 == 1:
                self.t_max[i] = 0.5
            if i % 5 == 2:
                self.t_max[i] = 5
            if i % 5 == 3:
                self.t_max[i] = 0.05
            if i % 5 == 4:
                self.t_max[i] = 0.5
        
        self.lambda_n = np.zeros(self.num_ue)
        for i in range(len(self.lambda_n)):                           # 每比特需要周期量 70~800 cycles/bits
            if i % 5 == 0:
                self.lambda_n[i] = 0.001
            if i % 5 == 1:
                self.lambda_n[i] = 0.01
            if i % 5 == 2:
                self.lambda_n[i] = 0.1
            if i % 5 == 3:
                self.lambda_n[i] = 0.001
            if i % 5 == 4:
                self.lambda_n[i] = 0.01
        # self.t_max = np.array([0.05,0.5,5,0.05,0.5])              #任务产生的时间阈值  单位 秒
        # self.lambda_n = np.array([0.001,0.01,0.1,0.001,0.01])     #任务可容忍的最大队列溢出概率
        self.alpha = 1    
        self.pr_n = np.zeros(self.num_ue)
        for i in range(len(self.pr_n)):                           # 每比特需要周期量 70~800 cycles/bits
            if i % 5 == 0:
                self.pr_n[i] = 19.600
            if i % 5 == 1:
                self.pr_n[i] = 1.960
            if i % 5 == 2:
                self.pr_n[i] = 0.196
            if i % 5 == 3:
                self.pr_n[i] = 19.600
            if i % 5 == 4:
                self.pr_n[i] = 1.960                                  #可调整，是时延和可靠性指标对优先级值影响的权重参数
        # self.pr_n = np.array([19.600,1.960,0.196,19.600, 1.960])  #任务优先级值共有三种,分别为best=19.6.medium=1.96.worst=0.196
                                                                  #五个用户设备对应了相应的优先级值
        # self.rf = np.zeros(self.num_ue)                                #卸载到Mec的任务可以分配到的资源
        # self.rw = np.zeros(self.num_ue)                                #任务分配到的无线带宽

        self.t_local = 0
        self.t_ser = 0
        # self.cost_local = 0
        # self.cost_ser = 0
        # self.cost_ser1 = 0
        # self.cost_ser2 = 0


    def step(self, actions):
        getreward, getcostlocal, bn, lambda_n , rf  = self.cal_reward(actions)
        return  getreward, getcostlocal, bn, lambda_n , rf

    def cal_reward(self, actions):
        theta_pr = 0
        pr_Q = 0
        tsum = 0
        reward = np.zeros(self.num_ue)
        cost_local = 0
        cost_local_record = np.zeros(self.num_ue)

        cost_ser = 0
        cost_ser1 = 0
        cost_ser2 = 0
        rw = [0 for _ in range(self.num_ue)] 
        rf = [0 for _ in range(self.num_ue)]
        vn = np.zeros(self.num_ue)

        for i in range(self.num_ue):
            theta_pr += self.action_space[actions[i]][0] * self.pr_n[i]       # 分配到的无线带宽比例公式的分母
            pr_Q += self.pr_n[i] * (1 if self.Q[i] > 0 else 0)                   # 分配到的mec资源比例公式的分母

        for i in range(self.num_ue):

            if actions[i] < 4:  # 任务选择本地执行
                self.t_local = self.bn[i] * self.dn[i] / self.action_space[actions[i]][1]      # 本地时延f_local=self.action_space[actions[i]][1]
                # self.t[i].append(self.t_local)                                                 # 记录时延
                # self.delta_t[i].append(self.t_max[i] - self.t_local)                           # 记录时延差
                cost_local = self.kmob * self.tau * self.action_space[actions[i]][1] ** 3  # 本地能耗
                #print('local cost', cost_local)

            else:  # action[i]>=4 即4,5,6,7 任务选择卸载执行
                rw[i] = (self.BW * self.pr_n[i]) / theta_pr  
                vn[i] = rw[i] * np.log2(1 + (self.g0 * self.action_space[actions[i]][2] /
                                                    (self.N0 * rw[i])) - pow(10, -5))           # 传输速率
                cost_ser1 =  self.action_space[actions[i]][2] * self.bn[i]/ vn[i]
                # print('upload energy1', cost_ser1)
                if self.F * (self.pr_n[i] * (1 if self.Q[i] > 0 else 0)) == 0 and  pr_Q == 0: # 传输能耗公式第一部分
                    rf[i] = 0
                else:
                    rf[i] = self.F * (self.pr_n[i] * (1 if self.Q[i] > 0 else 0)) / pr_Q       # 卸载的任务分配到的MEC计算资源
                # print('resource allocation', rf[i])
                # self.t_ser = (self.bn[i] * self.vn) + (self.Q[i] * self.Li / rf[i]) + (self.bn[i] * self.dn[i] / rf[i])                               # 卸载时延???
                # self.t[i].append(self.t_ser)
                cost_ser2 = self.kser * rf[i] * rf[i] * (self.bn[i] * self.dn[i])  # 服务器能耗公式第二部分
                # print('com energy', cost_ser2)
                # self.delta_t[i].append(self.t_max[i] - self.t_ser)
                cost_ser = cost_ser1 + cost_ser2  # 卸载计算能耗
                #print('cost_ser', cost_ser)
            tsum = self.bn[i] * self.action_space[actions[i]][0] * ((1 + self.Qx[i] -self.q0 * self.lambda_n[i]) * self.Q[i] - self.Qx[i] * self.lambda_n[i]+ (2 * (self.Q[i] - self.q0) * (self.Qz[i] + (self.Q[i] - self.q0) * (self.Q[i] - self.q0) - self.M2[i]) + self.Q[i] + self.Qy[i] - self.M1[i])* (1 if self.Q[i] > self.q0 else 0))
            if self.reward_mode == "lyapunov":
                lyapunov_term = (tsum / (10 ** 25)) + (self.V * (cost_ser + cost_local) / (10 ** 25))
                reward[i] = -lyapunov_term
            else:
                reward[i] = - (cost_ser + cost_local)
            cost_local_record[i] = cost_local



            # print('action:', actions[i])
            # print('reward:', reward[i])

        return np.array(reward),np.array(cost_local_record) ,self.bn, self.lambda_n, rf









