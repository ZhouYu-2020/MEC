import matlab
import matlab.engine
import numpy as np
# engine = matlab.engine.start_matlab() # Start MATLAB process
# engine = matlab.engine.start_matlab("-desktop") # Start MATLAB process with graphic UI
import matplotlib.pyplot as plt


class GPD:



    def __init__(self)   :
        self.engine = matlab.engine.start_matlab()  # Start MATLAB process
        self.PARA = [[[], []] for i in range(10)]
        print("start GPD")

    def _ensure_para_size(self, index):
        while len(self.PARA) <= index:
            self.PARA.append([[], []])


    def gpd(self, data, threshold , numberOfUE):
        self._ensure_para_size(numberOfUE)

        # 取得是过去slice的倍数的值  并且超过阈值
        slice = 100
        segment = len(data) // slice
        data = data[-1 * segment * slice:]
        # print(data)
        temp = []
        left = -1 * segment * slice
        for i in range(segment):
            right = left + slice
            if right == 0:
                mid = data[left:]
            else:
                mid = data[left:right]
            if max(mid) >= threshold:
                temp.append(max(mid))
            # print(mid)
            left += slice
        print("temp",len( temp))
        if not temp:
            return
        temp = matlab.double(temp)
        # temp = []
        threshold = [4.46 * 10 ** 7]
        # threshold = threshold.tolist()
        threshold = matlab.double(threshold)
        # res  = self.engine.gpfit(temp)

        ans = self.engine.gpd(temp, threshold)
        res = ans[0][0:2]

        # print(res)
        probability = ans[0][2]
        # # print(
        #     "====================================================probability===========================================",
        #     probability)
        self.PARA[numberOfUE][0].append(res[0])
        self.PARA[numberOfUE][1].append(res[1])

        return [res , probability]
        # return [res]



if __name__ == "__main__":
    aa = GPD()
    data = [i for i in range(20)]
    threshold = 1

    res = aa.gpd(data, threshold, 0)

    print(aa.PARA)
    print(res)
