########################
####### 20250402 #######
########################



# https://blog.csdn.net/weixin_44245188/article/details/135336845
import pywt



###### 小波分解+不同的小波 ######
# (///) -- 1
def disentangle(x, w, l) :  # [B, 1, N, W]

    coef = pywt.wavedec(x, w, level=l)  # 返回level+1个数组，第一个数组为逼近系数数组，后面的依次是细节系数数组

    coefl = [coef[0]]
    for i in range(len(coef)-1) :       # len(coef)=2
        coefl.append(None)

    coefh = [None]
    for i in range(len(coef)-1) :
        coefh.append(coef[i+1])

    xl = pywt.waverec(coefl, w)  # 反向重建低频
    xh = pywt.waverec(coefh, w)  # 反向重建高频

    return xl, xh

