import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from 图5绘制 import survival_sanwei, D, spower#输出时已经按照spower由小到大的顺序排序了

'0.定义存活函数矩阵survvival'
survival = [0.01, 0.05, 0.1, 0.5]# 存活率
lnS = [-math.log(i) for i in survival]# 存活率对数的负数


'''一、参考数据的导入（''input5.xlsx'',CO存活-lnS_60Co随LET_60Co变化），计算存活率0.01, 0.05, 0.1, 0.5时，对应的60CO剂量D1_survival、D2_survival、D3_survival
1.导入数据：将每一个数据组合按照lns1，lns2、lns3由大到小排列
2.代入survival，拟合插值求取剂量D'''

data = pd.read_excel('input5.xlsx')

D1 = data.iloc[:, 0]
lns1 = data.iloc[:, 1]
z = np.column_stack([D1,lns1])
z_sorted =z[z[:,1].argsort()]
D1,lns1 = z_sorted[:,0],z_sorted[:,1]

D2 = data.iloc[:41, 2]
lns2 = data.iloc[:41, 3]
z = np.column_stack([D2,lns2])
z_sorted =z[z[:,1].argsort()]
D2,lns2 = z_sorted[:,0],z_sorted[:,1]

D3 = data.iloc[:36, 4]
lns3 = data.iloc[:36, 5]
z = np.column_stack([D3,lns3])
z_sorted =z[z[:,1].argsort()]
D3,lns3 = z_sorted[:,0],z_sorted[:,1]

D4 = data.iloc[:26, 6]
lns4 = data.iloc[:26, 7]
z = np.column_stack([D4,lns4])
z_sorted =z[z[:,1].argsort()]
D4,lns4 = z_sorted[:,0],z_sorted[:,1]

# 使用 scipy.interpolate.interp1d 进行插值，得到文献存活率对应的存活剂量，每个都应该有3个（代表存活率0.01, 0.05, 0.1, 0.5对应的剂量）
'''
D1_survival = interp1d(lns1,D1)(survival)
D2_survival = interp1d(lns2,D2)(survival)
D3_survival = interp1d(lns3,D3)(survival)
'''
D4_survival = interp1d(lns4,D4)(survival)

'''二、存活概率为0.01, 0.05, 0.1, 0.5时，循环代入应的LET_1H中(survival_1H,D_1H)进行插值计算，可以得到每一个LET对应的在某一存活概率时
对应的剂量D_1H(286x3),如果在某一LET不存在那样的点，就在对应的LET填入NAN
0.导入'图5绘制'代码中的survival_sanwei,以及D（形状286x9,对应的剂量D是1-9Gy）,spower.
注意：图五绘制中的代码已经让survival_sanwei按照spower由小到大的顺序排列
1.循环插值计算D_survivl[i,:]=interp1d(survival_sanwei[i,:],D)(sruvival)
2.插值计算之前，判断survival_sanwei在不在插值的范围，如果超出范围，在该点输入NAN
3.最终得到格式287x3的D_survival矩阵'''
#survival在图5绘制中是存活率的对数的负数
spower = spower[:220]
D_survival = np.zeros((len(spower), len(lnS)))
for i in range(len(D)):
    for j in range(len(spower)):
        for k in range(len(lnS)):
            if np.min(survival_sanwei[j, :]) <= lnS[k] <= np.max(survival_sanwei[j, :]):
                # 提取当前行的 survival 值
                curr_survival = survival_sanwei[j, :]
                # 获取排序后的索引
                sorted_indices = np.argsort(curr_survival)
                # 根据索引排序 survival 和 D
                sorted_survival = curr_survival[sorted_indices]
                sorted_dose = D[sorted_indices]
                D_survival[j,k] = interp1d(sorted_survival,sorted_dose)(lnS[k])
            else:
                D_survival[j, k] = np.nan
print(D_survival.shape)
np.savetxt('D_survival',D_survival)


'''三、不同存活率下，RBE随lET变化
0.自变量、因变量的明确。
自变量：spower
因变量：
存活率0.01时，60Co对应的剂量为D1_survival[1](出束口)、D2_survival[1](顶峰)、D3_survival[1](SOBP)、D4_survival[1](42),1H对应的剂量为D_survival[1,:]
存活率0.05时，60Co对应的剂量为D1_survival[2](出束口)、D2_survival[1](顶峰)、D3_survival[1](SOBP)、D4_survival[1](42),1H对应的剂量为D_survival[2,:]
存活率0.1时，60Co对应的剂量为D1_survival[3](出束口)、D2_survival[2](顶峰)、D3_survival[2](SOBP)、D4_survival[1](42),1H对应的剂量为D_survival[3, :]
存活率0.5时，60Co对应的剂量为D1_survival[4](出束口)、D2_survival[3](顶峰)、D3_survival[3](SOBP)、D4_survival[1](42),1H对应的剂量为D_survival[4,:]
1.通过循环绘制曲线图，如果遇到因变量为nan,则不画
RBE1[i,j] = D_survival[i,:] / D1_survival[j] #出束口RBE：第1列存活概率0.01, 0.05, 0.1, 0.5
RBE2[i,j] = D_survival[i,:] / D2_survival[j] #顶峰RBE：第2列存活概率0.01, 0.05, 0.1, 0.5
RBE3[i,j] = D_survival[i,:] / D3_survival[j] #SOBP RBE：第3列存活概率0.01, 0.05, 0.1, 0.5
RBE4[i,j] = D_survival[i,:] / D4_survival[j] #42 RBE：第4列存活概率0.01, 0.05, 0.1, 0.5
'''
'''
# RBE_出束[i,j] = D_survival[i,:] / D1_survival[j] #出束口RBE：第1列存活概率0.01, 0.05, 0.1, 0.5
RBE1 = np.zeros((len(spower),len(survival)))
for i in range(len(survival)):
    for j in range(len(spower)):
        if np.any(np.isnan(D_survival[j, i])):
            RBE1[j, i] = np.nan
        else:
            RBE1[j, i] = D_survival[j, i] / D1_survival[i]


# RBE_顶峰[i,j] = D_survival[i,:] / D2_survival[j] #顶峰RBE：第2列存活概率0.01, 0.05, 0.1, 0.5
RBE2 = np.zeros((len(spower),len(survival)))
for i in range(len(survival)):
    for j in range(len(spower)):
        if np.any(np.isnan(D_survival[j, i])):
            RBE2[j, i] = np.nan
        else:
            RBE2[j, i] = D_survival[j, i] / D2_survival[i]

# RBE_SOBP[i,j] = D_survival[i,:] / D3_survivval[j] #SOBP RBE：第3列存活概率0.01, 0.05, 0.1, 0.5
RBE3 = np.zeros((len(spower),len(survival)))
for i in range(len(survival)):
    for j in range(len(spower)):
        if np.any(np.isnan(D_survival[j, i])):
            RBE3[j, i] = np.nan
        else:
            RBE3[j, i] = D_survival[j, i] / D3_survival[i]
'''
# RBE4[i,j] = D_survival[i,:] / D4_survival[j] #42 RBE：第4列存活概率0.01, 0.05, 0.1, 0.5
RBE4 = np.zeros((len(spower),len(survival)))
for i in range(len(survival)):
    for j in range(len(spower)):
        if np.any(np.isnan(D_survival[j, i])):
            RBE4[j, i] = np.nan
        else:
            RBE4[j,i] = D_survival[j, i] / D4_survival[i]
print('D4_survival',D4_survival)
np.savetxt('RBE4',RBE4)

# 循环绘图
'''
# 绘制出束口RBE随LET变化散点图，纵坐标：RBE，横坐标：LET(keV/μm) [LET = spower],存活率RBE1[:,0]——1%、RBE1[:,1]——5%、RBE1[:,2]——10%、RBE1[:,3]——50%
plt.plot(spower, RBE1[:, 0], drawstyle='steps-post', label='1% Survival')
plt.plot(spower, RBE1[:, 1], drawstyle='steps-post', label='5% Survival')
plt.plot(spower, RBE1[:, 2], drawstyle='steps-post', label='10% Survival')
plt.plot(spower, RBE1[:, 3], drawstyle='steps-post', label='50% Survival')
plt.xlabel('LET (keV/μm)')
plt.ylabel('RBE')
plt.subplots_adjust(bottom=0.15)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('Outgoing Beam RBE vs LET', ha='center')
plt.legend()
plt.show()


# 绘制顶峰RBE随LET变化散点图，纵坐标：RBE，横坐标：LET(keV/μm) [LET = spower],存活率RBE2[:,0]——1%、RBE2[:,1]——5%、RBE2[:,2]——10%、RBE2[:,3]——50%
plt.plot(spower, RBE2[:, 0], drawstyle='steps-post', label='1% Survival')
plt.plot(spower, RBE2[:, 1], drawstyle='steps-post', label='5% Survival')
plt.plot(spower, RBE2[:, 2], drawstyle='steps-post', label='10% Survival')
plt.plot(spower, RBE2[:, 3], drawstyle='steps-post', label='50% Survival')
plt.xlabel('LET (keV/μm)')
plt.ylabel('RBE')
plt.subplots_adjust(bottom=0.15)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('Peak Beam RBE vs LET', ha='center')
plt.legend()
plt.show()


# 绘制SOBP的RBE随LET变化散点图，纵坐标：RBE，横坐标：LET(keV/μm) [LET = spower],存活率RBE3[:,0]——1%、RBE3[:,1]——5%、RBE3[:,2]——10%、RBE3[:,3]——50%
plt.plot(spower, RBE3[:, 0], drawstyle='steps-post', label='1% Survival')
plt.plot(spower, RBE3[:, 1], drawstyle='steps-post', label='5% Survival')
plt.plot(spower, RBE3[:, 2], drawstyle='steps-post', label='10% Survival')
plt.plot(spower, RBE3[:, 3], drawstyle='steps-post', label='50% Survival')
plt.xlabel('LET (keV/μm)')
plt.ylabel('RBE')
plt.subplots_adjust(bottom=0.15)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('SOBP Beam RBE vs LET', ha='center')
plt.legend()
plt.show()
'''

# 绘制42的RBE随LET变化散点图，纵坐标：RBE，横坐标：LET(keV/μm) [LET = spower],存活率RBE3[:,0]——1%、RBE3[:,1]——5%、RBE3[:,2]——10%、RBE3[:,3]——50%
plt.plot(spower, RBE4[:, 0], drawstyle='steps-post', label='1% Survival')
plt.plot(spower, RBE4[:, 1], drawstyle='steps-post', label='5% Survival')
plt.plot(spower, RBE4[:, 2], drawstyle='steps-post', label='10% Survival')
plt.plot(spower, RBE4[:, 3], drawstyle='steps-post', label='50% Survival')
plt.xlabel('LET (keV/μm)')
plt.ylabel('RBE')
plt.subplots_adjust(bottom=0.15)
plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.title('Beam RBE of ref 42 vs LET', ha='center')
plt.legend()
plt.show()
