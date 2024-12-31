'''终于来到了图 6：还有一副就画完了，加油坚持住！！！
图六（剂量——survival 存活曲线图），在这幅图比较简单的工作是导入文献数据，绘制散点图。（'input4.xlsx'）
自变量：剂量 D
因变量：-ln(S)

需要进行的代码操作：
0.导入 LET,YD,λ_P 并搭建成一个数组(来自文件'input1.xlsx')，随后对 LET 从小到大排序,在将组合数组拆分
1.建立一个新的 let 数组,对(LET,YD)、(LET,λ_P)进行插值计算得到新的 yd,λ_p。
2.导入'图 3 图 4 绘制’中的 Y,λ_p 函数，以及其中的参数 μ_x, μ_y, ξ, ζ,yita1, yita_wuqion，计算 let 数组对应的α_let,β_let
3.计算对应的存活函数 survival。
自变量：剂量 D
因变量：-ln(S)

本代码包含 3 大部分：文献参考数据导入、LET-survival 计算、绘图
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 图1图2绘制 import SPOWER,YD,cal_λ_p# SPOWER为'input1.xlsx'数据库数据，YD,λ_P是根据数据库计算的，使用时要舍去第一行
from 图3图4绘制 import cal_α, cal_β, μ_x, μ_y, ξ, ζ, yita1, yita2

'''文献参考数据导入，来自‘input.xlsx’'''
data = pd.read_excel('input4.xlsx')
D1 = data.iloc[1:,0].astype(float)  # 假设数据应该是浮点数类型，将其转换为浮点数
lns1 = data.iloc[1:,1].astype(float)  # 假设数据应该是浮点数类型，将其转换为浮点数
D2 = data.iloc[1:,2].astype(float)
lns2 = data.iloc[1:,3].astype(float)
D3 = data.iloc[1:7,4].astype(float)
lns3 = data.iloc[1:7,5].astype(float)


'''LET-survival 计算
0.计算不同剂量下的λ_P，spower,YD,λ_P1 并搭建成一个数组(来自文件'input1.xlsx')，随后对 LET 从小到大排序,在将组合数组拆分
1.建立一个新的 let 数组,对(spower,YD)、(spower,λ_P)进行插值计算得到新的 yd,λ_p。
2.导入'图 3 图 4 绘制’中的 Y,λ_p 函数，以及其中的参数 μ_x, μ_y, ξ, ζ,yita1, yita_wuqion，计算 let 数组对应的α_let,β_let
3.计算对应的存活函数 survival。
'''
D_let = np.arange(0.25, 6, 0.25)
λ_P = cal_λ_p(Y = YD, spower = SPOWER, D = D_let)

z = np.column_stack([SPOWER[1:], YD[1:], λ_P[1:,:]])
z_sorted = z[z[:,0].argsort()]
spower, YD, λ_P = z_sorted[:,0],z_sorted[:,1],z_sorted[:,2:]

'''#第二步 插值获取yd、λ_p'''
let = [7.7,11,20]
yd = np.interp(let, spower, YD)

# 每个剂量下对应的λ_p,需要用循环不断对λ_P的每一列进行插值
λ_p = []
for i in range(len(λ_P[0,:])):
    λ_p.append(np.interp(let, spower, λ_P[:,i]))
λ_p = np.array(λ_p).T  # 将列表转换为数组并转置，以便后续使用列索引


'''第三步 获取(yd,λ_p[:,0])、(yd,λ_p[:,1])、(yd,λ_p[:,2])对应的α_let、β_let'''
# α_let获取
α_let = np.zeros((len(yd), len(D_let)))
for i in range(len(D_let)):
    α_let[:,i] = cal_α(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, Y_1H = yd, λ_p_1H = λ_p[:,i])
# 每一个lET的剂量存活曲线占一行(α.shape =3 x 23 )

# β的循环计算
β_let = np.zeros_like(λ_p)
for j in range(len(D_let)):
    # 计算 P_contribution 和 P_track
    P_contribution = (1 - np.exp(-ζ * λ_p)) / (ζ * λ_p)
    P_track = (1 - np.exp(-ξ * λ_p)) / (ξ * λ_p)
    # η分段函数
    η = np.zeros(len(yd))
    for i in range(len(yd)):
        if yd[i] > 55:
            η[i] = yita2 / λ_p[i,j]
        else:
            η[i] = yita1 /λ_p[i,j]

        β_let[i, j] = (η[i] * yd[i] ** 2 * P_contribution[i,j] * P_track[i,j] * μ_x * μ_y * (1 - np.exp(-λ_p[i,j]))) / (2 * λ_p[i,j])
# β_let的形状也是3 x 23

#第四步 存活率对数的负数计算
lns_let = np.zeros_like(α_let)
for i in range(len(D_let)):
    lns_let[:,i] = α_let[:,i] * D_let[i] + β_let[:,i] * D_let[i]**2
print(lns_let.shape)


'''绘图'''
plt.rcParams['font.sans-serif'] = ['SimHei']
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(D1, lns1, label='7,7keV/μm')
ax.scatter(D2, lns2, label='11keV/μm')
ax.scatter(D3, lns3, label='20keV/μm')
ax.plot(D_let, lns_let[0,:], label='7,7keV/μm')
ax.plot(D_let, lns_let[1,:], label='11keV/μm')
ax.plot(D_let, lns_let[2,:], label='20keV/μm')
ax.set_xlabel('剂量(Gy)')
ax.set_ylabel('-ln(S)')
ax.text(0.5, -0.16, '图6 剂量存活曲线图', transform=ax.transAxes, ha='center')
ax.legend()
ax.grid(True)
plt.subplots_adjust(bottom=0.15)
plt.show()