'''图5：计算不同剂量D对应的YD、λ_p，进而得到对应的α_sanwei、β_sanwei、suivival_sanwei
自变量：D(自定义)，得到不同剂量对应的YD、λ_p(回到'图1图2绘制'查看默认设置)
因变量：运行α_Greywolf()、β_Greywolf()得到最优参数，对应的YD、λ_p代入计算 α_sanwei、β_sanwei、suivival_sanwei
μ_x, μ_y, ξ, ζ = α_Greywolf()
α_predict = α(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, Y_1H = YD, λ_p_1H = λ_p)
yita1, yita_wuqion = β_Greywolf()
β_predict = β(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, yita1 = yita1, yita_wuqion = yita_wuqion, Y = YD, landa_p = λ_p)
本代码大概分为5部分：YD、λ_p的求取，α_sanwei的求取，β_sanwei求取，suivival_sanwei求取，三维绘图
'''
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from 图1图2绘制 import cal_Y,cal_λ_p
from 图3图4绘制 import α_Greywolf,cal_α,β_Greywolf,cal_β
#from 图3图4绘制 import α, β, μ_x, μ_y, ξ, ζ,yita1, yita2


'''一、YD、λ_p的求取
0.导入LET（'input1.xlsx')
1.定义剂量D的范围
2.引入YD = Y(D = D), λ_P = λ_P(Y = Y,D = D)求取YD、λ_p，'''

# 导入数据
data = pd.read_excel('input1.xlsx')
spower = data.iloc[:,2]

# 定义剂量
D = np.arange(1, 25, 0.25)

# 细胞核每剂量DSB产量YD(DSB/Gy/nuc)、每个粒子平均DSB产额λ_p的求取，
# 数据来自数据库'input1.xlsx',使用时需要去掉第一行
YD = cal_Y()
λ_p = cal_λ_p(Y = YD, spower = spower, D = D)
YD1 = YD[1:]
λ_p1 = λ_p[1:,:]


'''二、α的计算
0.导入'图3图4绘制.py'文件中的α_Greywolf、α函数，
1.直接运行α_Greywolf,得到参数 μ_x, μ_y, ξ, ζ = α_Greywolf()
2.将计算出来的μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ以及YD(287x9)、λ_p(287x9)中每一列循环代入计算α
3.将计算的结果加入α数组中
'''
# 参数的计算
μ_x, μ_y, ξ, ζ = α_Greywolf()

# 定义一个循环计算α_sanwei
α_sanwei = np.zeros((len(YD1), len(D)))
for i in range(len(D)):
    α_sanwei[:,i] = cal_α(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, Y_1H = YD1, λ_p_1H = λ_p1[:, i])
    #使用YD、λ_p_1H时需要去掉第一行



'''三、 β的计算
0.导入'图3图4绘制.py'文件中的β_Greywolf、β函数，
1.直接运行β_Greywolf,得到参数 yita1, yita_wuqion = β_Greywolf()
2.将计算出来的μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, yita1 = yita1, yita2 = yita2以及YD(287x9)、λ_p(287x9)中每一列循环代入计算β_sanwei
3.将计算的结果加入β_sanwei数组中
'''
# 参数的计算
z = β_Greywolf(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ)
yita1, yita2 = z
# yita1 = z

# 循环计算
β_sanwei = np.zeros_like(λ_p1)
for j in range(len(D)):
    # 计算 P_contribution 和 P_track
    P_contribution = (1 - np.exp(-ζ * λ_p1)) / (ζ * λ_p1)
    P_track = (1 - np.exp(-ξ * λ_p1)) / (ξ * λ_p1)
    # η分段函数
    η = np.zeros(len(YD1))
    for i in range(len(YD1)):
        if YD1.iloc[i] > 55:
            η[i] = yita2 / λ_p1[i,j]
        else:
            η[i] = yita1 / λ_p1[i,j]

        β_sanwei[i, j] = (η[i] * YD1.iloc[i] ** 2 * P_contribution[i,j] * P_track[i,j] * μ_x * μ_y * (1 - np.exp(-λ_p1[i,j]))) / (2 * λ_p1[i,j])

# 保存结果
np.savetxt('β_sanwei.txt', β_sanwei)


'''四、 survival-sanwei绘图
0.survival-sanwei的计算 '''

survival_sanwei = np.zeros_like(α_sanwei)
for i in range(len(D)):
    survival_sanwei[:,i] = α_sanwei[:,i] * D[i] + β_sanwei[:,i] * D[i]**2

np.savetxt('survival_sanwei.txt',survival_sanwei)
print(YD1.shape, λ_p1.shape, α_sanwei.shape, β_sanwei.shape, survival_sanwei.shape)


# 五、 绘制三维图形
# 绘制 3D 曲面图，首先让spower、α_sanwei、β_sanwei、survival_sanwei按照spower的顺序由小到大排序
z = np.column_stack([spower[1:], α_sanwei, β_sanwei, survival_sanwei])
z_sorted = z[z[:, 0].argsort()]
spower, α_sanwei, β_sanwei, survival_sanwei = z_sorted[:,0],z_sorted[:,1:len(D)+1], z_sorted[:,len(D)+1:2*len(D)+1], z_sorted[:,2*len(D)+1:3*len(D)+1]

# 创建spower、D的网格
Dose, LET = np.meshgrid(D, spower)


#α三维图绘制
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(Dose, LET, α_sanwei, cmap='viridis', label='α')
ax1.set_xlabel('Dose')
ax1.set_ylabel('LET')
ax1.set_zlabel('α')
#ax1.set_title('α Performance with LET and Dose', x=0.5, y=-0.16, transform=ax1.transAxes, ha='center')
ax1.legend()
plt.subplots_adjust(bottom=0.15)

# β三维图绘制
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(Dose, LET, β_sanwei, cmap='plasma', label='β')
ax2.set_xlabel('Dose')
ax2.set_ylabel('LET')
ax2.set_zlabel('β')
#ax2.set_title('β Performance with LET and Dose', x=0.5, y=-0.16, transform=ax2.transAxes, ha='center')
ax2.legend()
plt.subplots_adjust(bottom=0.15)

# survival存活曲线
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(Dose, LET, survival_sanwei, cmap='inferno', label='-ln(S)')
ax3.set_xlabel('Dose')
ax3.set_ylabel('LET')
ax3.set_zlabel('-ln(S)')
#ax3.set_title('-ln(S) Performance with LET and Dose', x=0.5, y=-0.16, transform=ax3.transAxes, ha='center')
ax3.legend()
plt.subplots_adjust(bottom=0.15)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.show()