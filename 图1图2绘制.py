'''由于κ、t的拟合一直不顺利，2024年10月4日，我打算先放弃每细胞核每剂量DSB产额YD(DSB/nuc/Gy)的最优参数的计算κ，t
转向画图，先完成海报的制作。本文件负责1.mfp、spower随δ电子能量变化曲线图。2.每细胞核每剂量DSB产额随δ电子能量变化曲线图。
的绘制。
图1.自变量：δ电子能量（E）,因变量：平均自由程(mfp)、停止功率(spower)。以及所需要的参考数据，将会通过input1.xlsx提供
图2.的自变量与图1.一致,因变量将会通过下面YD计算的模块计算出来，所需要的每细胞核每剂量DSB产额会通过input2.xlsx提供'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

'''图1（a)：mfp随电子能量变化曲线图，图1（b）spower随电子能量变化曲线图。下面我将分为3部分绘图（数据导入，图1（a），图1（b)),
其中的数据分为Geant4_DNA模拟数据为E、mfp、spwoer、track.参考数据来自Teabu Puful(2022,fig2,ICRU90)：E_ref,mfp_ref,
spower_ref、track_ref'''

# 数据导入
data = pd.read_excel('input1.xlsx')
E = data.iloc[:, 0]
MFP = data.iloc[:, 1]
SPOWER = data.iloc[:, 2]
TRACK = data.iloc[:, 3]
E_mfp_ref = data.iloc[:, 4]
mfp_ref = data.iloc[:, 5]
E_spower_ref = data.iloc[:, 6]
spower_ref = data.iloc[:, 7]
E_track_ref = data.iloc[:, 8]
track_ref = data.iloc[:, 9]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12  # 设置字体大小

fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 创建1行2列的子图布局

# 绘制第一个子图（对应原来的图1(a)）
ax = axes[0]
# 绘制 this work 的曲线，线条加粗
ax.semilogx(E[3:], MFP[3:], label='this work', color='blue', linewidth=2)
# 绘制 ICRU 90 的散点，点的大小增加，边框加粗
ax.scatter(E_mfp_ref, mfp_ref, label='ICRU 90', color='green', s=50, edgecolors='black', linewidths=1)
ax.set_xlabel('δ电子能量(eV)', fontsize=14)
ax.set_ylabel('平均自由程(nm)', fontsize=14)
# 设置图例的字体大小和位置
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(fontsize=12, loc='upper right')
# 设置坐标轴刻度朝内
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(direction='in', length=6)
# 添加标题
ax.set_title('平均自由程与δ电子能量关系图', fontsize=18)

# 绘制第二个子图（对应原来的图1(b)）
ax = axes[1]
# 绘制 this work 的曲线，线条加粗
ax.semilogx(E[3:], SPOWER[3:], label='this work', color='blue', linewidth=2)
# 绘制 ICRU 90 的散点，点的大小增加，边框加粗
ax.scatter(E_spower_ref, spower_ref, label='ICRU 90', color='green', s=50, edgecolors='black', linewidths=1)
ax.set_xlabel('δ电子能量(eV)', fontsize=14)
ax.set_ylabel('停止功率(keV/nm)', fontsize=14)
# 设置图例的字体大小和位置
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(fontsize=12, loc='upper right')
# 设置坐标轴刻度朝内
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(direction='in', length=6)
# 添加标题
ax.set_title('停止功率与δ电子能量关系图', fontsize=18)

plt.subplots_adjust(bottom=0.15, wspace=0.3)  # 调整子图间距等布局参数
plt.show()


'''图2绘制比图1更为复杂，自变量保持不变，但是需要通过代码计算处理得到因变量YD,我可以将函数封装，当输入mfp、spower、τ以及
相应的参数κ，t。t = 6.4  # 产生两次电离需要经过的距离，单位：nm。k = 4.67e-5  # 每簇团包含DSB，单位：DSB/cluster
图2的绘制分为3部分：YD函数模块、参考数据导入、绘制图2

0.数据导入(参考数据，齐亚平、王文静、Teabu),再将这些数据叠加变成117x2的矩阵，
(第一列为：LET_齐雅平+LET_Teaba+LET_王文静，第二列为：DSB_齐雅平+DSB_Teaba+DSB_王文静)
对参考数据进行排列得到spower_actual(由小到大排列)，DSB_actual按照spower_actual的变化分布。

将SPOWER、E、MFP、TRARK,也组合在一起，按照SPOWER由小到大排序分布，得到新的spower、e、mfp、track,最后对(spwoer,e),
(spower,mfp),(spower,track)插值得到spower_actual对应的mfp_actual、e_actual、track_actual

最后将mfp_actual、track_actual、spower_actuala按照δ电子能量e_actual由小到大排序得到
新的e_actual,mfp_actual,spower_actual,track_actual

1.灰狼优化算法求取最优参数κ,t

2.YD、λ函数

3.计算λ_p
'''

# 参考数据导入
data = pd.read_excel('input2.xlsx')
LET_齐雅平, LET_Teaba, LET_王文静 = data.iloc[:, 0],data.iloc[0:45, 2],data.iloc[0:13, 4]
DSB_齐雅平, DSB_Teaba, DSB_王文静 = data.iloc[:, 1],data.iloc[0:45, 3],data.iloc[0:13, 5]

# 对参考数据进行排列得到spower_actual(由小到大排列)，DSB_actual按照spower_actual的变化分布
z1 = np.concatenate((LET_齐雅平, LET_Teaba, LET_王文静), axis=0)
z2 = np.concatenate((DSB_齐雅平, DSB_Teaba, DSB_王文静), axis=0)
z = np.column_stack((z1, z2))
z_sorted = z[z[:, 0].argsort()]
spower_actual, YD_actual = z_sorted[:,0], z_sorted[:,1] #参考数据按照LET，由大到小排序

# 将SPOWER、E、MFP、TRACK,也组合在一起，按照SPOWER由小到大排序分布，得到新的spower、e、mfp、track
z = np.column_stack([SPOWER, E, MFP, TRACK])
z_sorted = z[z[:,0].argsort()]
spower , e, mfp, track = z_sorted[:,0], z_sorted[:,1], z_sorted[:,2], z_sorted[:,3]

# 最后对(spwoer,e),(spower,mfp),(spower,track)插值得到spower_actual对应的mfp_actual、e_actual、track_actual
mfp_actual = interp1d(spower,e)(spower_actual)
e_actual = interp1d(spower,mfp)(spower_actual)
track_actual = interp1d(spower,track)(spower_actual)

# 组合 按照e_actual由小到大排列
z = np.column_stack([e_actual,mfp_actual, spower_actual, track_actual])
z_sorted = z[z[:,0].argsort()]
e_actual,mfp_actual, spower_actual, track_actual = z_sorted[:,0], z_sorted[:,1], z_sorted[:,2], z_sorted[:,3]


'''
1.灰狼优化算法求取最优参数κ,t
首先定义一个计算YD的灰狼优化算法，参考依据'图3图4绘制
'''
D = range(1,2,1)
def Y_Greywolf(E = e_actual, mfp = mfp_actual, spower = spower_actual, τ = track_actual, YD_actual = YD_actual):
    # 定义目标函数
    def objective_function(params):
        D = 1
        κ, t = params
        beta = 0.7  # 相对速度
        c = 3e8  # 光速，单位：m/s
        me = 9.10938356e-31  # 电子静止质量，单位：kg
        Z = 2  # 电子的核电荷数
        C = 8.5e-3 * 1.602e-10  # 常数，单位J/m
        ρ = 1e3  # 水的密度，单位：kg/m³
        V = 5e-16  # 细胞核的体积，单位：m³
        R = np.cbrt(3 * V / (4 * np.pi))  # 细胞核半径，单位：m
        E1 = E * 1.602e-19  # 电子能量，单位：J
        LET = spower * 1.602e-10  # 停止功率，单位：J/m

        # A过程
        P = 1 - np.exp(-t / mfp)
        Ya = κ * P / mfp  # 每平均自由程产额，单位：DSB/nm

        # B过程
        N = np.zeros_like(Ya)
        for i in range(1, len(N)):
            N[i] = np.trapezoid(Ya[0:i], τ[0:i])  # 某一能量的二次电子每平均自由程DSB产额，单位DSB

        # 二次电子能谱S归一化
        Z_star = Z * (1 - np.exp(-125 * beta * Z ** (-2 / 3)))
        S = (C * (1 + E1 / (me * c ** 2)) + Z_star ** 2) / (((1 + E1 / (2 * me * c ** 2)) * beta * E1) ** 2)
        NS = N * S

        f1 = np.zeros_like(NS)
        f2 = np.zeros_like(NS)
        for i in range(1, len(f1)):
            f1[i] = np.trapezoid(NS[:i], E[:i])
            f2[i] = np.trapezoid(S[:i], E[:i])
        aver_N = np.zeros_like(f1)
        Yb = np.zeros_like(f1)

        for i in range(1, len(f1)):
            if f2[i] == 0:
                aver_N[i] = 0
                Yb[i] = 0
            else:
                aver_N[i] = f1[i] / f2[i]
                Yb[i] = aver_N[i] / mfp[i]

        # 总产额，单位：DSB/nm
        Yall = np.zeros_like(Ya)
        for i in range(len(Ya)):
            if np.isnan(Yb[i]):
                Yall[i] = Ya[i]
            else:
                Yall[i] = Ya[i] + Yb[i]

        ## Y、λ计算
        YD_predicted = Yall * 10e9 * ρ * V * D / LET  # 每剂量每个细胞核 DSB 个数 Y，单位 DSB/GY/nuc
        '''由于灰狼优化算法只能得到单输出的函数的优化参数，故我在这里限定剂量是每细胞核每剂量DSB产额的计算，在Y函数中输出的是剂量矩阵对应的
        细胞核中辐射DSB产额'''
        #mse = mean_squared_error(YD_actual, YD_predicted)
        mse = r2_score(YD_actual, YD_predicted)

        return mse

    # 灰狼优化算法实现
    def initialize_wolves(num_wolves, dim, param_bounds):
        wolves = []
        for _ in range(num_wolves):
            wolf = []
            for low, high in param_bounds:
                wolf.append(np.random.uniform(low, high))
            wolves.append(wolf)
        return np.array(wolves)

    def update_position(wolf, alpha, beta, delta, a, param_bounds):
        new_position = []
        for i, bounds in enumerate(param_bounds):
            A1 = 2 * a * np.random.rand() - a
            C1 = 2 * np.random.rand()
            D_alpha = abs(C1 * alpha[i] - wolf[i])
            X1 = alpha[i] - A1 * D_alpha

            A2 = 2 * a * np.random.rand() - a
            C2 = 2 * np.random.rand()
            D_beta = abs(C2 * beta[i] - wolf[i])
            X2 = beta[i] - A2 * D_beta

            A3 = 2 * a * np.random.rand() - a
            C3 = 2 * np.random.rand()
            D_delta = abs(C3 * delta[i] - wolf[i])
            X3 = delta[i] - A3 * D_delta

            new_value = (X1 + X2 + X3) / 3
            # 限制参数在边界内
            new_value = np.clip(new_value, bounds[0], bounds[1])
            new_position.append(new_value)
        return np.array(new_position)

    def grey_wolf_optimizer(num_wolves, dim, param_bounds, max_iter):
        wolves = initialize_wolves(num_wolves, dim, param_bounds)
        fitness = np.zeros(num_wolves)

        alpha_pos = np.zeros(dim)
        alpha_score = float('inf')

        beta_pos = np.zeros(dim)
        beta_score = float('inf')

        delta_pos = np.zeros(dim)
        delta_score = float('inf')

        fitness_history = []

        for iter in range(max_iter):
            for i in range(num_wolves):
                fitness[i] = objective_function(wolves[i])

                if fitness[i] < alpha_score:
                    alpha_score = fitness[i]
                    alpha_pos = wolves[i].copy()

                elif fitness[i] < beta_score:
                    beta_score = fitness[i]
                    beta_pos = wolves[i].copy()

                elif fitness[i] < delta_score:
                    delta_score = fitness[i]
                    delta_pos = wolves[i].copy()

            a = 2 - iter * (2 / max_iter)

            for i in range(num_wolves):
                wolves[i] = update_position(wolves[i], alpha_pos, beta_pos, delta_pos, a, param_bounds)

            fitness_history.append(alpha_score)
            print(f"Iteration {iter + 1}: Best Fitness = {alpha_score}")

        return alpha_pos, alpha_score, fitness_history

    # 定义参数搜索空间
    #param_bounds = [(10.5, 11.5), (0.00003995,0.00004010)]  # κ, t的搜索范围
    param_bounds = [(5,11), (0.00005, 0.000055)]  # κ, t的搜索范围


    # 执行灰狼优化
    num_wolves = 10
    dim = 2
    max_iter = 50

    best_params, best_fitness, fitness_history = grey_wolf_optimizer(num_wolves, dim, param_bounds, max_iter)
    print(best_params)
    return best_params

best_params = Y_Greywolf()
print('*************** κ，t **************')
print('best_params of κ，t =',best_params)


'''三、定义YD以及绘图
1.YD函数
2.λ_p函数'''
#每细胞核每剂量DSB产额的计算 DSB/GY/nuc
def cal_Y(param = best_params, E = E, mfp = MFP, spower = SPOWER, τ = TRACK):
    # 参数定义
    t,κ = param
    beta = 0.7  # 相对速度
    c = 3e8  # 光速，单位：m/s
    me = 9.10938356e-31  # 电子静止质量，单位：kg
    Z = 2  # 电子的核电荷数
    C = 8.5e-3 * 1.602e-10  # 常数，单位J/m
    ρ = 1e3  # 水的密度，单位：kg/m³
    V = 5e-16  # 细胞核的体积，单位：m³
    R = np.cbrt(3 * V / (4 * np.pi))  # 细胞核半径，单位：m
    E1 = E * 1.602e-19  # 电子能量，单位：J
    LET = spower * 1.602e-10  # 停止功率，单位：J/m

    # A过程
    P = 1 - np.exp(-t / mfp)
    Ya = κ * P / mfp  # 每平均自由程产额，单位：DSB/nm

    # B过程
    N = np.zeros_like(Ya)
    for i in range(1, len(N)):
        N[i] = np.trapezoid(Ya[0:i], τ[0:i])  # 某一能量的二次电子每平均自由程DSB产额，单位DSB

    # 二次电子能谱S归一化
    Z_star = Z * (1 - np.exp(-125 * beta * Z ** (-2 / 3)))
    S = (C * (1 + E1 / (me * c ** 2)) + Z_star ** 2) / (((1 + E1 / (2 * me * c ** 2)) * beta * E1) ** 2)
    NS = N * S
    f1 = np.zeros_like(NS)
    f2 = np.zeros_like(NS)

    for i in range(1, len(f1)):
        f1[i] = np.trapezoid(NS[:i], E[:i])
        f2[i] = np.trapezoid(S[:i], E[:i])
    aver_N = np.zeros_like(f1)
    Yb = np.zeros_like(f1)

    for i in range(1, len(f1)):
        if f2[i] == 0:
            aver_N[i] = 0
            Yb[i] = 0
        else:
            aver_N[i] = f1[i] / f2[i]
            Yb[i] = aver_N[i] / mfp[i]

    # 总产额，单位：DSB/nm
    Yall = np.zeros_like(Ya)
    for i in range(len(Ya)):
        if np.isnan(Yb[i]):
            Yall[i] = Ya[i]
        else:
            Yall[i] = Ya[i] + Yb[i]

    # 每剂量每个细胞核 DSB 个数 Y，单位 DSB/GY/nuc
    Y = Yall * 10e9 * ρ * V / LET
    return Y
# 求取每细胞核每剂量DSB产额（DSN/nuc/Gy）
YD = cal_Y()

'''四 λ_p函数定义以及计算
'''
#每个δ电子产生的平均DSB产额
def cal_λ_p(Y = YD, spower = SPOWER, D = D):
    ρ = 1e3  # 水的密度，单位：kg/m³
    V = 5e-16  # 细胞核的体积，单位：m³
    R = np.cbrt(3 * V / (4 * np.pi))  # 细胞核半径，单位：m
    LET = spower * 1.602e-10  # 停止功率，单位：J/m

    N = np.zeros((len(spower), len(D)))
    for i in range(len(spower)):
        for j in range(len(D)):
            N[i, j] = Y[i] * D[j]  # 每个细胞核辐射产生 DSB 个数 N，单位 DSB/nuc

    n = np.zeros((len(spower),len(D)))
    for i in range(len(spower)):
        for j in range(len(D)):
            n[i,j] = np.pi * R ** 2 * ρ * D[j] / LET[i]  # 细胞核中的粒子数 n，单位：个

    #每个细胞中辐射诱导的DSB产额为N
    λ = N / n  # 每个粒子产生的DSB平均产额，单位：DSB/GY/nuc/个
    λ_p = λ / (1 - np.exp(-λ))  # 有效粒子数的有效平均DSB产额
    return λ_p
λ_P = cal_λ_p()

'''图五 图2——细胞核每剂量DSB产额随LET变化曲线图的绘制
#首先需要对LET进行从小到大的排序'''
z = np.column_stack((SPOWER[1:], YD[1:]))
z_sorted = z[z[:, 0].argsort()]
spower_sorted,YD_sorted = z_sorted[:, 0], z_sorted[:, 1]

#绘图2
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12  # 设置字体大小

fig, ax = plt.subplots(figsize=(10, 6))

# 自定义不同的形状和颜色
markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 绘制散点图，增加点的大小，设置线条宽度为 0
ax.scatter(spower_sorted, YD_sorted, label='this work', marker=markers[0], color=colors[0], s=80, linewidths=0)
ax.scatter(LET_王文静, DSB_王文静, label='王文静', marker=markers[1], color=colors[1], s=80, linewidths=0)
ax.scatter(LET_齐雅平, DSB_齐雅平, label='齐雅平', marker=markers[2], color=colors[2], s=80, linewidths=0)
ax.scatter(LET_Teaba, DSB_Teaba, label='Teaba', marker=markers[3], color=colors[3], s=80, linewidths=0)

ax.set_xlabel('LET(keV/μm)', fontsize=14)
ax.set_ylabel('每细胞核每剂量 DSB 产额(DSB/nuc/Gy)', fontsize=14)

# 设置横坐标为十进制坐标
ax.set_xscale('linear')

# 设置纵坐标为十进制坐标
ax.ticklabel_format(style='plain', axis='y')

# 设置图例的字体大小、位置和阴影效果
ax.legend(fontsize=12, loc='upper right', shadow=True)

# 设置网格线的颜色、样式和透明度
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

plt.subplots_adjust(bottom=0.15)

# 设置坐标轴刻度朝内
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(direction='in', length=6)

plt.show()

