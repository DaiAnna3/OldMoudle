'''绘制图3-4

前期准备工作：计算spower_actual对应的YD_actaul、λ_p_actual，利用灰狼优化算法进一步计算得到图3对应的α_predict.
YD_actual、λ_p_actual的计算需要导入'图1图2绘制.py'文件中的YD、λ_p函数进行计算，将计算的结果导入到一个input.xlsx文件中。
（input.xlsx文件中包含spower_PIDE、α_PIDE、β_PIDE、YD_actual、λ_p_actual信息）

图3（a,α_predict与α_actual随LET变化曲线图）：需要将YD_actual、λ_p_actual导入，自变量（LET）,因变量（α_predict与α_actual）
需要通过代码函数（α_GreyWolves）实现

图3（b,β_predict与β_actual随LET变化曲线图）：需要将YD_actual、λ_p_actual、ξ、ζ、μ_x、μ_y导入自变量（LET）,因变量（β_predict
与β_actual）需要通过代码函数（灰狼优化、β_GreyWolves）实现

图4（α/β随LET变化曲线图）：不需要额外输入参数，自变量：LET,因变量：α/β_predict与α/β_actual

10.07 发现β数值拟合因子并不好，可能需要一个共同拟合。由于时间因素，我觉得η的值不能是一个定值，可能与λ_p的值有关，我希望画一个看看
'''

from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from 图1图2绘制 import cal_Y,cal_λ_p
from sklearn.metrics import mean_squared_error
import pandas as pd

'''前期数据准备：目的是为了获得YD_PIDE、λ_p_PIDE。大致分为3步：第一步，导入所需spower_PIDE、α_PIDE、β_PIDE之后，
对初始spower、YD、λ_p进行插值得到YD_PIDE、λ_p_PIDE'''

data = pd.read_excel('input3.xlsx')
spower_PIDE = data.iloc[:,0]
α_PIDE = data.iloc[:,1]
β_PIDE = data.iloc[:,2]

data = pd.read_excel('input1.xlsx')
spower = data.iloc[:,2]
YD = cal_Y()# 来自 input1.xlsx
λ_p = cal_λ_p()# 来自 input1.xlsx

#让spower从大到小排列后，进行插值计算
z = np.column_stack((spower,YD,λ_p))
z_sorted = z[z[:, 0].argsort()]
spower = z_sorted[:, 0]
YD = z_sorted[:, 1]
λ_p = z_sorted[:, 2]

Y_predict = interp1d(spower,YD)(spower_PIDE)
λ_p_predict = interp1d(spower,λ_p)(spower_PIDE)
#plt.plot(spower_PIDE,Y_predict)
#plt.plot(spower_PIDE,λ_p_predict)


'''灰狼优化算法：我将这一部分封装成一个名为α_Greywolf的计算的函数，用于α的最优参数μ_x, μ_y, ξ, ζ的计算'''
def α_Greywolf(Y_1H = Y_predict, λ_p_1H = λ_p_predict , α_actual_1H = α_PIDE):
    # 定义目标函数
    def objective_function(params):
        μ_x, μ_y, ξ, ζ = params
        # 考虑到过度杀伤效应，并非所有辐射诱导的dsb都对细胞死亡有贡献
        P_contribution = (1 - np.exp(-np.float64(ζ) * λ_p_1H)) / (np.float64(ζ) * λ_p_1H)
        # 不与同一个粒子导出的DSB末端连接
        P_track = (1 - np.exp(-np.float64(ξ) * λ_p_1H)) / (np.float64(ξ) * λ_p_1H)
        # α_1H定义
        α_predict_1H = Y_1H * P_contribution * (1 - μ_x * (P_track)) * μ_y

        mse1 = mean_squared_error(α_actual_1H, α_predict_1H)
        return mse1

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

    # 定义参数搜索空间μ_x, μ_y, ξ, ζ
    param_bounds = [(0.13, 0.3), (0.0055, 0.006), (0.0003, 0.0004), (0.00001, 1)]
    # 执行灰狼优化
    num_wolves = 10
    dim = 4
    max_iter = 100
    best_params, best_fitness, fitness_history = grey_wolf_optimizer(num_wolves, dim, param_bounds, max_iter)

    # 参数输出
    print('********** μ_x, μ_y, ξ, ζ ***********')
    print('μ_x, μ_y, ξ, ζ = ', best_params)
    return best_params
μ_x, μ_y, ξ, ζ = α_Greywolf()

# α计算
def cal_α(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, Y_1H = Y_predict, λ_p_1H = λ_p_predict):
     # 考虑到过度杀伤效应，并非所有辐射诱导的dsb都对细胞死亡有贡献
    P_contribution = (1 - np.exp(-np.float64(ζ) * λ_p_1H)) / (np.float64(ζ) * λ_p_1H)
    # 不与同一个粒子导出的DSB末端连接
    P_track = (1 - np.exp(-np.float64(ξ) * λ_p_1H)) / (np.float64(ξ) * λ_p_1H)
    # α_1H定义
    α_predict_1H = Y_1H * P_contribution * (1 - μ_x * (P_track)) * μ_y
    return α_predict_1H
α_predict = cal_α()
np.savetxt('α_predict', α_predict)

'''灰狼优化算法：我将这一部分封装成一个名为β_Greywolf的计算的函数，用于β的最优参数yita1,yita2的计算，阈值界限当产额大于50DSB/nuc/GY'''
def β_Greywolf(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, spower = spower_PIDE, Y = Y_predict, landa_p = λ_p_predict, beta_actual = β_PIDE):
    # 定义目标函数
    def objective_function(params):
        yita1, yita2 = params
        # 计算 P_contribution 和 P_track
        P_contribution = (1 - np.exp(-ζ * landa_p)) / (ζ * landa_p)
        P_track = (1 - np.exp(-ξ * landa_p)) / (ξ * landa_p)

        # 分段函数η
        beta_predicted = []
        for i in range(len(spower)):
            if spower[i] > 10:
                η = yita2 / landa_p[i]
            else:
                η = yita1 / landa_p[i]
            beta_predicted.append(
                (η * Y[i] ** 2 * P_contribution[i] * P_track[i] * μ_x * μ_y * (1 - np.exp(-landa_p[i]))) / (
                            2 * landa_p[i]))

        mse = mean_squared_error(beta_actual, beta_predicted)
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
    #param_bounds = [(0.0177, 0.077),(0.0177, 0.077)]  # yita1, yita2的搜索范围
    param_bounds = [(0.08, 0.09), (0.08, 0.09)]  # yita1, yita2的搜索范围

    # 执行灰狼优化
    num_wolves = 10
    dim = 2
    max_iter = 200

    best_params, best_fitness, fitness_history = grey_wolf_optimizer(num_wolves, dim, param_bounds, max_iter)

    # 参数输出
    yita1, yita2 = best_params
    print('********** μ_x, μ_y, ξ, ζ, η1, η2 ***********')
    print('μ_x, μ_y, ξ, ζ, η1, η2 = ', μ_x, μ_y, ξ, ζ, yita1, yita2)
    return best_params

yita1, yita2 = β_Greywolf()

# β计算
def cal_β(μ_x = μ_x, μ_y = μ_y, ξ = ξ, ζ = ζ, yita1 = yita1, yita2 = yita2, spower = spower_PIDE, Y = Y_predict, landa_p = λ_p_predict):
    # 计算 P_contribution 和 P_track
    P_contribution = (1 - np.exp(-ζ * landa_p)) / (ζ * landa_p)
    P_track = (1 - np.exp(-ξ * landa_p)) / (ξ * landa_p)

    # 分段函数η
    beta_predicted = []
    for i in range(len(spower)):
        if spower[i] > 10:
            η = yita2 / landa_p[i]
        else:
            η = yita1 / landa_p[i]
        beta_predicted.append((η * Y[i] ** 2 * P_contribution[i] * P_track[i] * μ_x * μ_y * (1 - np.exp(-landa_p[i]))) / (2 * landa_p[i]))

    return beta_predicted

β_predict = cal_β()


'''已知α_PIDE、β_PIDE、spower_PIDE,已经求出α_predict、β_predict。我们可以绘制α_PIDE、α_predict随spower_PIDE变化曲线图。
也可以绘制出β_PIDE、β_predict随spower_PIDE变化曲线图。以及α/β_PIDE、α/β_predict随spower_PIDE变化曲线图'''

plt.rcParams['font.sans-serif'] = ['SimHei']

# 计算 α/β_PIDE 和 α/β_predict
# 计算 α/β_PIDE 和 α/β_predict，避免除数为零
alpha_over_beta_PIDE = [a / b if b!= 0 else None for a, b in zip(α_PIDE, β_PIDE)]
alpha_over_beta_predict = [a / b if b!= 0 else None for a, b in zip(α_predict, β_predict)]

# 绘制 α_PIDE、α_predict 随 spower_PIDE 变化曲线图
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(spower_PIDE, α_PIDE, s=80, c='skyblue', label='PIDE数据库α', marker='o', edgecolors='k', linewidths=1)
ax1.plot(spower_PIDE, α_predict, linewidth=2, linestyle='--', color='orange', label='模型预测α')
ax1.set_xlabel('LET(keV/μm)')
ax1.set_ylabel('α value')
#ax1.set_title('图3(a) PIDE数据库α、模型预测α随传能线密度变化曲线图', x=0.5, y=-0.16, transform=ax1.transAxes, ha='center')
ax1.legend()
plt.subplots_adjust(bottom=0.15)
plt.show()

# 绘制 β_PIDE、β_predict 随 spower_PIDE 变化曲线图
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(spower_PIDE, β_PIDE, s=80, c='lightgreen', label='PIDE数据库β', marker='s', edgecolors='k', linewidths=1)
ax2.plot(spower_PIDE, β_predict, linewidth=2, linestyle='-.', color='purple', label='模型预测β')
ax2.set_xlabel('LET(keV/μm)')
ax2.set_ylabel('β value')
#ax2.set_title('图3(b) PIDE数据库β、模型预测β随传能线密度变化曲线图', x=0.5, y=-0.16, transform=ax2.transAxes, ha='center')
ax2.legend()
plt.subplots_adjust(bottom=0.15)
plt.show()

# 绘制 α/β_PIDE、α/β_predict 随 spower_PIDE 变化曲线图
fig3, ax3 = plt.subplots(figsize=(10, 6))
np.savetxt('spower_PIDE',spower_PIDE)
print(alpha_over_beta_PIDE)
ax3.scatter(spower_PIDE, alpha_over_beta_PIDE, s=80, c='salmon', label='PIDE数据库α/β', marker='d', edgecolors='k', linewidths=1)
ax3.plot(spower_PIDE, alpha_over_beta_predict, linewidth=2, linestyle=':', color='teal', label='模型预测α/β')
ax3.set_xlabel('LET(keV/μm)')
ax3.set_ylabel('α/β value')
#ax3.set_title('图4 PIDE数据库α/β、模型预测α/β随传能线密度变化曲线图', x=0.5, y=-0.16, transform=ax3.transAxes, ha='center')
ax3.legend()
plt.subplots_adjust(bottom=0.15)
plt.show()