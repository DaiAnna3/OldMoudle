from 图7绘制 import spower, RBE4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 导入参考数据
data = pd.read_excel('input6.xlsx')
let_1 = data.iloc[0:6, 0]
rbe_1 = data.iloc[0:6, 1]
let_10 = data.iloc[0:6, 2]
rbe_10 = data.iloc[0:6, 3]
let_50 = data.iloc[0:6, 4]
rbe_50 = data.iloc[0:6, 5]

let_1 = np.array(let_1)
let_10 = np.array(let_10)
let_50 = np.array(let_50)
rbe_1 = np.array(rbe_1)
rbe_10 = np.array(rbe_10)
rbe_50 = np.array(rbe_50)

# 图8绘制
plt.plot(spower, RBE4[:, 0], 'o-', drawstyle='steps-post', label='1% Survival', antialiased=True)
plt.plot(spower, RBE4[:, 1], 'o-', drawstyle='steps-post', label='5% Survival', antialiased=True)
plt.plot(spower, RBE4[:, 2], 'o-', drawstyle='steps-post', label='10% Survival', antialiased=True)
# plt.plot(spower, RBE4[:, 3], 'o-', drawstyle='steps-post', label='50% Survival', antialiased=True)

plt.xlabel('LET (keV/μm)')
plt.ylabel('RBE')
plt.subplots_adjust(bottom=0.15)
plt.legend()
plt.show()