import simlation_align
import fa_two_layer
import net_autograd
import data_gen
import numpy as np
import torch
from torch import nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
sns.set_palette(sns.color_palette())
rc('text', usetex=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Generate alignment plot for linear network and lr data
n, d = (50, 150)
step = 10e-5
n_step = 2000
n_iter = 10
p_start = 300
p_end = 1000
p_step = 25
p_list = [500, 1000, 2000, 4000, 8000, 16000]
reg_list = [0, 1, 2]
df_lr = simulation_align.get_autograd_align_df(n, d, p_list, reg_list, 'non', 'lr', step, n_step, n_iter)
df_lr.to_csv('df_lr_large.csv', index=False)
simulation_align.plot_align(df_lr, 'align_autograd_lr_fig_large.pdf')
