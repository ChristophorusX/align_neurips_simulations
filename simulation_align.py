import fa
import data_gen
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 28})
# plt.rcParams["figure.figsize"] = (9, 9)
# rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)

n, d = (50, 150)
step = 10e-7
n_step = 20000
n_iter = 2
activation = 'relu'
p_start = 500
p_end = 1000
p_step = 100
p_list = np.arange(start=p_start, stop=p_end + p_step, step=p_step)
reg_list = [0, 5, 10, 15]


def get_align_df(p_list, reg_list, activation):
    reg_align_df = []
    for reg in reg_list:
        align_table = []
        for p in p_list:
            align_array = []
            for iter in range(n_iter):
                # X, y = data_gen.lr_data(n, d)
                X, y = data_gen.rand_nn_data(n, d, p)
                seed = np.random.randint(100000)
                net = fa.TwoLayerNetwork(activation, d, p, n, seed)
                loss_fa, beta, b = net.feedback_alignment(
                    X, y, step, regular=reg, n_steps=n_step)
                align = np.inner(beta, b) / \
                    np.linalg.norm(beta) / np.linalg.norm(b)
                align_array.append(align)
            align_table.append(align_array)
        align_table = np.array(align_table)
        flattened_table = align_table.flatten()
        p_index = np.repeat(p_list, n_iter)
        reg_index = np.repeat(reg, len(p_list) * n_iter)
        activation_index = np.repeat(activation, len(p_list) * n_iter)
        combined_table = np.vstack(
            (p_index, flattened_table, reg_index, activation_index)).T
        align_df = pd.DataFrame(data=combined_table, columns=[
                                r"$p$ Hidden Layer Width", 'Alignment', r"Regularization $\lambda$", 'Activation'])
        align_df[r"$p$ Hidden Layer Width"] = align_df[r"$p$ Hidden Layer Width"].astype(int)
        align_df['Alignment'] = align_df['Alignment'].astype(np.double)
        align_df[r"Regularization $\lambda$"] = align_df[r"Regularization $\lambda$"].astype(int)
        reg_align_df.append(align_df)
    df = pd.concat(reg_align_df)
    return df


df_relu = get_align_df(p_list, reg_list, 'relu')
df_relu

align_relu_plot = sns.lineplot(x=r"$p$ Hidden Layer Width", y='Alignment',
                               hue=r"Regularization $\lambda$", data=df_relu, legend="full")
align_relu_fig = align_relu_plot.get_figure()
align_relu_fig.savefig('align_relu_fig.pdf')
