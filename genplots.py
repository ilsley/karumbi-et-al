import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import os
import sys

def trim_data(data: pd.Series):
    # normalise and exclude lowly expressed regions on the edges
    filtered_data = (data / data.mean()).where(lambda x : x > 0.2).dropna() 
    # fit onto a common axis
    x = np.linspace(0, 1, num = len(filtered_data), endpoint = True)
    f = interpolate.splrep(x, filtered_data)
    xdata = np.linspace(0.2, 1, num = 800, endpoint = True)
    ydata = interpolate.splev(xdata, f, der = 0)

    # remove background - measured in specific region
    background = np.quantile(ydata[100:500], q = 0.1)  # slightly different to Mathematica version
    no_background = pd.Series(ydata - background).apply(lambda x : x if x > 0 else 0)
    return(no_background)


if len(sys.argv) < 3:
    print('Please specify the source and destination directories')
    sys.exit()

input_path = sys.argv[1]
image_path = sys.argv[2]

if not os.path.isdir(image_path):
    print("Image directory does not exist")
    sys.exit()
   
for file in os.listdir(os.fsencode(input_path)):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): 
        fullname = os.path.join(input_path, filename)
        print(fullname)
        df = pd.read_csv(fullname, index_col = 0)
        df_mod = df.apply(trim_data, axis = 1)
        df_mean = df_mod.apply(np.mean)
        df_bar = df_mod.apply(np.std) / np.sqrt(df_mod.shape[0])
        df_up = df_mean + df_bar
        df_down = df_mean - df_bar
        plt = df_mean.plot(color = "black")
        plt.fill_between(range(df_mean.shape[0]), df_up, df_down,
                 color='gray', alpha=0.2)
        fig = plt.get_figure()
        fig.savefig(os.path.join(image_path, os.path.splitext(filename)[0]))
        fig.clf()
