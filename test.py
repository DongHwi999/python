import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_ep = pd.read_excel("ep.xlsx", header=None)
input_fp = pd.read_excel("fp.xlsx")

#input_fp = input_fp.drop(index='1', axis=0)
print(input_ep.shape)
#input_ep.plot()
#plt.show()

#input_fp.plot()
#plt.show()