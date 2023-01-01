import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use("seaborn")
import SMABackTester as SMA

tester = SMA.SMABacktester("EURUSD=X", 50, 200, "2004-01-01", "2020-06-30")
tester.test_strategy()
tester.plot_results()
plt.show()

tester.optimize_parameters((25, 50, 1), (100, 200, 1))
tester.plot_results()
plt.show()
