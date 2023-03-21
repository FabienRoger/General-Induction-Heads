# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use("ggplot")
# %%
powers = pd.read_csv("powers.csv", index_col=0)
powers.head()
# %%
# print histograms of A and B powers
plt.title("A and B powers")
min_A_pow = powers["A_pow"].min()
min_B_pow = powers["B_pow"].min()
m = min(min_A_pow, min_B_pow)
max_A_pow = powers["A_pow"].max()
max_B_pow = powers["B_pow"].max()
M = max(max_A_pow, max_B_pow)
plt.hist([powers["A_pow"], powers["B_pow"]], bins=30, range=(m, M), label=["A", "B"], alpha=1) # type: ignore
plt.legend()
plt.xlabel("induction power")
plt.ylabel("count")
# %%
# print top 10 A and B powers and bottom 10 A and B powers
print("Top 10 A powers")
print(powers.sort_values("A_pow", ascending=False).head(10))
print("Top 10 B powers")
print(powers.sort_values("B_pow", ascending=False).head(10))
print("Bottom 10 A powers")
print(powers.sort_values("A_pow", ascending=True).head(10))
print("Bottom 10 B powers")
print(powers.sort_values("B_pow", ascending=True).head(10))
#%%