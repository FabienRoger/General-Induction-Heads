# %%
from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use("ggplot")
# %%
powers = pd.read_csv("powers.csv")
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

# ro_A_power = powers[powers["tok_str"] == " ro"]["A_pow"].item()
# ro_B_power = powers[powers["tok_str"] == " ro"]["B_pow"].item()
# ger_A_power = powers[powers["tok_str"] == "ger"]["A_pow"].item()
# ger_B_power = powers[powers["tok_str"] == "ger"]["B_pow"].item()
# plt.axvline(ro_A_power, color="red", label="' ro' A power", alpha=0.5, linestyle="--")
# plt.axvline(ro_B_power, color="blue", label="' ro' B power", alpha=0.5, linestyle="--")
# plt.axvline(ger_A_power, color="red", label="'ger' A power", alpha=0.5, linestyle="-.")
# plt.axvline(ger_B_power, color="blue", label="'ger' B power", alpha=0.5, linestyle="-.")

plt.legend()
plt.xlabel("induction power")
plt.ylabel("count")
# %%
from IPython.display import display
# print top 10 A and B powers and bottom 10 A and B powers
print("Top 10 A powers")
display(powers.sort_values("A_pow", ascending=False).head(10))
display(powers.sort_values("B_pow", ascending=False).head(10))
display(powers.sort_values("A_pow", ascending=True).head(10))
display(powers.sort_values("B_pow", ascending=True).head(10))
# print("Top 10 B powers")
# print(powers.sort_values("B_pow", ascending=False).head(10))
# print("Bottom 10 A powers")
# print(powers.sort_values("A_pow", ascending=True).head(10))
# print("Bottom 10 B powers")
# print(powers.sort_values("B_pow", ascending=True).head(10))
#%%
# check for the " Fab" token
print(powers[powers["tok_str"] == "ien"])
# %%
# %%
from transformer_lens import HookedTransformer
import torch

# %%
model_name = "attn-only-2l"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: HookedTransformer = HookedTransformer.from_pretrained(model_name).to(device)  # type: ignore
#%%
for p in model.parameters():
    p.requires_grad = False
#%%
def tokenize(text) -> torch.Tensor:
    return model.to_tokens(text)


def tokenize_no_bos(text) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=False)


def tokenize_strs(text) -> list[str]:
    return model.to_str_tokens(text)

print(tokenize_strs(" roger"))
# %%
