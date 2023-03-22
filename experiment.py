# %%
from transformer_lens import HookedTransformer
import torch
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use("ggplot")
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


# %%
s = " Fabien amlzealkealzmek Fab"
print(tokenize_strs(s))
logits = model(s)
last_pred_probs = torch.softmax(logits, dim=-1)[0, -1]
fab_token, ien_token = int(tokenize_no_bos(" Fab").item()), int(tokenize_no_bos("ien").item())
prob_Fab, prob_ien = last_pred_probs[fab_token].item(), last_pred_probs[ien_token].item()
print(prob_Fab, prob_ien)
# %%
s = "amlzealkealzmek Fab"
logits = model(s)
last_pred_probs = torch.softmax(logits, dim=-1)[0, -1]
prob_Fab, prob_ien = last_pred_probs[fab_token].item(), last_pred_probs[ien_token].item()
print(prob_Fab, prob_ien)
# %%
vocab_size = logits.shape[-1]
# %%
from typing import Callable, Optional

MIN_TOKEN = 4


def random_induction_and_non_induction_seqs(
    batch_size: int, seq_len: int, induction_pos: int, fixed_A: int | None | torch.Tensor = None, fixed_B: int | None | torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert induction_pos >= 0 and induction_pos <= seq_len - 4
    seqs_with_induction = torch.randint(MIN_TOKEN, vocab_size, (batch_size, seq_len))

    if fixed_A is not None:
        if isinstance(fixed_A, torch.Tensor):
            fixed_A = fixed_A[torch.randint(0, fixed_A.shape[0], (batch_size,))]
        seqs_with_induction[:, -2] = fixed_A
    if fixed_B is not None:
        if isinstance(fixed_B, torch.Tensor):
            fixed_B = fixed_B[torch.randint(0, fixed_B.shape[0], (batch_size,))]
        seqs_with_induction[:, -1] = fixed_B
    seqs_with_induction[:, induction_pos : induction_pos + 2] = seqs_with_induction[:, -2:]

    # permute all tokens in each sequence expect the last two
    seqs_without_induction = seqs_with_induction[:, :-2].clone()
    permutations = torch.argsort(torch.rand(seqs_without_induction.shape), dim=-1)
    seqs_without_induction = seqs_without_induction[torch.arange(batch_size).unsqueeze(-1), permutations]
    seqs_without_induction = torch.cat([seqs_without_induction, seqs_with_induction[:, -2:]], dim=-1)

    # check shapes
    assert seqs_with_induction.shape == (batch_size, seq_len) and seqs_without_induction.shape == (batch_size, seq_len)
    return seqs_with_induction, seqs_without_induction


print("\n".join(map(str, random_induction_and_non_induction_seqs(2, 10, 3))))
#%%
# find the neirest neighbours of a token based on logit lens (unembedding matrix)
unembed_matrix = model.unembed.W_U


def find_nearest_neighbours(tokens: torch.Tensor, k: int = 10) -> torch.Tensor:
    normed_unembed = unembed_matrix / torch.norm(unembed_matrix, dim=0, keepdim=True)
    token_w = unembed_matrix[:, tokens]
    cosines = torch.einsum("ji,jl->li", normed_unembed, token_w)
    _, inds = torch.topk(cosines, k=k+1, largest=True)
    return inds[:, 1:].to(tokens.device)


print(model.tokenizer.batch_decode(find_nearest_neighbours(tokenize_no_bos(" man")[0])))  # type: ignore
# %%


def log_probs_to_log_odd_ratio(log_probs: torch.Tensor) -> torch.Tensor:
    # odd_ratio = exp(log_probs) / (1 - exp(log_probs))
    # log_odd_ratio = log(odd_ratio) = log(exp(log_probs) / (1 - exp(log_probs))) = log(exp(log_probs)) - log(1 - exp(log_probs))
    return log_probs - torch.log1p(-torch.exp(log_probs))


def measure_induction_power(
    seqs_with_induction: torch.Tensor,
    seqs_without_induction: torch.Tensor,
    get_toks_of_interest: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def get_logprobs_of_interest(seqs: torch.Tensor, tokens_of_interest: torch.Tensor) -> torch.Tensor:
        """Both of shape (batch_size, seq_len)"""
        # prepend bos and remove last token
        tokens = torch.cat([torch.ones(seqs.shape[0], 1, dtype=torch.long), seqs[:, :-1]], dim=-1)
        logits = model(tokens.to(device))[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs_of_interest = torch.gather(logprobs, dim=-1, index=tokens_of_interest.to(device))
        return logprobs_of_interest

    tokens_of_interest = get_toks_of_interest(seqs_with_induction)
    before = get_logprobs_of_interest(seqs_with_induction, tokens_of_interest)
    after = get_logprobs_of_interest(seqs_without_induction, tokens_of_interest)
    diff = log_probs_to_log_odd_ratio(before) - log_probs_to_log_odd_ratio(after)
    return diff, before, after


import gc
from tqdm import trange

gc.collect()

fixed_strs = (" Fab", "ien")
# fixed_strs = (" Fa", "ien")
# fixed_strs = (" M", "arius")
# fixed_strs = ("arius", " M")
A, B = [int(tokenize_no_bos(s).item()) for s in fixed_strs]
# A, B = None, None
# A = None
# B = None

B_IDX = 0
A_IDX = 1
K = 3
NN_IDXES = list(range(2, 2 + K))
RANDOM_IDX = 2 + K

def get_toks_of_interest(seqs_with_induction: torch.Tensor) -> torch.Tensor:
    A_toks = seqs_with_induction[:, -2]
    B_toks = seqs_with_induction[:, -1]
    B_nns = find_nearest_neighbours(B_toks, k=K)
    rdm_toks = torch.randint(MIN_TOKEN, vocab_size, (seqs_with_induction.shape[0],))
    return torch.stack([B_toks, A_toks, *B_nns.T, rdm_toks], dim=-1)

scenarios = {
    "fixed A=' Fab' and B='ien'": (A, B),
    "fixed A=' Fab'": (A, None),
    "fixed B='ien'": (None, B),
    "no fixed A or B": (None, None),
}

diffs_per_scenarios = {}
before_and_after_per_scenarios = {}

length = 24
pos = 10
batch_size = 256
N = 16

def measure(fixed_A=None, fixed_B=None):
    all_diffs = []
    all_befores = []
    all_afters = []
    for _ in trange(N):
        seqs_with, seqs_without = random_induction_and_non_induction_seqs(batch_size, length, pos, fixed_A=fixed_A, fixed_B=fixed_B)
        diffs, before, after = measure_induction_power(seqs_with, seqs_without, get_toks_of_interest)
        all_diffs.append(diffs)
        all_befores.append(before)
        all_afters.append(after)
    diffs_c = torch.cat(all_diffs, dim=0)
    befores_c = torch.cat(all_befores, dim=0)
    afters_c = torch.cat(all_afters, dim=0)
    return diffs_c, befores_c, afters_c
#%%

for scenario_name, (fixed_A, fixed_B) in scenarios.items():
    diffs_c, befores_c, afters_c = measure(fixed_A=fixed_A, fixed_B=fixed_B)
    
    diffs_per_scenarios[scenario_name] = diffs_c
    before_and_after_per_scenarios[scenario_name] = (befores_c, afters_c)
# #%%
# print(diffs_c.mean(0))
#%%

for scenario_name, diffs_c in diffs_per_scenarios.items():
    plt.hist(diffs_c[:, B_IDX].cpu().numpy(), bins=30, label=scenario_name, range=(-10, 20), histtype=u'step')
plt.title("Histogram of induction strengths over 4096 sequences")
plt.legend(loc="upper left")
# %%
m = -18
M = 6
fig, axs = plt.subplots(1, len(scenarios), figsize=(20, 5))
for i, (scenario_name, (befores_c, afters_c)) in enumerate(before_and_after_per_scenarios.items()):
    before_log_od = log_probs_to_log_odd_ratio(befores_c[:, B_IDX])
    after_log_od = log_probs_to_log_odd_ratio(afters_c[:, B_IDX])
    im = axs[i].hist2d(before_log_od.cpu().numpy(), after_log_od.cpu().numpy(), bins=100, range=((m, M), (m, M)), cmap="Blues")
    
    avg_point = (before_log_od.mean().cpu().item(), after_log_od.mean().cpu().item())
    corresponding_point_on_diag = (avg_point[1], avg_point[1])
    # draw double pointed arrow from avg point to corresponding point on diag and write "induction strength on it"
    axs[i].annotate("", xy=avg_point, xytext=corresponding_point_on_diag, arrowprops=dict(arrowstyle="<->", color="black"))
    x = (avg_point[0] + corresponding_point_on_diag[0]) / 2
    y = (avg_point[1] + corresponding_point_on_diag[1]) / 2
    axs[i].text(x, y, "avg induction\nstrength", color="black", ha="center", va="center")
    axs[i].plot([M, m], [M, m], color="black")
    axs[i].set_xlabel("log odd-ratio on induction")
    axs[i].set_ylabel("log odd-ratio on shuffled")
    axs[i].set_title(scenario_name)
    axs[i].set_aspect("equal")
fig.suptitle("2D histogram of log odd-ratios before and after shuffling the sequences\n")
plt.tight_layout()
# %%
import numpy as np


def token_powers(tok) -> tuple[float, float, float, float]:
    def get_toks_of_interest(seqs_with_induction: torch.Tensor) -> torch.Tensor:
        B_toks = seqs_with_induction[:, -1]
        return B_toks.unsqueeze(-1)

    seqs_with, seqs_without = random_induction_and_non_induction_seqs(
        batch_size,
        length,
        pos,
        fixed_A=tok,
        fixed_B=None,
    )
    diffs, _, _ = measure_induction_power(seqs_with, seqs_without, get_toks_of_interest)
    A_power = diffs[:, 0].mean().item()
    seqs_with, seqs_without = random_induction_and_non_induction_seqs(
        batch_size,
        length,
        pos,
        fixed_A=None,
        fixed_B=tok,
    )
    A_power_sigma = diffs[:, 0].std().item() / np.sqrt(batch_size)

    diffs, _, _ = measure_induction_power(seqs_with, seqs_without, get_toks_of_interest)
    B_power = diffs[:, 0].mean().item()

    B_power_sigma = diffs[:, 0].std().item() / np.sqrt(batch_size)

    return A_power, B_power, A_power_sigma, B_power_sigma


all_powers = []
# %%
recompute = False

if recompute:
    for i in trange(MIN_TOKEN, MIN_TOKEN + 10000):
        tok = i
        A_pow, B_pow, A_pow_std, B_pow_std = token_powers(tok)
        tok_str = model.tokenizer.decode([tok])  # type: ignore
        all_powers.append((tok, tok_str, A_pow, B_pow, A_pow_std, B_pow_std))

    for i, (tok, tok_str, A_pow, B_pow, A_pow_std, B_pow_std) in enumerate(all_powers):
        tok_str_escaped = repr(tok_str)[1:-1].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r").replace("\f", "\\f")
        all_powers[i] = (tok, tok_str_escaped, A_pow, B_pow, A_pow_std, B_pow_std)
    df = pd.DataFrame(all_powers, columns=["tok", "tok_str", "A_pow", "B_pow", "A_pow_std", "B_pow_std"])
    df.to_csv("powers.csv", escapechar="\\", index=False)
#%%
# df = pd.read_csv("powers.csv", escapechar="\\")
# proportion = 0.1
# top_A_tokens = torch.tensor(df.sort_values("A_pow", ascending=False).head(int(len(df) * proportion))["tok"].astype(int).values, dtype=torch.long)
K = 20
NN_IDXES = list(range(2, 2 + K))
RANDOM_IDX = 2 + K
diffs, *_ = measure()
plt.hist(diffs[:, B_IDX].cpu().numpy(), bins=30, label="B", histtype=u'step')
for i in [0, 4, 14]:
    n_neighb = NN_IDXES[i]
    name = f"{i+1}th closest token to B"
    plt.hist(diffs[:, n_neighb].cpu().numpy(), bins=30, label=name, histtype=u'step')
plt.hist(diffs[:, RANDOM_IDX].cpu().numpy(), bins=30, label="Random tokens", histtype=u'step', color="black")
plt.xlabel("induction strength")
plt.ylabel("count")
plt.title("Induction strength of for B and tokens near B")
plt.legend()
# for i in [0,1, 5, 15]:
#     idx = i + 1 if i != 0 else 0
#     label = f"{i}th closest token to B" if i != 0 else "B"
#     plt.hist(diffs[:, idx].cpu().numpy(), bins=30, alpha=0.5, label=label)
# plt.legend()
# #%%
# bot_A_tokens = torch.tensor(df.sort_values("A_pow", ascending=True).head(int(len(df) * proportion)).index.astype(int).values, dtype=torch.long)
# K = 20
# diffs, *_ = measure(bot_A_tokens, None)

# for i in [0,1, 5, 15]:
#     idx = i + 1 if i != 0 else 0
#     label = f"{i}th closest token" if i != 0 else "B"
#     plt.hist(diffs[:, idx].cpu().numpy(), bins=30, alpha=0.5, label=label)
# plt.legend()
# # %%
# # 'a' defined earlier as pandas index, find place where it is not finite
# for i, row in df.iterrows():
#     if not np.isfinite(row["A_pow"]):
#         print(i,row)
#         break
# # %%

# %%
