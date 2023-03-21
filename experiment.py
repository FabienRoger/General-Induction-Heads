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
    batch_size: int, seq_len: int, induction_pos: int, fixed_A: Optional[int] = None, fixed_B: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert induction_pos >= 0 and induction_pos <= seq_len - 4
    seqs_with_induction = torch.randint(MIN_TOKEN, vocab_size, (batch_size, seq_len))

    if fixed_A:
        seqs_with_induction[:, -2] = fixed_A
    if fixed_B:
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
    _, inds = torch.topk(cosines, k=k, largest=True)
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
# fixed_strs = (" M", "arius")
# fixed_strs = ("arius", " M")
A, B = [int(tokenize_no_bos(s).item()) for s in fixed_strs]
A, B = None, None
# A = None
# B = None

B_IDX = 0
A_IDX = 1
K = 3
NN_IDXES = list(range(2, 2 + K))


def get_toks_of_interest(seqs_with_induction: torch.Tensor) -> torch.Tensor:
    A_toks = seqs_with_induction[:, -2]
    B_toks = seqs_with_induction[:, -1]
    B_nns = find_nearest_neighbours(B_toks, k=K)
    return torch.stack([B_toks, A_toks, *B_nns.T], dim=-1)


length = 24
pos = 10
batch_size = 64
N = 32
all_diffs = []
all_befores = []
all_afters = []
all_seqs_with = []
all_seqs_without = []
for _ in trange(30):
    seqs_with, seqs_without = random_induction_and_non_induction_seqs(batch_size, length, pos, fixed_A=A, fixed_B=B)
    diffs, before, after = measure_induction_power(seqs_with, seqs_without, get_toks_of_interest)
    all_diffs.append(diffs)
    all_seqs_with.append(seqs_with)
    all_seqs_without.append(seqs_without)
    all_befores.append(before)
    all_afters.append(after)
diffs_c = torch.cat(all_diffs, dim=0)
seqs_with_c = torch.cat(all_seqs_with, dim=0)
seqs_without_c = torch.cat(all_seqs_without, dim=0)
befores_c = torch.cat(all_befores, dim=0)
afters_c = torch.cat(all_afters, dim=0)
# #%%
# print(diffs_c.mean(0))
#%%
from matplotlib import pyplot as plt

plt.hist(diffs_c[:, 1].cpu().numpy(), bins=100, label="A", alpha=0.5)
plt.hist(diffs_c[:, 0].cpu().numpy(), bins=100, label="B", alpha=0.5)
plt.hist(diffs_c[:, 2].cpu().numpy(), bins=100, label="B nn", alpha=0.5)
plt.title("log P on induction - log P on shuffled (last is still A)")
plt.legend()
# %%
# before v after heatmap (2D histogram)
idx = NN_IDXES[0]
# for idx, name in ((0, "B"),):
for idx, name in ((0, "B"), (2, "B nn"), (1, "A")):
    plt.hist2d(befores_c[:, idx].cpu().numpy(), afters_c[:, idx].cpu().numpy(), bins=100, range=((-20, 0), (-20, 0)))
    plt.plot([10, -20], [10, -20], color="red")
    plt.xlabel("induction")
    plt.ylabel("shuffled")
    plt.colorbar()
    plt.title(name)
    plt.show()
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
for i in trange(MIN_TOKEN, MIN_TOKEN + 10000):
    tok = i
    A_pow, B_pow, A_pow_std, B_pow_std = token_powers(tok)
    tok_str = model.tokenizer.decode([tok])  # type: ignore
    all_powers.append((tok, tok_str, A_pow, B_pow, A_pow_std, B_pow_std))
#%%
copy = all_powers.copy()
# %% save to df
import pandas as pd

for i, (tok, tok_str, A_pow, B_pow, A_pow_std, B_pow_std) in enumerate(all_powers):
    tok_str_escaped = repr(tok_str)[1:-1]
df = pd.DataFrame(all_powers, columns=["tok", "tok_str", "A_pow", "B_pow", "A_pow_std", "B_pow_std"])
df.to_csv("powers.csv", escapechar="\\", index=False)
