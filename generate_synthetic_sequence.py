import torch
import numpy as np
import pandas as pd
from torch import nn
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler   # only for type hints


@torch.no_grad()
def generate_synthetic_sequence(
    model: nn.Module,
    initial_lab_values: torch.Tensor,          # 1-D tensor: [num_feats] + [cat_id]
    num_timesteps: int,
    lab_columns: List[str],                    # order: all numerical cols, then the categorical col
    le: LabelEncoder,                          # fitted on the categorical column
    scaler: Optional[StandardScaler] = None,   # fitted *only* on the numerical columns
):
    """
    Greedy autoregressive generation of `num_timesteps` future lab values.

    * The model is called twice at each step: once in 'numerical' mode and
      once in 'categorical' mode.
    * Assumes **one** categorical feature per time-step (the one you
      label-encoded with `le`).
    * Returns a DataFrame with columns == `lab_columns` and an extra
      'avisitn' column (1 … n).
    """
    device = next(model.parameters()).device
    initial_lab_values = initial_lab_values.to(device)

    # ------------------------------------------------------------------
    # 1. Split the baseline vector into numerical and categorical parts
    # ------------------------------------------------------------------
    n_num = model.num_numerical_features
    baseline_num = initial_lab_values[:n_num]               # (n_num,)
    baseline_cat = initial_lab_values[n_num:].long()        # () or (1,)

    mask_token_id = le.transform(["[MASK]"])[0]

    baseline_num_mask = ~torch.isnan(baseline_num)          # (n_num,)
    baseline_cat_mask = baseline_cat != mask_token_id       # (1,)  bool

    # ------------------------------------------------------------------
    # 2. Prepare immutable encoder inputs (batch == 1)
    # ------------------------------------------------------------------
    src_num_tensor = baseline_num.unsqueeze(0)              # (1, n_num)
    src_cat_tensor = baseline_cat.unsqueeze(0).unsqueeze(1) # (1, 1)

    src_num_mask   = baseline_num_mask.unsqueeze(0)         # (1, n_num)
    src_cat_mask   = baseline_cat_mask.unsqueeze(0).unsqueeze(1)  # (1, 1)

    # ------------------------------------------------------------------
    # 3. Containers for autoregressive generation
    # ------------------------------------------------------------------
    num_tokens   = [baseline_num]           # list[Tensor(n_num,)]
    cat_tokens   = [baseline_cat]           # list[Tensor(1,)]

    num_masks    = []
    cat_masks    = []

    # ------------------------------------------------------------------
    # 4. Greedy loop
    # ------------------------------------------------------------------
    for t in range(num_timesteps):
        # ---------- build decoder inputs ----------
        if t == 0:
            tgt_num_seq  = baseline_num.unsqueeze(0).unsqueeze(0)     # (1, 1, n_num)
            tgt_cat_seq  = baseline_cat.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1)

            tgt_num_mask = baseline_num_mask.unsqueeze(0).unsqueeze(0)          # (1, 1, n_num)
            tgt_cat_mask = baseline_cat_mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1)
        else:
            # stack predictions so far (skip the baseline position 0)
            tgt_num_seq  = torch.stack(num_tokens[1:], dim=0).unsqueeze(0)      # (1, t, n_num)
            tgt_cat_seq  = torch.stack(cat_tokens[1:], dim=0).unsqueeze(0).unsqueeze(2)  # (1, t, 1)

            tgt_num_mask = torch.stack(num_masks, dim=0).unsqueeze(0)           # (1, t, n_num)
            tgt_cat_mask = torch.stack(cat_masks, dim=0).unsqueeze(0).unsqueeze(2)       # (1, t, 1)

        # causal mask for decoder self-attention
        seq_len = tgt_num_seq.size(1)
        causal_mask = None
        if seq_len > 0:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )

        # ---------- numerical prediction ----------
        pred_num = model(
            src=src_num_tensor,
            tgt=tgt_num_seq,
            src_mask=src_num_mask,
            tgt_mask=tgt_num_mask,
            tgt_attn_mask=causal_mask,
            mode="numerical",
        )  # (1, seq_len, n_num)
        next_num = pred_num[0, -1, :]                   # (n_num,)
        next_num = torch.nan_to_num(next_num, nan=0.0)
        num_tokens.append(next_num)
        num_masks.append(torch.isfinite(next_num))

        # ---------- categorical prediction ----------
        pred_cat_logits = model(
            src=src_cat_tensor,
            tgt=tgt_cat_seq,
            src_mask=src_cat_mask,
            tgt_mask=tgt_cat_mask,
            tgt_attn_mask=causal_mask,
            mode="categorical",
        )  # (1, seq_len, num_categories)
        next_cat = torch.argmax(pred_cat_logits[0, -1, :]).long()    # ()
        cat_tokens.append(next_cat)
        cat_masks.append(next_cat != mask_token_id)

    # ------------------------------------------------------------------
    # 5. Stack generated tokens (drop baseline position 0)
    # ------------------------------------------------------------------
    num_seq_tensor = torch.stack(num_tokens[1:], dim=0)              # (n_steps, n_num)
    cat_seq_tensor = torch.stack(cat_tokens[1:], dim=0)              # (n_steps,)

    # ------------------------------------------------------------------
    # 6. De-normalise numericals, decode categorical IDs
    # ------------------------------------------------------------------
    num_np = num_seq_tensor.cpu().numpy()
    if scaler is not None:
        num_np = scaler.inverse_transform(num_np)                    # (n_steps, n_num)

    cat_np   = cat_seq_tensor.cpu().numpy().astype(int)              # (n_steps,)
    cat_text = le.inverse_transform(cat_np)                          # original strings

    # ------------------------------------------------------------------
    # 7. Assemble the DataFrame
    # ------------------------------------------------------------------
    num_cols = lab_columns[:n_num]
    cat_col  = lab_columns[n_num]      # assumes exactly one categorical column

    df_num = pd.DataFrame(num_np, columns=num_cols)
    df_cat = pd.DataFrame({cat_col: cat_text})

    generated_df = pd.concat([df_num, df_cat], axis=1)
    generated_df["avisitn"] = np.arange(1, len(generated_df) + 1)

    return generated_df
