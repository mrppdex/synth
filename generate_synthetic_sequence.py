@torch.no_grad()
def generate_synthetic_sequence(
    model,
    initial_lab_values: torch.Tensor,   # numericals ⨁ label-encoded categorical id
    num_timesteps: int,
    lab_columns: list[str],
    le,                                 # fitted LabelEncoder
    scaler=None,                        # StandardScaler for numericals, or None
):
    """
    Returns a DataFrame with *all* numerical columns first and the single
    categorical column last, plus 'avisitn'.
    """
    device = next(model.parameters()).device
    initial_lab_values = initial_lab_values.to(device)

    # ------------------------------------------------------------
    # 1. split baseline vector into numerical part + one cat id
    # ------------------------------------------------------------
    n_num = model.num_numerical_features          # how many numericals
    base_num = initial_lab_values[:n_num]          # (n_num,)
    base_cat = initial_lab_values[n_num:].long()   # scalar tensor ()

    mask_token_id = le.transform(["[MASK]"])[0]

    base_num_mask = ~torch.isnan(base_num)         # (n_num,)
    base_cat_mask = (base_cat != mask_token_id)    # ()

    # ------------------------------------------------------------
    # 2. build immutable encoder inputs  — shape (1, 1, *)
    # ------------------------------------------------------------
    src_num_tensor = base_num.unsqueeze(0)               # (1, n_num)
    src_cat_tensor = base_cat.view(1, 1, 1)              # (1, 1, 1)

    src_num_mask   = base_num_mask.unsqueeze(0)          # (1, n_num)
    src_cat_mask   = base_cat_mask.view(1, 1, 1)         # (1, 1, 1)

    # ------------------------------------------------------------
    # 3. containers for the autoregressive loop
    # ------------------------------------------------------------
    num_tokens = [base_num]          # list of (n_num,) tensors
    num_masks  = []

    cat_tokens = [base_cat]          # list of scalar tensors
    cat_masks  = []

    # ------------------------------------------------------------
    # 4. greedy generation
    # ------------------------------------------------------------
    for t in range(num_timesteps):
        # ---------- build decoder inputs ----------
        if t == 0:
            tgt_num_seq  = base_num.unsqueeze(0).unsqueeze(0)       # (1, 1, n_num)
            tgt_num_mask = base_num_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n_num)

            tgt_cat_seq  = base_cat.view(1, 1, 1)                   # (1, 1, 1)
            tgt_cat_mask = base_cat_mask.view(1, 1, 1)              # (1, 1, 1)
        else:
            # ----- numerical -----
            tgt_num_seq  = torch.stack(num_tokens[1:], dim=0).unsqueeze(0)      # (1, t, n_num)
            tgt_num_mask = torch.stack(num_masks,  dim=0).unsqueeze(0)           # (1, t, n_num)

            # ----- categorical  (keep 3-D: add .unsqueeze(-1)) -----
            cat_tensor  = torch.stack(cat_tokens[1:], dim=0).unsqueeze(-1)       # (t, 1)
            mask_tensor = torch.stack(cat_masks,  dim=0).unsqueeze(-1)           # (t, 1)
            tgt_cat_seq  = cat_tensor.unsqueeze(0)                               # (1, t, 1)
            tgt_cat_mask = mask_tensor.unsqueeze(0)                              # (1, t, 1)

        # causal mask for decoder self-attention
        seq_len = tgt_num_seq.size(1)           # same for cat & num
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

        next_num = pred_num[0, -1, :]                # (n_num,)
        next_num = torch.nan_to_num(next_num, nan=0.0)
        num_tokens.append(next_num)
        num_masks .append(torch.isfinite(next_num))

        # ---------- categorical prediction ----------
        pred_cat_logits = model(
            src=src_cat_tensor,
            tgt=tgt_cat_seq,
            src_mask=src_cat_mask,
            tgt_mask=tgt_cat_mask,
            tgt_attn_mask=causal_mask,
            mode="categorical",
        )  # (1, seq_len, num_categories)

        next_cat = torch.argmax(pred_cat_logits[0, -1, :]).long()   # scalar ()
        cat_tokens.append(next_cat)
        cat_masks .append(next_cat != mask_token_id)

    # ------------------------------------------------------------
    # 5. stack generated tokens (skip baseline position 0)
    # ------------------------------------------------------------
    num_seq_tensor = torch.stack(num_tokens[1:], dim=0)           # (n_steps, n_num)
    cat_seq_tensor = torch.stack(cat_tokens[1:], dim=0)           # (n_steps,)

    # ------------- de-normalise & decode ------------------------
    num_np = num_seq_tensor.cpu().numpy()
    if scaler is not None:
        num_np = scaler.inverse_transform(num_np)

    cat_np   = cat_seq_tensor.cpu().numpy().astype(int)
    cat_text = le.inverse_transform(cat_np)

    # ------------- assemble DataFrame ---------------------------
    num_cols = lab_columns[:n_num]
    cat_col  = lab_columns[n_num]

    df_num = pd.DataFrame(num_np, columns=num_cols)
    df_cat = pd.DataFrame({cat_col: cat_text})

    generated_df = pd.concat([df_num, df_cat], axis=1)
    generated_df["avisitn"] = np.arange(1, len(generated_df) + 1)

    return generated_df
