"""Loss functions. Each takes (model, batch, aux) and returns a scalar; the training
loop does not know which one it is running.

`batch` is the forget batch for unlearning losses, or the only batch for plain LM
training. `aux` is the retain batch, present when a retain regularizer is in use.
Labels are -100 on prompt and pad, so every loss here acts on answer tokens only.
"""

import torch
import torch.nn.functional as F


def lm_loss(model, batch, aux=None, replay_weight=1.0):
    """Mean cross-entropy over answer tokens, plus a general-text rehearsal term.

    Without the `aux` (replay) term, ten epochs on 8.7k short templated sentences drives
    training loss to ~0.01 and WikiText perplexity from 48.75 to 9033: the model learns
    the facts and forgets English. See HANDOFF decision 18.
    """
    loss = model(**batch).loss
    if aux is not None:
        loss = loss + replay_weight * model(**aux).loss
    return loss


def seq_logprob(model, batch):
    """Per-example sum of log p(token) over answer tokens. Shape (B,)."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
    tgt = batch["labels"][:, 1:]
    mask = tgt != -100
    lp = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return (lp * mask).sum(-1)


def gradient_ascent(model, batch, aux=None, retain_weight=1.0):
    """GA: maximize loss on the forget batch. With `aux`, this is gradient difference
    (GA+GD): the retain term is ordinary training that holds utility up."""
    loss = -model(**batch).loss
    if aux is not None:
        loss = loss + retain_weight * model(**aux).loss
    return loss


def npo(model, batch, aux=None, *, ref_model, beta=0.1, retain_weight=1.0):
    """NPO (Zhang et al., arXiv 2404.05868): treat forget examples as dispreferred in a
    DPO-style objective against a frozen reference.

        L = (2/beta) * E[ log(1 + (p_theta / p_ref)^beta ) ]

    Bounded, unlike GA, so it does not collapse the model. With `aux`, NPO+GD.
    """
    lp = seq_logprob(model, batch)
    with torch.no_grad():
        lp_ref = seq_logprob(ref_model, batch)
    loss = -(2.0 / beta) * F.logsigmoid(-beta * (lp - lp_ref)).mean()
    if aux is not None:
        loss = loss + retain_weight * model(**aux).loss
    return loss


def build(method, *, ref_model=None, beta=0.1, retain_weight=1.0):
    """Name from a config to a loss callable."""
    if method == "lm":
        return lm_loss
    if method == "ga_gd":
        return lambda m, b, aux=None: gradient_ascent(m, b, aux, retain_weight=retain_weight)
    if method == "npo_gd":
        assert ref_model is not None, "npo_gd needs a frozen reference model"
        return lambda m, b, aux=None: npo(m, b, aux, ref_model=ref_model, beta=beta, retain_weight=retain_weight)
    raise ValueError(f"unknown method {method!r}")
