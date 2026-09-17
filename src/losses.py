"""Loss functions. Each takes (model, batch) and returns a scalar. The training loop
does not know which one it is running.

Batches come from data.encode: labels are -100 on prompt and pad, so every loss
here acts on answer tokens only.
"""


def lm_loss(model, batch):
    """Mean cross-entropy over answer tokens. Phase 1 and Phase 4, and the retain term of GD."""
    return model(**batch).loss
