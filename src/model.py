"""Load model and tokenizer. The pad token is set here and nowhere else."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device() -> torch.device:
    """cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(name_or_path: str, dev: torch.device | None = None):
    """fp32 model in eval mode on the best device, plus tokenizer with pad = eos, right padding."""
    dev = dev or device()
    tok = AutoTokenizer.from_pretrained(name_or_path)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch.float32).to(dev)
    model.eval()
    return model, tok
