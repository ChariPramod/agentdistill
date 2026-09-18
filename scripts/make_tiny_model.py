"""Build the tiny model that the CPU rehearsal trains.

A randomly-initialized Llama-shaped model over the repo's fixture tokenizer. It cannot do the task and is not
supposed to: the rehearsal checks that every stage executes, not that anything learns. Random weights are the
honest choice here, because a rehearsal that produced plausible-looking numbers would invite someone to read
them.

The alternative -- a real ~0.5B instruct model -- downloads several hundred megabytes and needs network access,
which puts the rehearsal behind exactly the kind of setup step it exists to avoid.

    python scripts/make_tiny_model.py --out artifacts/tiny/model
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "tests" / "fixtures" / "tokenizer"
#: The tiny model's stand-in for `train.base_model_revision`. The model is a local path, so a Hub revision means
#: nothing for it; what pins it is that a clean rehearsal rebuilds byte-identical weights from this seed.
SEED = 0


def build(out: Path, tokenizer_dir: Path = TOKENIZER_DIR, seed: int = SEED) -> Path:
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    config = LlamaConfig(
        vocab_size=max(tok.vocab_size, len(tok)),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        # Long enough for the tiny config's max_seq_len; the rehearsal truncates rather than trains on more.
        max_position_embeddings=2048,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    out.mkdir(parents=True, exist_ok=True)
    # Seeded right before initialization, so nothing that ran earlier in the process moves the weights.
    torch.manual_seed(seed)
    LlamaForCausalLM(config).save_pretrained(out)
    tok.save_pretrained(out)
    (out / "TINY").write_text(
        "Randomly initialized. It cannot do the task and its outputs mean nothing.\n"
        "Built by scripts/make_tiny_model.py for the CPU rehearsal.\n"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "tiny" / "model")
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    args = ap.parse_args(argv)

    if (args.out / "config.json").exists():
        print(f"tiny model already at {args.out}")
        return 0
    path = build(args.out, args.tokenizer)
    print(f"tiny model at {path} — randomly initialized, its outputs mean nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
