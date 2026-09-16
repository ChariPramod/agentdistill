"""Build the tiny BPE tokenizer fixture.

Run this to regenerate `tests/fixtures/tokenizer/`:

    python tests/fixtures/build_tokenizer.py

A purpose-built tokenizer keeps the mask tests hermetic (no Hub download in CI) and small (~60 KB rather than the
~2 MB of a real vocabulary), while still being a genuine fast BPE tokenizer with real offset mappings -- which is
the property the masking code depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent
OUT = HERE / "tokenizer"

SPECIALS = ["<|endoftext|>", "<|pad|>"]

CORPUS = [
    "You are a support agent for an online store. Answer using the tools provided.",
    "Where is the order for customer c_9? I placed it last week and it has not arrived.",
    "I will look that up. Let me check the order status for this customer right away.",
    'search_orders {"customer_id": "c_9", "limit": 5}',
    'refund_order {"order_id": "o_1", "reason": "damaged", "amount": 42.5}',
    'update_address {"order_id": "o_2", "street": "12 Main Street", "city": "Boston"}',
    'track_package {"tracking_number": "1Z999AA10123456784"}',
    'cancel_order {"order_id": "o_3"}',
    'escalate_to_human {"reason": "policy exception requested"}',
    '[{"order_id": "o_1", "status": "shipped", "carrier": "UPS", "eta": "2026-09-18"}]',
    '{"ok": true, "refund_id": "r_77", "amount": 42.5}',
    '{"error": "order not found"}',
    "Order o_1 shipped yesterday and should arrive on Thursday. Anything else I can help with?",
    "I have issued a refund of $42.50 to your original payment method. It takes three to five business days.",
    "Your new shipping address has been saved. The package will be redirected automatically.",
    "I am sorry about the delay. I have escalated this to a specialist who will contact you within one business day.",
    "Thanks, that is all I needed today.",
    "tool result assistant user system call end tools",
    "customer order refund address package tracking status amount reason limit id",
]
# The templates spell their markers in ordinary text, so the corpus must contain them for sane tokenization.
CORPUS += [
    "<|tools|> <|/tools|> <|assistant|> <|user|> <|system|> <|tool_result|> <|call|> <|/call|> <|end|>",
    "<|assistant turns=1|> <|assistant turns=2|> <|assistant turns=3|>",
]


def build() -> None:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1200,
        special_tokens=["<|unk|>", *SPECIALS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS * 40, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(OUT)

    # Attach the default (tool-capable) template so `AutoTokenizer.from_pretrained(OUT)` is usable as-is.
    template = (HERE / "templates" / "toolchat.jinja").read_text()
    cfg_path = OUT / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["chat_template"] = template
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {OUT} (vocab {fast.vocab_size})")


if __name__ == "__main__":
    build()
