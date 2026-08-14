# text/ — Text-layer concerns

This directory handles everything between "the user types a string" and "the
model prints text." The transformer itself only accepts and produces numbers,
so this layer performs the translation between human-readable text and those
numbers.

## First-pass map

```text
prompt string
    ↓ pt_encode()
prompt token IDs
    ↓ pt_forward() once per position
logits (one raw next-token score per vocabulary entry)
    ├─ prefill: use the known next prompt ID and repeat
    └─ decode: pt_sample() chooses a new ID
                    ↓ pt_decode()
              text/byte piece
```

A **token** is a numbered text piece. It may be a whole word, part of a word,
punctuation, or a single byte. The **vocabulary** is the fixed table that maps
between these pieces and their token IDs.

During **prefill**, the model processes the prompt tokens supplied by the user.
During **decode**, it repeatedly chooses and prints new tokens. These names
describe two phases of one generation loop, not two different models.

## Files

- `pt_text.h` — tokenizer, sampler, and generation API
- `pt_text.c` — all text-layer implementation

## What's here

- **Tokenizer**: load `tokenizer.bin`, use BPE to encode text (string → token
  IDs), and decode a token ID back into a text piece. **BPE**, or byte-pair
  encoding, repeatedly joins smaller neighboring pieces into vocabulary pieces.
- **Sampler**: turn the model's logits into a next-token choice. It supports
  greedy selection, temperature, and top-p (nucleus) sampling.
- **Generate**: run the prefill and decode phases and print each decoded piece
  immediately (`printk` uses serial output on the Pi and `printf` on a host).

## Suggested reading order

1. Read `pt_text.h` for the tokenizer and sampler data kept between calls.
2. In `pt_text.c`, read tokenizer loading and skim the sorting helpers.
3. Follow `pt_encode()` and `pt_decode()` to see both text translations.
4. Read `pt_sample()` for the next-token decision.
5. Finish with `pt_generate()`, which connects this layer to `pt_forward()`.

On the first pass, focus on that data flow. The exact Shell sort, heap sort,
random-number bit operations, and BPE search costs can be studied later.

## Dependencies

- `model/llama2.h` — config, weights, state, forward pass types
- `ops/core/pt_ops.h` — argmax, softmax (used by sampler)
- `ops/core/pt_math.h` — `pt_expf` (used by softmax inside the sampler)
- `libpi/` — printk, timer_get_usec, string functions

## Testing

```bash
cd dev/tests/generate && ./run.sh
```

The script assembles the QPU matrix-vector kernel and builds the Pi generation
integration and inference-benchmark programs. Running them on the target Pi is
a separate step.
