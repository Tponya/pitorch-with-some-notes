#ifndef PITORCH_TEXT_H
#define PITORCH_TEXT_H

/*
 * Text layer: tokenizer, sampler, and generation loop.
 * Everything between "user types a string" and "model produces a string."
 *
 * Read the tokenizer declarations first, the sampler second, and pt_generate()
 * last. Implementations and control flow are in pt_text.c.
 */

#include <stdint.h>
#include "llama2.h"

/* ── tokenizer ── */

/*
 * C NOTE: Encoding needs to search by text while preserving each entry's
 * original token ID. This pair lets a separate index be sorted by string
 * without rearranging the vocabulary expected by the model.
 */
typedef struct {
    char *str;
    int id;
} pt_token_index_t;

/*
 * LLM NOTE: vocab[id] is the text piece belonging to a token ID. A piece is
 * not necessarily a word; it may be part of a word, punctuation, or a byte.
 * vocab_scores contains tokenizer merge preferences, not model logits or
 * next-token probabilities. sorted_vocab supports the opposite lookup from
 * text piece to ID.
 *
 * C NOTE: These pointers refer to fixed storage owned by pt_text.c. The
 * tokenizer does not allocate or free a private copy of each array.
 *
 * CAUTION: That storage is shared and has fixed capacities. Initializing a
 * second tokenizer replaces the tables used by the first tokenizer.
 */
typedef struct {
    char **vocab;
    float *vocab_scores;
    int vocab_size;
    int max_token_length;
    /* sorted index for encode (built lazily on first encode call) */
    int sorted_ready;
    pt_token_index_t *sorted_vocab;
} pt_tokenizer_t;

/*
 * FLOW: Parse a tokenizer payload that is already in memory. `data` points
 * directly to tokenizer.bin bytes, while vocab_size comes from the matching
 * model configuration. The loader copies the text into its fixed pools and
 * leaves `t` pointing to the prepared tables, so the source payload may be
 * overwritten afterward.
 */
void pt_tokenizer_init(pt_tokenizer_t *t, const void *data, int vocab_size);

/*
 * Load tokenizer from a combined model+tokenizer file.
 * The tokenizer binary immediately follows the model weights.
 * combined_file points to the start of the file (same pointer passed to pt_load_weights).
 * This locates the appended payload and then calls pt_tokenizer_init(); it does
 * not read from a filesystem here.
 */
void pt_load_tokenizer(pt_tokenizer_t *t, const void *combined_file, int vocab_size);

/*
 * Convert one token ID to a text or raw-byte piece. prev_token is needed for the
 * tokenizer's beginning-of-sequence spacing rule. The returned pointer refers
 * to internal storage; the caller must not free or modify it.
 */
const char *pt_decode(pt_tokenizer_t *t, int prev_token, int token);

/*
 * LLM NOTE: BPE encoding starts with small text pieces and repeatedly merges
 * neighboring pieces found in the vocabulary. `bos` and `eos` are integer
 * yes/no flags requesting special beginning/end token IDs.
 *
 * CAUTION: The caller supplies both the output array and `n_tokens`, where the
 * final count is written. This compact API has no capacity argument and cannot
 * detect an output array that is too small; the generation path provides 512
 * slots, and every other caller must make its own input-size guarantee.
 */
void pt_encode(pt_tokenizer_t *t, const char *text, int bos, int eos,
               int *tokens, int *n_tokens);

/* ── sampler ── */

/*
 * LLM NOTE: temperature controls how strongly the sampler favors high scores;
 * zero requests a deterministic highest-score choice and bypasses top-p. With
 * nonzero temperature, a topp value strictly between 0 and 1 limits random
 * selection to a high-probability group. rng_state remembers the position in a
 * reproducible pseudo-random number sequence.
 */
typedef struct {
    int vocab_size;
    float temperature;
    float topp;
    uint64_t rng_state;
} pt_sampler_t;

void pt_sampler_init(pt_sampler_t *s, int vocab_size,
                     float temp, float topp, uint64_t seed);

/*
 * Choose and return one token ID. When temperature is nonzero, this function
 * changes the supplied logits array in place while turning scores into
 * probabilities; the next model forward pass replaces those values.
 */
int pt_sample(pt_sampler_t *s, float *logits);

/* ── generation ── */

/*
 * FLOW: Encode the prompt, process its known tokens during prefill, then sample,
 * decode, and print new pieces one at a time. max_tokens limits total forward
 * positions (prompt plus generated positions) and is capped by cfg->seq_len;
 * it is not purely a count of newly printed tokens. matvec is the CPU or GPU
 * calculation function selected during model initialization.
 */
void pt_generate(const pt_config_t *cfg, const pt_weights_t *w, pt_state_t *s,
                 pt_tokenizer_t *tok, pt_sampler_t *sampler,
                 const char *prompt, int max_tokens, pt_matvec_fn matvec);

#endif
