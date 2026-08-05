#ifndef PITORCH_PT_H
#define PITORCH_PT_H

/*
 * pt.h — The public map of PiTorch's top-level model interface.
 *
 * A header file describes types and functions that other C files may use; it
 * is not an execution path and is not run by itself. A function declaration
 * ending in a semicolon announces what a function accepts and returns. Its
 * implementation is the full body inside { } in a .c file, mainly pt.c here.
 * examples/generate.c creates a pt_context_t and calls pt_pi_init(); training
 * programs use the same context with extra training storage enabled.
 *
 * API means "Application Programming Interface": the agreed set of names and
 * inputs that other files use to work with this part of PiTorch.
 *
 * Reading guide:
 *   1. Read pt_context_t as one organized package for a prepared model.
 *   2. Read the host/Pi initialization declarations as two setup choices.
 *   3. Read the small inference helpers.
 *   4. Skim the shard, training, and profiling declarations for now; their
 *      implementations belong to later stages of the learning plan.
 *
 * Read pt.c next to see how these declarations are implemented.
 */
/*
 * C NOTE: These first two lines and the final #endif form an include guard.
 * Before compilation, the preprocessor handles instructions beginning with #.
 * While preparing one .c file, #ifndef asks whether PITORCH_PT_H has already
 * been defined. The first visit defines it; another direct or indirect visit
 * skips to #endif. Separate .c files may still include this header normally.
 * This checking happens while building the program, not while it is running.
 */

/*
 * C NOTE: These headers provide the type names used below. Including them does
 * not cause the compiler to search for and run their matching .c files. During
 * the build, each selected .c file is compiled separately; the linker later
 * joins the compiled calls to the compiled function implementations.
 */
#include "llama2.h"
#include "pt_train.h"
#include "pt_shard.h"
#include "trace.h"

/*
 * FLOW: A caller creates one pt_context_t, an initialization function fills it,
 * and later model operations reuse it. Think of it as an organizer for a model
 * that is ready to run—not as a copy of every model number.
 *
 * C NOTE: typedef names this struct layout pt_context_t; it is a type name, not
 * an actual model object. `pt_context_t ctx;` creates one object named ctx.
 * A line such as `pt_config_t cfg;` below declares a field named cfg whose type
 * is pt_config_t; it is not a function. `ctx.cfg` names the cfg field inside
 * the actual struct: reading it reads that field, and assigning to it changes
 * the same struct. The dot itself does not make a copy.
 *
 * A declaration such as `pt_context_t *ctx` says that the parameter can hold
 * the address of a pt_context_t; it does not discover an address automatically.
 * A caller supplies one with &model, as in `pt_pi_init(&model, ...)`. The setup
 * function can then use ctx->cfg to reach and change a field inside the caller's
 * actual context variable.
 * This project is procedural C: its functions perform steps and deliberately
 * update structs. Merely organizing code into functions is not the separate
 * programming style called "functional programming."
 */
typedef struct {
    /*
     * LLM NOTE: cfg is the model's blueprint (sizes such as layer count and
     * vocabulary size). w stores addresses of the learned weights already in
     * memory. shared_weights records whether two model parts reuse one weight
     * table. matvec selects the CPU or GPU routine for the model's heavy
     * matrix-vector calculations.
     */
    pt_config_t       cfg;
    pt_weights_t      w;
    int               shared_weights;
    pt_matvec_fn      matvec;

    /*
     * LLM NOTE: Inference means using the model to make predictions without
     * changing its learned weights. state describes its working memory. This
     * includes temporary activation vectors (lists of numbers calculated
     * between neural-network layers) and the KV cache, information deliberately
     * retained from earlier token positions. pos is the current token position
     * used by the one-step helper pt_forward_step().
     */
    pt_state_t        state;
    int               pos;        /* current decode position (for pt_forward_step) */

    /*
     * LLM NOTE: These fields are needed only when changing the model through
     * training. max_T is the largest training sequence length for which memory
     * was prepared. Passing max_T == 0 during initialization leaves this
     * training section unused; study its details in the later training stage.
     */
    pt_activations_t  acts;
    pt_grads_t        grads;
    pt_backward_buf_t bb;
    int               max_T;

    /*
     * A trace is an optional timing record used to measure model operations.
     * C NOTE: A pointer value of NULL means it does not point to a trace.
     */
    pt_trace_t       *trace;

    /* Number of Raspberry Pi GPU workers; 0 on a desktop/host build. */
    int               num_qpus;
} pt_context_t;

/*
 * C NOTE: The preprocessor chooses one setup interface while building:
 *   - without __RPI__, compile the desktop/host declarations below;
 *   - with __RPI__, skip to the Raspberry Pi declarations after #else.
 * Only the chosen section becomes part of that build.
 *
 * In both setup functions, void *weight_data is a generic memory address. The
 * setup code knows how to interpret the model configuration and weights stored
 * at that address. The lines ending in ; below are declarations, not calls:
 * they let callers compile while the linker later connects them to the complete
 * implementations in pt.c.
 */
#ifndef __RPI__
/*
 * FLOW: Prepare a model on a normal computer. The host memory allocator supplies
 * memory when pt_host_init() requests it; pt_free() releases those working
 * buffers back to that allocator so they can be reused. The caller still owns
 * weight_data. The Pi has no matching ordinary cleanup call because it uses
 * fixed memory regions rather than memory lent by an operating system, and this
 * bare-metal program normally keeps running until reset or power-off.
 *
 * ctx:         organizer to fill in.
 * weight_data: address of the model file after it has been loaded into RAM.
 * max_T:       0 for inference only; greater than 0 prepares training storage
 *              for sequences up to that length.
 */
void pt_host_init(pt_context_t *ctx, void *weight_data, int max_T);
void pt_free(pt_context_t *ctx);
#else
/*
 * HARDWARE NOTE: Prepare a model on the Raspberry Pi. pt_pi_init() reads the
 * model description, records where its weights are, reserves working memory,
 * enables GPU help for heavy calculations, and fills ctx. It does not load the
 * tokenizer; generate.c loads that before this call because tokenization belongs
 * to the separate text-generation layer.
 *
 * ctx:         organizer to fill in.
 * weight_data: address of the model bytes already placed in RAM, such as
 *              0x02000000 by the SD-card setup.
 * num_qpus:    number of GPU workers to use, normally 12 on this hardware.
 * max_T:       0 for inference only; greater than 0 also prepares training.
 * arena_bytes: size of the temporary GPU calculation workspace.
 */
void pt_pi_init(pt_context_t *ctx, void *weight_data,
                int num_qpus, int max_T, unsigned arena_bytes);

/*
 * LLM NOTE: This advanced setup is for a model divided across several Pis.
 * A shard is one Pi's portion of the model. The function prepares only that
 * portion and writes a description of it into shard_out. Study this with the
 * later distributed-execution files; ordinary single-Pi generation uses
 * pt_pi_init() above.
 */
void pt_pi_init_shard(pt_context_t *ctx, pt_shard_info_t *shard_out,
                      void *shard_data, int num_qpus, int max_T,
                      unsigned arena_bytes);
#endif

/* ── Inference helpers: small operations on a prepared context ── */

/*
 * LLM NOTE: Process one input token at the current position through the model's
 * transformer layers. The result is one raw score, called a logit, for every
 * possible next token; this helper uses argmax to return the token with the
 * highest score. It is a lower-level alternative to the complete text loop in
 * pt_generate(), whose sampler also supports random choices. This helper then
 * advances ctx->pos by one.
 */
int pt_forward_step(pt_context_t *ctx, int token);

/*
 * Start a fresh sequence by clearing information cached from earlier tokens
 * and moving the current position back to zero.
 */
void pt_reset_kv(pt_context_t *ctx);

/*
 * Print a short summary of the model blueprint and compute device.
 * C NOTE: const promises this function will not change ctx through this pointer.
 */
void pt_print_config(const pt_context_t *ctx);

/* ── Desktop/host-only file helper ─────────────────────────── */

#ifndef __RPI__
/*
 * Read a complete file into RAM and return its starting address. If out_size
 * is not NULL, also write the number of bytes there. This helper is unnecessary
 * on the Pi, where the boot setup has already placed model data in RAM.
 */
void *pt_read_file(const char *path, long *out_size);
#endif

/* ── Training: intentionally brief until the later training stage ── */

/*
 * LLM NOTE: Perform one model-learning step on T token positions and return a
 * loss value, a number measuring how wrong the predictions were. lr is the
 * learning rate, which controls the size of the weight update. This requires
 * initialization with max_T greater than or equal to T.
 */
float pt_train_step(pt_context_t *ctx, const int *tokens, int T, float lr);

/* ── Profiling: optional timing measurements ───────────────── */

/*
 * Enable operation-timing records in a trace. On a host, passing NULL asks this
 * function to allocate trace storage; on the Pi, the caller must supply it.
 * pt_disable_trace() stops recording but does not release the storage.
 */
void pt_enable_trace(pt_context_t *ctx, pt_trace_t *preallocated);
void pt_disable_trace(pt_context_t *ctx);

#endif
