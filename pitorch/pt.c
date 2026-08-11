/*
 * pt.c — Build and operate PiTorch's top-level model context.
 *
 * This file supplies the function bodies announced in pt.h. It is not an
 * entry point and does not run from top to bottom by itself. Callers such as
 * examples/generate.c enter a particular function—most importantly
 * pt_pi_init()—and that function coordinates lower-level model, memory, and
 * hardware helpers.
 *
 * The main setup flow is:
 *
 *   model bytes already in RAM
 *       -> copy seven configuration integers and record weight addresses
 *       -> reserve working memory
 *       -> select CPU or GPU-backed matrix calculations
 *       -> leave the caller's existing pt_context_t filled and ready to use
 *
 * Reading guide:
 *   1. Follow the Pi matrix-vector wrapper and simple scratch allocator.
 *   2. Read pt_pi_init() as the single-Pi setup used by generate.c.
 *   3. Skim pt_pi_init_shard(); it previews later multi-Pi work.
 *   4. Compare the shorter desktop/host setup and cleanup path.
 *   5. Finish with the shared inference, file, tracing, and training helpers.
 *
 * The detailed transformer math and allocation helpers are signposts into
 * later learning stages. After this file, return to generate.c to trace the
 * complete setup once, then continue with pitorch/text/README.md.
 */

/*
 * C NOTE: pt.h supplies this file's public declarations and context type.
 * pt_ops.h supplies small math operations such as argmax() and the host CPU
 * matrix-vector routine. string.h supplies memset(), which fills bytes in a
 * memory region. The preprocessor includes ordinary file and heap services
 * only for a non-Pi build.
 */
#include "pt.h"
#include "pt_ops.h"
#include <string.h>
#ifndef __RPI__
#include <stdlib.h>
#include <stdio.h>
#endif

/*
 * C NOTE: __RPI__ is selected when the program is built, not while it runs.
 * A Pi build keeps the hardware section below and discards the host section
 * after #else. A host build does the reverse.
 */
#ifdef __RPI__
/*
 * HARDWARE NOTE: These headers expose bare-metal Pi services. The ARM CPU
 * remains in control; selected matrix-vector calculations are handed to QPUs,
 * the VideoCore GPU's parallel calculation workers. The GPU arena is temporary
 * GPU-visible workspace and is separate from the longer-lived model state
 * allocated later in this file.
 */
#include "rpi.h"
#include "gpu.h"
#include "arena.h"
#include "mailbox.h"
#include "profiler.h"
#include "matvec.h"

/* ── Pi globals for GPU matvec wrapper ──────────────────────── */

/*
 * FLOW: The same transformer code is used on the Pi and on a host computer;
 * that is what "portable" means here. It needs a matrix-vector function that
 * accepts W, x, y, out_dim, and in_dim and returns no value. This required list
 * of input types and return type is called a function signature.
 *
 * matvec_gpu() is a wrapper: a small helper around another function. It matches
 * the signature the transformer expects, prepares the temporary GPU workspace,
 * and then calls the Pi-specific GPU routine. That signature has no place for
 * a QPU count, so the shared variable below remembers the setup choice.
 *
 * C NOTE: static at file level makes a name private to this .c file while its
 * value lasts for the program's lifetime. This shared setting means the Pi path
 * is designed around one active model setup rather than independent contexts.
 */
static int g_num_qpus;

/*
 * LLM NOTE: A matrix-vector multiplication combines a large weight matrix W
 * with an input vector x to produce y. Transformers perform this operation many
 * times, so it is the part accelerated here; the entire model is not moved from
 * the CPU to the GPU.
 *
 * HARDWARE NOTE: Resetting the arena rewinds its temporary allocation position
 * so this multiplication can reuse the workspace left by the previous one. It
 * does not erase the model weights. The final argument could point to a perf_t
 * record containing timing and GPU-cycle measurements. NULL is a special
 * pointer value meaning "no object is supplied," so this call performs the
 * calculation without saving that optional record.
 */
static void matvec_gpu(const float *W, const float *x, float *y,
                       int out_dim, int in_dim) {
    gpu_arena_reset();
    smatvec_tmu(W, x, y, out_dim, in_dim, g_num_qpus, NULL);
}

/* ── Pi bump allocator (starting address calculated in pt_pi_init) ── */

/*
 * HARDWARE NOTE: Bare-metal code has no operating system providing ordinary
 * malloc() here. This small "bump allocator" divides one ARM-memory region.
 * A buffer is simply an area reserved for holding data. state_base is the first
 * address available for these buffers; scratch_off is the number of bytes from
 * that starting address that have already been assigned.
 *
 * This is different from the GPU arena above: scratch_alloc() provides
 * long-lived inference or training buffers, while the GPU arena is temporary
 * calculation workspace reset before each matrix-vector operation.
 *
 * All of these ARM buffers are created together and reused for as long as the
 * model runs. Their contents are overwritten or cleared when needed, but their
 * memory remains assigned to the model. Individual free operations would add
 * bookkeeping that this fixed, same-lifetime layout does not need; setup can
 * reuse the entire region at once by returning scratch_off to zero.
 */
static unsigned state_base;
static unsigned scratch_off;

/*
 * FLOW: Allocation helpers in llama2.c and pt_train.c call this function when
 * they need a buffer of a certain byte size. It gives them the next unused
 * address, then moves scratch_off forward so the following buffer will begin
 * after this one.
 *
 * C NOTE: void * is a generic address. `(bytes + 15u) & ~15u` rounds a size up
 * to a multiple of 16 bytes; this keeps each following address 16-byte aligned.
 * Alignment means the address is divisible by the chosen boundary. unsigned is
 * an integer type that cannot represent negative values, which suits addresses,
 * byte counts, and offsets. The `u` marks an unsigned number.
 *
 * CAUTION: This helper does not check its limit on each request. The setup
 * functions calculate the final required address and compare it with the Pi's
 * available ARM RAM below.
 */
static void *scratch_alloc(unsigned bytes) {
    void *p = (void *)(state_base + scratch_off);
    scratch_off += (bytes + 15u) & ~15u;
    return p;
}

/*
 * FLOW: Prepare the single-Pi context used by examples/generate.c. Its
 * notmain() function creates the actual ctx variable and supplies &ctx as this
 * function's first argument. "Supplies" or "passes" means placing a value into
 * a function call. &ctx is the address where that original struct lives; the
 * ctx parameter below holds that address and can change the original fields.
 * This function returns no value, but when it finishes, execution resumes in
 * notmain() with that same struct filled and ready to use.
 *
 * weight_data is not the model bytes themselves. It is a pointer containing the
 * address of the first byte. At that address are the checkpoint's raw bytes:
 * seven configuration integers followed by millions of learned weight values.
 * The loader interprets them by treating agreed byte positions as particular
 * integers or floating-point arrays.
 *
 * A memory layout is the plan assigning each range of RAM a purpose, such as
 * model weights, inference state, or GPU workspace. num_qpus chooses the GPU
 * worker count. max_T == 0 requests inference state; a positive value also
 * reserves training data. arena_bytes sizes the temporary GPU workspace.
 *
 * CAUTION: ctx stores pointers into weight_data rather than owning a copy, so
 * those model bytes must remain in place for as long as the context is used.
 */
void pt_pi_init(pt_context_t *ctx, void *weight_data,
                int num_qpus, int max_T, unsigned arena_bytes) {
    /*
     * C NOTE: `pt_context_t ctx;` in notmain() initially contains unspecified
     * leftover bits. memset gives the struct a known starting condition by
     * writing zero into all of it. Integer fields such as pos and max_T become
     * zero. Pointer fields—fields meant to hold addresses—become null until the
     * later setup assigns real addresses. This does not clear model weights,
     * which live in the separate region reached through weight_data.
     * sizeof(*ctx) means "the size of the struct at this address," not the
     * smaller size of the address itself.
     */
    memset(ctx, 0, sizeof(*ctx));

    /*
     * HARDWARE NOTE: Enable the QPUs and request the GPU-visible arena used by
     * matvec_gpu(). A performance counter is a hardware measurement such as
     * elapsed microseconds, executing cycles, idle cycles, or stalled cycles.
     * perf_init() prepares those counters in case code requests a performance
     * record later; it does not start measuring this setup code.
     */
    qpu_enable();
    perf_init();
    gpu_arena_init(arena_bytes);

    /* Keep the dispatch setting and a reportable copy in the context. */
    g_num_qpus = num_qpus;
    ctx->num_qpus = num_qpus;

    /*
     * FLOW: Make a quick sanity check before treating this address as a model.
     * The first 32-bit integer in the checkpoint is dim, the width of its main
     * vectors. An impossible value often means the SD image was absent or put
     * at the wrong address. This is only a plausibility check, not complete
     * validation of every header value or byte in the file.
     *
     * C NOTE: `(volatile int *)weight_data` says to treat the generic address
     * as an address of an int; the leading * then reads the int stored there.
     * volatile requires a real read from that RAM address. panic() prints the
     * diagnostic and stops normal execution on this bare-metal path.
     */
    /* validate weight data */
    int dim0 = *(volatile int *)weight_data;
    if (dim0 <= 0 || dim0 > 65536)
        panic("no weights at 0x%x (dim=%d)\n"
              "SD card: initramfs weights/<model>.bin 0x2000000\n",
              (unsigned)weight_data, dim0);

    /*
     * FLOW: Copy the seven small configuration values, inspect whether the
     * input embedding and output classifier share a weight table, then fill
     * ctx->w with addresses of the large weight arrays already inside RAM. The
     * embedding turns a token ID into an initial vector; the classifier produces
     * next-token scores. Despite its name, pt_load_weights() does not copy all
     * those arrays.
     *
     * C NOTE: Array positions start at zero, so [5] reads the sixth header int.
     * Casting to const int * promises this expression will only read the bytes.
     * Follow the two loaders in llama2.c later for the binary offsets.
     */
    pt_load_config(&ctx->cfg, weight_data);
    ctx->shared_weights = ((const int *)weight_data)[5] > 0;
    pt_load_weights(&ctx->w, &ctx->cfg, weight_data);
    ctx->matvec = matvec_gpu;
    ctx->max_T  = max_T;

    /*
     * FLOW: Build the long-lived ARM-memory map manually. pt_file_size()
     * calculates the complete model-checkpoint span from the header; it does
     * not ask a filesystem. The expression chooses the following 1 MiB-aligned
     * address, and scratch_off = 0 starts allocation there.
     *
     *   [model checkpoint bytes][alignment gap][state / training buffers]
     *
     * CAUTION: The combined generation image may append tokenizer bytes after
     * the weights, but pt_file_size() does not count them. New state can
     * therefore overlap those original tokenizer bytes. generate.c deliberately
     * copies the tokenizer into its own arrays before calling this function.
     * This address arithmetic also assumes the Pi's 32-bit memory layout; it is
     * not a general pattern for converting pointers on a 64-bit host.
     */
    /* place buffers after the weights (1MB aligned) */
    unsigned weight_base = (unsigned)weight_data;
    unsigned file_size = pt_file_size(weight_data);
    state_base = (weight_base + file_size + 0x100000) & ~0xFFFFF;
    scratch_off = 0;

    /*
     * LLM NOTE: Training needs far more memory than inference. It must save
     * activations (intermediate layer results) from many token positions and
     * calculate gradients describing how the loss responds to learned weights.
     * It also needs temporary backward-calculation buffers. Inference needs
     * current vectors, logits (raw next-token scores), and a KV cache, but no
     * gradients or saved training history.
     */
    if (max_T > 0) {
        /* ── training mode ── */
        /*
         * FLOW: Let the training helpers request consecutive slices from the
         * bump allocator. These hold saved activations and temporary backward
         * data; their exact array layouts belong to the later training stage.
         */
        /* All scratch allocations must come before grad_base to avoid overlap */
        pt_scratch_alloc_activations(&ctx->acts, &ctx->cfg, max_T, scratch_alloc);
        pt_scratch_alloc_backward_buf(&ctx->bb, &ctx->cfg, max_T, scratch_alloc);

        int tmp_sz = ctx->cfg.dim > ctx->cfg.hidden_dim
                   ? ctx->cfg.dim : ctx->cfg.hidden_dim;
        ctx->bb.d_temp = scratch_alloc(tmp_sz * 4);

        pt_scratch_alloc_state(&ctx->state, &ctx->cfg, scratch_alloc);

        /*
         * FLOW: Place the large gradient array at the next 1 MiB boundary after
         * every scratch request. Choosing this address only after those requests
         * prevents the two manually managed regions from overlapping.
         */
        /* grads: 1MB-aligned, placed after all scratch allocations */
        unsigned grad_base = (state_base + scratch_off + 0xFFFFF) & ~0xFFFFF;
        pt_scratch_alloc_grads(&ctx->grads, &ctx->cfg, ctx->shared_weights, grad_base);

        /*
         * LLM NOTE: GPU-assisted backward calculations sometimes need a weight
         * matrix rearranged (transposed). Reserve one reusable area large enough
         * for the larger candidate rather than storing every transpose. The
         * gradient count is in floats, so multiplying by 4 converts it to bytes;
         * the final mask aligns the next address to 16 bytes.
         */
        /* w_transpose: placed after grads in memory */
        unsigned wt_size = ctx->cfg.vocab_size * ctx->cfg.dim;
        unsigned ht_size = ctx->cfg.hidden_dim * ctx->cfg.dim;
        unsigned tb_elems = wt_size > ht_size ? wt_size : ht_size;
        unsigned tb_base = grad_base + (unsigned)ctx->grads._n_params * 4;
        tb_base = (tb_base + 15u) & ~15u;
        ctx->bb.w_transpose = (float *)tb_base;

        /*
         * HARDWARE NOTE: There is no operating system here to reject an
         * oversized allocation. arm_ram_end() reports the first address beyond
         * RAM assigned to the ARM CPU. Compute the farther end of the two memory
         * regions and stop if it crosses that boundary. `>> 20` converts bytes
         * to a compact MiB-sized display value.
         *
         * CAUTION: This is a final layout check; some allocation helpers have
         * already assigned pointers and cleared caches or gradients.
         */
        /* verify firmware gave us enough ARM RAM */
        uint32_t ram_end = arm_ram_end();
        uint32_t state_need = state_base + scratch_off;
        uint32_t grad_need  = tb_base + (unsigned)tb_elems * 4;
        uint32_t need = state_need > grad_need ? state_need : grad_need;
        printk("memory: state=0x%x grads=0x%x end=0x%x ram=0x%x (%dMB)\n",
               state_base, grad_base, need, ram_end, ram_end >> 20);
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x — check fixup.dat\n", need, ram_end);
    } else {
        /*
         * FLOW: Inference only needs current calculation vectors, one logit per
         * vocabulary token, and the KV cache. The cache retains attention keys
         * and values from earlier positions so they need not be recomputed for
         * every new token. The allocator also clears both caches.
         */
        /* ── inference only ── */
        pt_scratch_alloc_state(&ctx->state, &ctx->cfg, scratch_alloc);

        uint32_t ram_end = arm_ram_end();
        uint32_t need = state_base + scratch_off;
        printk("memory: state=0x%x end=0x%x ram=0x%x (%dMB)\n",
               state_base, need, ram_end, ram_end >> 20);
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x — check fixup.dat\n", need, ram_end);
    }

    pt_print_config(ctx);
}

/*
 * FLOW: Advanced preview—prepare one model shard for pipeline execution across
 * several Pis. A shard contains only one board's portion of the model. That
 * board is called a rank; it processes its assigned transformer layers and the
 * surrounding distributed code passes the current activation vector between
 * ranks. This reduces the weights and per-layer memory stored on each board,
 * although communication can add overhead.
 *
 * shard_out receives this rank's number, global layer range, local layer count,
 * and ownership flags. The embedding changes an initial token ID into a model
 * vector; the output head changes the final vector into vocabulary logits. Only
 * ranks whose shard flags assign them the embedding or output head store those
 * components. Follow pt_shard.c in the later distributed stage; for now, notice
 * that this repeats the ordinary Pi setup while sizing locally owned pieces.
 */
void pt_pi_init_shard(pt_context_t *ctx, pt_shard_info_t *shard_out,
                      void *shard_data, int num_qpus, int max_T,
                      unsigned arena_bytes) {
    memset(ctx, 0, sizeof(*ctx));
    memset(shard_out, 0, sizeof(*shard_out));

    qpu_enable();
    perf_init();
    gpu_arena_init(arena_bytes);

    g_num_qpus = num_qpus;
    ctx->num_qpus = num_qpus;

    /* validate shard data */
    int dim0 = *(volatile int *)shard_data;
    if (dim0 <= 0 || dim0 > 65536)
        panic("no shard at 0x%x (dim=%d)\n", (unsigned)shard_data, dim0);

    /*
     * FLOW: Keep two views of the model. global_cfg describes the original
     * complete model, while shard_out identifies the layers and edge components
     * present in this rank's file. pt_load_shard_weights() fills ctx->w with
     * addresses into the already resident shard; it does not copy the weights.
     */
    /* Read global config + shard info */
    pt_config_t global_cfg;
    pt_load_shard_header(&global_cfg, shard_out, shard_data);

    ctx->shared_weights = ((const int *)shard_data)[5] > 0;

    /* Load shard weights */
    pt_load_shard_weights(&ctx->w, &global_cfg, shard_out, shard_data);

    ctx->matvec = matvec_gpu;
    ctx->max_T  = max_T;

    /*
     * C NOTE: Assigning one whole struct to another does make a field-by-field
     * copy. After that copy, only ctx's n_layers is changed to the local count.
     * Allocation helpers must size per-layer arrays for this Pi, while vector,
     * vocabulary, attention, and sequence dimensions remain those of the full
     * model. The global layer identities remain recorded in shard_out.
     */
    /* Set cfg.n_layers = n_local for allocation sizing.
     * All other fields (dim, hidden, vocab, seq_len) stay global. */
    ctx->cfg = global_cfg;
    ctx->cfg.n_layers = shard_out->n_local;

    /* Place buffers after the shard (1 MB aligned) */
    unsigned shard_base = (unsigned)shard_data;
    unsigned file_size = pt_shard_file_size(shard_data);
    state_base = (shard_base + file_size + 0x100000) & ~0xFFFFF;
    scratch_off = 0;

    if (max_T > 0) {
        /*
         * LLM NOTE: Saved activations and backward buffers are sized for only
         * this rank's transformer layers. This particular batched-training path
         * uses those saved activations instead of an autoregressive KV cache, so
         * it deliberately skips ordinary inference-state allocation here.
         */
        /* Activations and backward buffers sized for LOCAL layers */
        pt_scratch_alloc_activations(&ctx->acts, &ctx->cfg, max_T, scratch_alloc);
        pt_scratch_alloc_backward_buf(&ctx->bb, &ctx->cfg, max_T, scratch_alloc);

        int tmp_sz = ctx->cfg.dim > ctx->cfg.hidden_dim
                   ? ctx->cfg.dim : ctx->cfg.hidden_dim;
        ctx->bb.d_temp = scratch_alloc(tmp_sz * 4);

        /* No KV cache needed for training (skip pt_scratch_alloc_state) */

        /*
         * FLOW: Begin gradient storage after the earlier scratch requests. A
         * separate grad_cfg copy controls what the allocation helper reserves;
         * changing that copy does not change ctx->cfg.
         */
        /* Grads for the components this rank will train. */
        unsigned grad_base = (state_base + scratch_off + 0xFFFFF) & ~0xFFFFF;

        /*
         * C NOTE: This assignment copies the struct. `!has_embed` means "does
         * not have the embedding." Setting the copy's vocabulary size to zero
         * is an allocation-control signal, not a change to the real model.
         */
        /* Build a grad config: only include embed if has_embed */
        pt_config_t grad_cfg = ctx->cfg;  /* n_layers already = n_local */
        if (!shard_out->has_embed) {
            /* No embedding-table gradient allocation is requested on this rank. */
            grad_cfg.vocab_size = 0;
        }
        pt_scratch_alloc_grads(&ctx->grads, &grad_cfg, ctx->shared_weights, grad_base);

        /* Assigning 0 makes these null pointers, marking unavailable gradients. */
        /* If no embed, null out token_embedding grad pointer */
        if (!shard_out->has_embed) {
            ctx->grads.token_embedding = 0;
            ctx->grads.wcls = 0;
        }
        /* If no head, null out rms_final grad pointer */
        if (!shard_out->has_head)
            ctx->grads.rms_final_weight = 0;

        /*
         * LLM NOTE: Reserve one reusable transposition workspace for the largest
         * matrix relevant to this rank. A rank with the embedding or output head
         * may need a vocabulary-sized matrix; a layer-only rank needs only its
         * transformer-layer shapes. This does not store every transpose at once.
         */
        /* w_transpose: sized for owned components only */
        unsigned wt_elems;
        if (shard_out->has_head || shard_out->has_embed) {
            unsigned vt = (unsigned)global_cfg.vocab_size * global_cfg.dim;
            unsigned ht = (unsigned)global_cfg.hidden_dim * global_cfg.dim;
            wt_elems = vt > ht ? vt : ht;
        } else {
            /* Layer-only rank: just need max layer weight dim */
            unsigned dt = (unsigned)global_cfg.dim * global_cfg.dim;
            unsigned ht = (unsigned)global_cfg.hidden_dim * global_cfg.dim;
            wt_elems = dt > ht ? dt : ht;
        }
        unsigned tb_base = grad_base + (unsigned)ctx->grads._n_params * 4;
        tb_base = (tb_base + 15u) & ~15u;
        ctx->bb.w_transpose = (float *)tb_base;

        /* Verify RAM */
        uint32_t ram_end = arm_ram_end();
        uint32_t need = tb_base + (unsigned)wt_elems * 4;
        printk("shard memory: state=0x%x grads=0x%x wt_end=0x%x ram=0x%x (%dMB)\n",
               state_base, grad_base, need, ram_end, ram_end >> 20);
        printk("  shard=%dMB acts+bwd=%dKB grads=%dMB wt=%dMB\n",
               file_size >> 20, scratch_off >> 10,
               (ctx->grads._n_params * 4) >> 20, (wt_elems * 4) >> 20);
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x\n", need, ram_end);
    } else {
        /*
         * FLOW: Inference allocates current-token vectors, logits, and a KV
         * cache for the local layer count. Layer-independent vector and
         * vocabulary dimensions still come from the complete model.
         */
        /* Inference-only with shard */
        pt_scratch_alloc_state(&ctx->state, &ctx->cfg, scratch_alloc);

        uint32_t ram_end = arm_ram_end();
        uint32_t need = state_base + scratch_off;
        if (need > ram_end)
            panic("need 0x%x but firmware gave 0x%x\n", need, ram_end);
    }

    pt_print_config(ctx);
}

#else /* Mac host */

/*
 * FLOW: This is the non-Raspberry-Pi host path, currently intended for a Mac.
 * It produces the same kind of prepared context as pt_pi_init(), but uses
 * dynamically allocated host memory and the CPU matrix-vector routine instead
 * of a manual Pi memory map and QPUs.
 *
 * pt_load_config() copies the small header into cfg. pt_load_weights() only
 * records addresses inside weight_data, so the caller must keep that buffer
 * alive while ctx is used. A positive max_T additionally prepares training
 * storage; ordinary inference state is allocated in either mode.
 *
 * CAUTION: This function expects a fresh context. Calling it again on an
 * initialized context without pt_free() would lose the old allocation addresses.
 * Training changes weights in the RAM buffer; saving them to disk is separate.
 */
void pt_host_init(pt_context_t *ctx, void *weight_data, int max_T) {
    memset(ctx, 0, sizeof(*ctx));

    pt_load_config(&ctx->cfg, weight_data);
    ctx->shared_weights = ((const int *)weight_data)[5] > 0;
    pt_load_weights(&ctx->w, &ctx->cfg, weight_data);
    ctx->matvec = smatvec_cpu;
    ctx->max_T  = max_T;

    /*
     * LLM NOTE: Training must retain activations, gradients, and backward
     * workspace for as many as max_T positions. These helpers obtain ordinary
     * host heap—the runtime-managed memory pool used by malloc(). Study their
     * internal layouts in the training stage.
     */
    if (max_T > 0) {
        pt_alloc_activations(&ctx->acts, &ctx->cfg, max_T);
        pt_alloc_grads(&ctx->grads, &ctx->cfg, ctx->shared_weights);
        pt_alloc_backward_buf(&ctx->bb, &ctx->cfg, max_T);
    }

    pt_alloc_state(&ctx->state, &ctx->cfg);
}

/*
 * FLOW: Release the host working-buffer groups created during setup. Training
 * buffers exist only when max_T > 0; inference state always exists. free()
 * returns heap storage to the host allocator for reuse—it does not save or
 * erase the model file.
 *
 * CAUTION: ctx only points into the caller's weight_data, so pt_free() does not
 * release that separate buffer. After this call the freed fields must not be
 * used, and the same context must not be freed a second time without new setup.
 */
void pt_free(pt_context_t *ctx) {
    if (ctx->max_T > 0) {
        pt_free_activations(&ctx->acts);
        pt_free_grads(&ctx->grads);
        pt_free_backward_buf(&ctx->bb);
    }
    pt_free_state(&ctx->state);
    /*
     * CAUTION: This implementation frees any attached host trace, whether it
     * was allocated by pt_enable_trace() or supplied by the caller. A supplied
     * host trace must therefore be heap storage that may legally be freed here.
     * Treat its ownership as transferred to ctx: the caller must not separately
     * free it, and a local variable declared inside a function would not be safe.
     */
    if (ctx->trace) {
        free(ctx->trace);
        ctx->trace = NULL;
    }
}

#endif

/* ── inference helpers ──────────────────────────────────────── */

/*
 * FLOW: Process token at the current sequence position, advance the position,
 * and return a deterministic next-token choice. pt_forward() runs the
 * transformer using the selected ctx->matvec routine and fills state.logits
 * with one raw score for every vocabulary token.
 *
 * LLM NOTE: A logit is a raw model score, not yet a probability. argmax()
 * returns the ID whose score is largest. This greedy helper therefore bypasses
 * the sampler in pt_text.c, which can apply temperature, top-p, and randomness.
 * The returned ID is not processed automatically; a caller may feed it into a
 * later call as the next input token.
 *
 * CAUTION: No bounds check occurs here. Before calling, the caller must ensure
 * ctx->pos is between 0 and ctx->cfg.seq_len - 1, inclusive.
 */
int pt_forward_step(pt_context_t *ctx, int token) {
    pt_forward(&ctx->cfg, &ctx->w, &ctx->state, token, ctx->pos, ctx->matvec);
    ctx->pos++;
    return argmax(ctx->state.logits, ctx->cfg.vocab_size);
}

/*
 * FLOW: Start a logically new token sequence. The KV cache contains attention
 * information from earlier positions; clearing it prevents the next sequence
 * from inheriting that history, and resetting pos makes its first position 0.
 *
 * LLM NOTE: An attention head is one parallel attention calculation. head_dim
 * is the number of values in one head; kv_dim is the total stored key or value
 * width across all KV heads. Each cache has:
 *
 *   n_layers * seq_len * kv_dim floating-point values
 *
 * C NOTE: sizeof(float) converts the element count to the byte count required
 * by memset(). The key and value caches are separate arrays of that same size.
 * This helper assumes inference state was allocated successfully.
 */
void pt_reset_kv(pt_context_t *ctx) {
    int head_dim = ctx->cfg.dim / ctx->cfg.n_heads;
    int kv_dim   = ctx->cfg.n_kv_heads * head_dim;
    unsigned kv_bytes = ctx->cfg.n_layers * ctx->cfg.seq_len * kv_dim * sizeof(float);
    memset(ctx->state.key_cache,   0, kv_bytes);
    memset(ctx->state.value_cache, 0, kv_bytes);
    ctx->pos = 0;
}

/*
 * FLOW: Print the same model summary through the Pi serial console or a host
 * terminal. c is a shorter pointer to ctx->cfg, not a copied configuration;
 * const says this function reads through it without changing the fields.
 *
 * LLM NOTE: The compact `Kv` in the first line means roughly "thousand
 * vocabulary entries" because integer division drops the remainder from
 * vocab_size / 1000.
 * It is not the key/value cache; the detailed line separately shows KV heads.
 */
void pt_print_config(const pt_context_t *ctx) {
    const pt_config_t *c = &ctx->cfg;
#ifdef __RPI__
    printk("\npitorch (%dd %dL %dKv) | %d QPUs\n",
           c->dim, c->n_layers, c->vocab_size / 1000, ctx->num_qpus);
    printk("  dim=%d hidden=%d L=%d H=%d kv=%d V=%d seq=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads,
           c->n_kv_heads, c->vocab_size, c->seq_len);
    if (ctx->max_T > 0)
        printk("  training: max_T=%d\n", ctx->max_T);
#else
    printf("pitorch (%dd %dL %dKv) | CPU\n",
           c->dim, c->n_layers, c->vocab_size / 1000);
    printf("  dim=%d hidden=%d L=%d H=%d kv=%d V=%d seq=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads,
           c->n_kv_heads, c->vocab_size, c->seq_len);
    if (ctx->max_T > 0)
        printf("  training: max_T=%d\n", ctx->max_T);
#endif
}

/* ── host utilities ─────────────────────────────────────────── */

#ifndef __RPI__
/*
 * FLOW: Load an entire binary file into one host-memory buffer: open it, seek
 * to the end to measure it, rewind, allocate the measured number of bytes,
 * read them, and close the file. Model setup can then point into that buffer.
 *
 * C NOTE: FILE * is the C library's open-file handle, and "rb" means read raw
 * binary bytes. The returned void * is a generic starting address. out_size is
 * optional: when it is not NULL, `*out_size = sz` writes the size into the
 * caller's variable.
 *
 * CAUTION: The returned bytes are not a C string, so no '\0' is appended. The
 * caller owns this heap buffer and must eventually call free() separately;
 * pt_free() releases context working state, not this model buffer. This compact
 * project helper checks open failure but assumes measuring, allocation, and the
 * full read succeed, so it is intended for trusted project files.
 */
void *pt_read_file(const char *path, long *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}
#endif

/* ── profiling ──────────────────────────────────────────────── */

/*
 * FLOW: Attach optional storage for timing records and initialize it. A
 * non-NULL preallocated pointer supplies storage chosen by the caller. On a
 * host, passing NULL asks malloc() for storage; on the Pi, no host allocator is
 * available here, so the caller must supply it. If ctx->trace remains NULL,
 * tracing simply stays disabled while model calculations continue normally.
 *
 * C NOTE: pt_trace_init() clears old events and enables recording. Calling this
 * repeatedly can replace the old storage address, so set up tracing once or
 * manage the previous storage first. See pt_free() above for the host ownership
 * caution.
 */
void pt_enable_trace(pt_context_t *ctx, pt_trace_t *preallocated) {
    if (preallocated) {
        ctx->trace = preallocated;
    } else {
#ifndef __RPI__
        ctx->trace = (pt_trace_t *)malloc(sizeof(pt_trace_t));
#endif
    }
    if (ctx->trace) pt_trace_init(ctx->trace);
}

/*
 * FLOW: Stop recording new timing events without deleting events already
 * collected or releasing their storage.
 */
void pt_disable_trace(pt_context_t *ctx) {
    if (ctx->trace) ctx->trace->enabled = 0;
}

/* ── convenience training step ──────────────────────────────── */

/*
 * FLOW: Perform one complete learning step using the buffers prepared during
 * initialization:
 *
 *   1. Clear gradients left from the previous step.
 *   2. Run T tokens forward, save intermediate activations, and measure loss.
 *   3. Run backward through those calculations to find weight gradients.
 *   4. Apply one SGD update to the in-memory weights.
 *   5. Return the loss measured before that update.
 *
 * LLM NOTE: Loss is a number measuring prediction error. A gradient describes
 * how a small weight change would affect that loss. Backpropagation computes
 * gradients by following the saved calculations in reverse; it does not mean
 * generating text backward. SGD (stochastic gradient descent) then applies:
 *
 *   weight = weight - learning_rate * gradient
 *
 * Gradients are cleared first because backward calculations intentionally add
 * many contributions into reusable arrays. The modified weights remain only in
 * the loaded RAM buffer until separate code saves a checkpoint.
 *
 * C NOTE: tr is a shorter copy of the trace address and may be NULL; trace
 * begin/end calls safely do nothing in that case. Their nested labels measure
 * stages without changing the model mathematics.
 *
 * CAUTION: This wrapper assumes valid training setup and inputs: max_T > 0,
 * 2 <= T <= ctx->max_T, and token IDs inside the vocabulary. It does not check
 * those requirements itself.
 */
float pt_train_step(pt_context_t *ctx, const int *tokens, int T, float lr) {
    pt_trace_t *tr = ctx->trace;

    pt_trace_begin(tr, "train_step", "train", -1);

    pt_trace_begin(tr, "zero_grads", "train", -1);
    pt_zero_grads(&ctx->grads);
    pt_trace_end(tr);

    pt_trace_begin(tr, "forward", "fwd", -1);
    float loss = pt_forward_train(&ctx->cfg, &ctx->w, &ctx->acts,
                                  tokens, T, ctx->matvec, tr);
    pt_trace_end(tr);

    pt_trace_begin(tr, "backward", "bwd", -1);
    pt_backward(&ctx->cfg, &ctx->w, &ctx->grads, &ctx->acts,
                &ctx->bb, tokens, T, ctx->matvec, tr);
    pt_trace_end(tr);

    pt_trace_begin(tr, "sgd", "train", -1);
    pt_sgd_update(&ctx->w, &ctx->grads, lr, &ctx->cfg);
    pt_trace_end(tr);

    pt_trace_end(tr);
    return loss;
}
