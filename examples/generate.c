/*
 * generate.c — Interactive text generation on a single Pi Zero.
 *
 * Type a prompt over UART, get a story back. Tokens stream in real-time
 * at ~12 tok/s (stories15M with D-cache enabled).
 * UART is a simple serial connection to another computer. A token is a word
 * or word-piece, and inference means running an already-trained model to
 * produce output. The D-cache keeps recently used data near the CPU.
 *
 * ── How to run ──────────────────────────────────────────────────────
 *
 *   cd examples && ./run.sh generate           # deploy to Pi 0 (default)
 *   cd examples && PI_DEVICE=2 ./run.sh generate   # deploy to Pi 2
 *
 * ── SD card ─────────────────────────────────────────────────────────
 *
 *   initramfs weights/stories15M_full.bin 0x2000000
 *   (combined file: model weights + tokenizer)
 *
 * Reading guide: first read the constants that describe the memory, GPU, and
 * generation choices. Then read read_prompt() as the UART input boundary.
 * Finally, follow notmain() through one-time setup and into its prompt loop.
 * The lower-level functions named pt_* are signposts into PiTorch; this file
 * focuses on how they are assembled into an interactive application.
 *
 * ════════════════════════════════════════════════════════════════════
 */

/*
 * C NOTE: A header supplies declarations so this file can call code defined
 * elsewhere. rpi.h and mmu.h expose bare-metal (no operating system)
 * Raspberry Pi services; pt.h exposes the model context; pt_text.h exposes
 * tokenization and generation.
 */
#include "rpi.h"
#include "mmu.h"
#include "pt.h"
#include "pt_text.h"

/*
 * C NOTE: #define creates a named value that the C preprocessor substitutes
 * before compilation; these names make the choices below easy to find.
 *
 * HARDWARE NOTE: The boot configuration places the combined model file at
 * byte address 0x02000000 (32 MiB). Casting that integer to void * turns it
 * into an untyped pointer: an address that the loader functions interpret.
 * A QPU is one of the VideoCore GPU's small parallel processors; all 12 are
 * made available for matrix-vector calculations, a basic operation that mixes
 * model weights with current values. The arena is a 1 MiB (1,048,576-byte)
 * temporary GPU workspace, not storage for the model itself.
 *
 * LLM NOTE: MAX_TOKENS caps the total prompt-plus-generation steps in
 * pt_generate().
 * Temperature controls randomness. At 0.0 the sampler always chooses the
 * highest-scoring token ("greedy" decoding), so TOPP is not used in this
 * configuration; top-p would otherwise restrict sampling to a high-probability
 * group of candidate tokens.
 */
#define WEIGHT_ADDR  ((void *)0x02000000)
#define NUM_QPUS     12
#define ARENA_SIZE   (1 * 1024 * 1024)
#define MAX_TOKENS   256
#define TEMPERATURE  0.0f
#define TOPP         0.9f

/*
 * FLOW: Read one prompt from UART (the serial link to the user's terminal).
 * uart_get8() waits for one byte, so the program pauses here while the user
 * types. The char * parameter points to storage owned by the caller; max_len
 * prevents this helper from writing beyond that storage. static keeps this
 * helper private to this source file. The loop reserves one byte for '\0'.
 *
 * CAUTION: If the user reaches that limit, the function returns without
 * consuming later input bytes; those bytes will be read by the next prompt.
 */
static int read_prompt(char *buf, int max_len) {
    int i = 0;
    while (i < max_len - 1) {
        int c = uart_get8();
        /*
         * Terminals may send either line-feed ('\n') or carriage-return ('\r').
         */
        if (c == '\n' || c == '\r') break;
        /*
         * C NOTE: Delete (0x7f) and backspace ('\b') are different byte values.
         * Moving back, printing a space, and moving back again visually erases
         * the last character on a simple serial terminal as well as removing
         * it from the buffer by decrementing i.
         */
        if (c == 0x7f || c == '\b') {
            if (i > 0) { i--; uart_put8('\b'); uart_put8(' '); uart_put8('\b'); }
            continue;
        }
        /* Echo the byte so locally typed characters appear on screen. */
        buf[i++] = (char)c;
        uart_put8((uint8_t)c);
    }
    /*
     * C NOTE: '\0' terminates a C string; it is data, not a printed character.
     */
    buf[i] = '\0';
    return i;
}

/*
 * FLOW: On this bare-metal program there is no operating system to call the
 * usual main(). The small startup runtime initializes essentials, then enters
 * notmain(). This function performs setup once and serves prompts forever.
 */
void notmain(void) {
    /*
     * HARDWARE NOTE: The MMU (Memory Management Unit) establishes how address
     * ranges behave and enables CPU caches for normal RAM. Caches keep recently
     * used data close to the CPU and are important for inference speed.
     * Hardware peripheral addresses remain uncached so device reads and writes
     * stay exact.
     */
    mmu_init_and_enable();

    /*
     * FLOW: Load the tokenizer before setting up the model. Model setup uses
     * some of the same memory, so loading the tokenizer later could be too late.
     * pt_load_tokenizer() copies it to a safe place now.
     *
     * LLM NOTE: The tokenizer changes text into number IDs for the model, then
     * changes generated IDs back into text. vocab_size is the number of IDs the
     * model knows.
     *
     * C NOTE: tmp_cfg and tok are structs, which are variables that hold several
     * related values. & means "the address of this variable"; it lets a function
     * fill in that variable. Read the two loader functions later for details.
     */
    pt_config_t tmp_cfg;
    pt_load_config(&tmp_cfg, WEIGHT_ADDR);

    pt_tokenizer_t tok;
    pt_load_tokenizer(&tok, WEIGHT_ADDR, tmp_cfg.vocab_size);

    /*
     * FLOW: Build the long-lived model context and prepare GPU acceleration.
     * pt_context_t groups the parsed configuration, pointers to weights, the
     * inference state (including cached data from earlier token positions), and
     * the selected matrix-vector routine. Passing max_T as 0 requests
     * inference-only state; training needs additional buffers that this example
     * does not allocate.
     * Follow pt_pi_init() later for the detailed memory map and GPU setup.
     */
    pt_context_t ctx;
    pt_pi_init(&ctx, WEIGHT_ADDR, NUM_QPUS, 0, ARENA_SIZE);

    /*
     * LLM NOTE: The sampler turns the model's score for every vocabulary token
     * into the next token choice. timer_get_usec() supplies a seed (the starting
     * value for a random-number sequence); with TEMPERATURE set to zero the same
     * scores always produce the same choice, but this setup also supports
     * randomness.
     */
    pt_sampler_t sampler;
    pt_sampler_init(&sampler, ctx.cfg.vocab_size, TEMPERATURE, TOPP,
                    timer_get_usec());

    /*
     * FLOW: Reuse the initialized model for any number of prompts. An endless
     * loop is normal here because there is no operating-system shell to return
     * to. printk() writes through the Pi's serial console.
     */
    char prompt[256];
    while (1) {
        printk("> ");
        /* C NOTE: sizeof(prompt) gives this array's byte capacity: 256 here. */
        int len = read_prompt(prompt, sizeof(prompt));
        /* continue starts the next loop iteration without running generation. */
        if (len == 0) { printk("\n"); continue; }

        printk("\n...");
        /*
         * LLM NOTE: pt_generate() tokenizes the prompt, runs the prompt tokens
         * through the transformer—the neural-network design used here—during
         * "prefill", then predicts and prints new tokens one at a time during
         * "decode". ctx.matvec is a function pointer: it stores the address of
         * the GPU-backed matrix-vector operation used by the model's math. Follow
         * pt_generate() later for those stages and its stopping rules.
         */
        pt_generate(&ctx.cfg, &ctx.w, &ctx.state, &tok, &sampler,
                    prompt, MAX_TOKENS, ctx.matvec);
        printk("\n\n");
    }
}
