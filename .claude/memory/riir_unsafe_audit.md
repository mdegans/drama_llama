# RIIR moeflux: unsafe audit (2026-04-28, post-Phase-6)

Audit pass over `crates/moeflux/src/riir/` after Phase 6 cutover.
Goal: catalog the unsafe surface, note where caller-discipline holds
vs. where compile-time enforcement could be retrofitted.

## Inventory by category

### FFI / libm (unavoidable)

- `rope.rs:102-103, 106-107, 117-122` — 6× `unsafe { cosf/sinf/powf }`
  via `extern "C"` libm bindings.
- `sdpa.rs:146, 167` — 2× `unsafe { expf }`.
- `weight_file.rs:101` — `unsafe { Mmap::map(&file) }` via memmap2.

All pure functions / well-defined library contracts. No further
action.

### Shared-storage Metal buffer accessors

`metal::Buffer::contents()` returns a raw pointer to GPU-shared
memory. Reading or writing through it without ensuring the GPU is
idle is UB. Every site relies on caller-side discipline:

- `metal.rs:264-282` — `MtlBuffer::to_vec` / `as_mut_slice`. Has
  `# Safety` headings + inline SAFETY comments. Function-level
  contract clearly stated.
- `linear_attn_forward.rs:1204` — `read_buffer_to_vec`. Tightened
  2026-04-28 with full `# Safety` heading.
- `linear_attn_forward.rs:306` — `zero_f32_buffer`. Tightened
  2026-04-28: only reachable from `memory_clear`, which drains the
  deferred ring at the top.
- `state_snapshot.rs:475, 485` — `read_buffer_bytes` /
  `write_buffer_bytes`. Tightened 2026-04-28: only reachable from
  `state_save` / `state_load`, both of which drain.
- `gpu_lm_head.rs:148-181` — host-to-GPU + GPU-to-host memcpys
  bracketed by `wait_until_completed` on the LM head dispatch.
- `mod.rs:748-895` — host↔GPU staging in `step_internal` /
  `layer_forward_dump`. Same pattern.

**All discipline-correct.** No bugs found; no immediate action.

### Manual `Send` impls / pointer storage

- `mtl_weight_buf.rs:66` — `unsafe impl Send for MtlWeightBuf`.
  `MtlWeightBuf` wraps an mmap'd Metal buffer + `NonNull<u8>` base
  pointer. Read-only weights; `Metal::Buffer` is documented `Send`;
  `WeightFile` outlives the buffer. Sound.
- `gpu_lm_head.rs:71` — `unsafe impl Send for GpuLmHead`. Single-
  owner (a field on `RsCtx`). Sound.
- `prefetch.rs` — `DataPrefetchPtr` / `ExpertFilesPtr` store
  `usize`-encoded raw pointers to bypass `!Send` propagation. The
  unsafe accessors (`as_mut_slice`, `as_ref`) are `unsafe fn` with
  drain-before-touch contracts. Sound.

### Slice constructions over typed pointers

- `state_snapshot.rs:211, 217, 384, 392` — serialize/deserialize
  Vec<f32> as bytes. Lengths derived from `.len()`. Sound.

### Build script

- `moeflux-sys/build.rs` — no unsafe.
- `moeflux/src/riir/metal.rs:SHADER_SOURCE` — `include_str!` is
  safe.

## "Could probably be safer" candidates (Phase 7)

The discipline at every Metal-buffer site is "caller must ensure no
in-flight GPU work writes to this buffer." This is enforced by code
review + structural ordering, not by the type system. A `GuardToken`
pattern could lift this to compile-time enforcement:

```rust
// Phase 7 sketch — NOT currently implemented.
pub struct GpuIdleGuard<'a, T> {
    buf: &'a MtlBuffer<T>,
    _idle: PhantomData<&'a CommandQueue>,
}
impl<'a, T: Copy> GpuIdleGuard<'a, T> {
    pub fn new(buf: &'a MtlBuffer<T>, q: &'a CommandQueue) -> Self { ... }
    pub fn read(&self) -> &[T] { ... } // safe — guard proves idle
    pub fn write(&mut self) -> &mut [T] { ... }
}
```

**Cost**: every call site that currently passes `&MtlBuffer<f32>` →
host slice would need to acquire a guard, which means threading a
`&CommandQueue` deeper than today. Most call sites already have it
nearby (it's owned by the same `Ctx`), but the borrow-checker
arithmetic gets fiddly when the same `Ctx` is mid-mutation.

**Verdict**: real engineering effort, real safety win, defer to
Phase 7. The current code is sound; the pattern is well-understood;
no bugs have ever been traced to a missed wait. Pick this up if
either (a) a future bug actually traces to a missed wait, or (b) a
new contributor mis-handles the discipline.

## What's NOT here

- `moeflux-sys/` itself — bindgen-generated `extern "C"` only. The
  C-side oracle is test-only behind `diff-oracle`; not part of the
  production unsafe surface.
- The Phase-7 typed `memory_seq_rm` `Result<(), CannotTruncateLinear>`
  is a separate concern (typed-error API, not unsafe-code surface).
