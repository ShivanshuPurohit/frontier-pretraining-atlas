# Chapter 19 — Frameworks: Megatron, TorchTitan, MaxText, and the Field

A 10T-total/200B-active MoE on 100,000 GB200 GPUs will be built on top of somebody else's parallelism engine, patched in dozens of places, and the choice of *whose* engine determines which bugs the team inherits. This chapter reads the field at the level of dispatcher classes, open GitHub issues, and merged pull requests rather than marketing claims, because at this scale the deciding factor is rarely "does it support MoE" (everything now does) and almost always "what breaks when EP, communication overlap, uneven pipeline layouts, FSDP, and CPU offload are all turned on simultaneously."

## Megatron-Core: the reference stack, read at the commit level

Megatron-Core is the library a 10T/200B build will most likely fork, and as of July 2026 it has a primary source detailed enough to make that decision on evidence: NVIDIA's ["Scalable Training of Mixture-of-Experts Models with Megatron Core"](https://arxiv.org/abs/2603.07685) (Mar 2026), benchmarked against Megatron-Core v0.16 and cited by section throughout this chapter. It reports **1,233 TFLOPS/GPU on GB300 and 1,048 TFLOPS/GPU on GB200 for DeepSeek-V3-685B**, and **974/919 TFLOPS/GPU for Qwen3-235B**, under force-balanced routing. The architecture matters more than the headline numbers, and it's organized around a sequence of "walls" — parallelism, communication, memory, compute-efficiency — each solved by a shipped mechanism, not a roadmap item.

**Parallel Folding is the load-bearing idea.** Attention wants high TP/CP; MoE wants high EP and low expert-TP, since sharding an already-small expert matrix with TP just fragments it further. Traditional frameworks force `EP ≤ DP` by nesting EP inside the data-parallel group. **MoE Parallel Folding** (arXiv:2504.14960, Apr 2025 — shipped code, not a proposal) decouples this: attention gets its own topology (`TP × CP × DP × PP`), MoE layers get a separate one (`ETP × EP × EDP × PP`), sharing only PP for gradient-flow correctness (§3.2.4). Worked example: attention at TP=4/CP=2/DP=8/PP=4 (256 GPUs) traditionally caps EP at DP=8; under folding EP reaches 64 — 8x higher at the identical GPU count (§3.3.3). A sharper example: CP=8 plus EP=8 needs only 8 GPUs under folding versus 64 under the traditional constraint, since CP and EP can share the same GPU group. This is process-group management, and it's what makes EP64 on a 256-GPU DeepSeek-V3 config arithmetically possible. The companion paper's own earlier numbers — 49.3% MFU on Mixtral-8x22B, 39.0% on Qwen2-57B-A14B, validated to 1,024 GPUs — are the honest floor before dispatchers, CUDA-Graph coverage, and FP8 are layered on.

**Three dispatcher backends, one class hierarchy.** `token_dispatcher.py` (1,859 lines, checked July 2026) confirms the paper's description at the class level: a unified `MoEFlexTokenDispatcher` delegates to `_HybridEPManager`, `_DeepepManager`, or `_NCCLEPManager`. **DeepEP** (Hopper, cross-node, MIT-licensed) and **HybridEP** (NVIDIA's GB200/GB300 NVL72-specific kernel) are the two production backends beyond raw NCCL all-to-all, both wired through `fused_a2a.py`. The surprising code-level fact: `fused_a2a.py` imports `from deep_ep import HybridEPBuffer` — HybridEP, despite being "developed by NVIDIA" in the paper's prose, is upstreamed into DeepSeek's own DeepEP repository on a `hybrid-ep` branch rather than shipped as a separate NVIDIA package — a narrow but concrete NVIDIA/DeepSeek infra collaboration signal absent from either org's public messaging.

The communication numbers this zoo solves for are large: DeepSeek-V3's 58 MoE layers × 2 ops/layer gives 116 dispatch/combine operations per forward pass, doubling on backward (§2.2.1, §4.2.1). Table 7 is the sharpest available statement of why NVL72 topology matters for EP: at EP=64, GB200 HybridEP dispatch latency is 675µs versus 930µs for plain all-to-all, while the same comparison on H100 (crossing nodes) is 4,626µs versus 9,164µs — GB200 HybridEP stays flat across EP 8→64 because EP64 fits inside the NVL72 domain, while H100 all-to-all blows up 4-8x crossing node boundaries. Unoptimized, EP all-to-all consumes 20-60% of training time depending on whether EP stays intra-NVLink (~20% GB200) or crosses nodes (40-60% H100) (§4.2). Overlap is handled by a **1F1B all-to-all overlap scheme** — "conceptually a DualPipe-like bidirectional schedule built on top of standard 1F1B" — in `combined_1f1b.py`, via `--overlap-moe-expert-parallel-comm` (§4.2.3). Tellingly, a feature request for native DualPipeV support (issue #1524, closed Dec 2025) was resolved by pointing to this scheme rather than porting DeepSeek's DualPipeV verbatim, and an unresolved sub-thread asks whether the VPP-based approach needs more inter-node communication than true DualPipe at equal bubble rate — an open, unanswered doubt inside Megatron-Core's own community.

<figure class="vz">
<div class="scroll"><svg style="min-width:620px" viewBox="0 0 653.0 318" role="img" aria-label="Pipeline schedules: GPipe fill-and-drain versus 1F1B, four stages, eight microbatches">
<g style="--step:.12s">
<text class="t-ttl" x="0" y="20">Fill-and-drain (GPipe): all forwards, then all backwards</text>
<text class="t-mut" x="0" y="44">stage 0</text>
<rect class="f-hair" x="92" y="30" width="561.0" height="20" rx="2"/>
<text class="t-mut" x="0" y="67">stage 1</text>
<rect class="f-hair" x="92" y="53" width="561.0" height="20" rx="2"/>
<text class="t-mut" x="0" y="90">stage 2</text>
<rect class="f-hair" x="92" y="76" width="561.0" height="20" rx="2"/>
<text class="t-mut" x="0" y="113">stage 3</text>
<rect class="f-hair" x="92" y="99" width="561.0" height="20" rx="2"/>
<rect class="build f-acc-22" style="--t:0" x="92" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:0" x="100" y="44" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:1" x="109" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:1" x="117" y="44" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:1" x="109" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:1" x="117" y="67" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:2" x="126" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:2" x="134" y="44" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:2" x="126" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:2" x="134" y="67" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:2" x="126" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:2" x="134" y="90" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="44" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="67" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="90" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="113" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:4" x="160" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:4" x="168" y="44" text-anchor="middle">4</text>
<rect class="build f-acc-22" style="--t:4" x="160" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:4" x="168" y="67" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:4" x="160" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:4" x="168" y="90" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:4" x="160" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:4" x="168" y="113" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:5" x="177" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:5" x="185" y="44" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:5" x="177" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:5" x="185" y="67" text-anchor="middle">4</text>
<rect class="build f-acc-22" style="--t:5" x="177" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:5" x="185" y="90" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:5" x="177" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:5" x="185" y="113" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:6" x="194" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:6" x="202" y="44" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:6" x="194" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:6" x="202" y="67" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:6" x="194" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:6" x="202" y="90" text-anchor="middle">4</text>
<rect class="build f-acc-22" style="--t:6" x="194" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:6" x="202" y="113" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:7" x="211" y="30" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:7" x="219" y="44" text-anchor="middle">7</text>
<rect class="build f-acc-22" style="--t:7" x="211" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:7" x="219" y="67" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:7" x="211" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:7" x="219" y="90" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:7" x="211" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:7" x="219" y="113" text-anchor="middle">4</text>
<rect class="build f-acc-22" style="--t:8" x="228" y="53" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:8" x="236" y="67" text-anchor="middle">7</text>
<rect class="build f-acc-22" style="--t:8" x="228" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:8" x="236" y="90" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:8" x="228" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:8" x="236" y="113" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:9" x="245" y="76" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:9" x="253" y="90" text-anchor="middle">7</text>
<rect class="build f-acc-22" style="--t:9" x="245" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:9" x="253" y="113" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:10" x="262" y="99" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:10" x="270" y="113" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:11" x="279" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:11" x="296" y="113" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:13" x="313" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:13" x="330" y="113" text-anchor="middle">1</text>
<rect class="build f-loud-25" style="--t:13" x="313" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:13" x="330" y="90" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:15" x="347" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:15" x="364" y="113" text-anchor="middle">2</text>
<rect class="build f-loud-25" style="--t:15" x="347" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:15" x="364" y="90" text-anchor="middle">1</text>
<rect class="build f-loud-25" style="--t:15" x="347" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:15" x="364" y="67" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:17" x="381" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:17" x="398" y="113" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:17" x="381" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:17" x="398" y="90" text-anchor="middle">2</text>
<rect class="build f-loud-25" style="--t:17" x="381" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:17" x="398" y="67" text-anchor="middle">1</text>
<rect class="build f-loud-25" style="--t:17" x="381" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:17" x="398" y="44" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:19" x="415" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="432" y="113" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:19" x="415" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="432" y="90" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:19" x="415" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="432" y="67" text-anchor="middle">2</text>
<rect class="build f-loud-25" style="--t:19" x="415" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="432" y="44" text-anchor="middle">1</text>
<rect class="build f-loud-25" style="--t:21" x="449" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="466" y="113" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:21" x="449" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="466" y="90" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:21" x="449" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="466" y="67" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:21" x="449" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="466" y="44" text-anchor="middle">2</text>
<rect class="build f-loud-25" style="--t:23" x="483" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:23" x="500" y="113" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:23" x="483" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:23" x="500" y="90" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:23" x="483" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:23" x="500" y="67" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:23" x="483" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:23" x="500" y="44" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:25" x="517" y="99" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:25" x="534" y="113" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:25" x="517" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:25" x="534" y="90" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:25" x="517" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:25" x="534" y="67" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:25" x="517" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:25" x="534" y="44" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:27" x="551" y="76" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:27" x="568" y="90" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:27" x="551" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:27" x="568" y="67" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:27" x="551" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:27" x="568" y="44" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:29" x="585" y="53" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:29" x="602" y="67" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:29" x="585" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:29" x="602" y="44" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:31" x="619" y="30" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:31" x="636" y="44" text-anchor="middle">7</text>
<text class="t-mut" x="92" y="136">peak activations in flight per stage: m = 8 microbatches</text>
<text class="t-ttl" x="0" y="158">1F1B: same bubble, but at most p microbatches in flight</text>
<text class="t-mut" x="0" y="182">stage 0</text>
<rect class="f-hair" x="92" y="168" width="561.0" height="20" rx="2"/>
<text class="t-mut" x="0" y="205">stage 1</text>
<rect class="f-hair" x="92" y="191" width="561.0" height="20" rx="2"/>
<text class="t-mut" x="0" y="228">stage 2</text>
<rect class="f-hair" x="92" y="214" width="561.0" height="20" rx="2"/>
<text class="t-mut" x="0" y="251">stage 3</text>
<rect class="f-hair" x="92" y="237" width="561.0" height="20" rx="2"/>
<rect class="build f-acc-22" style="--t:0" x="92" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:0" x="100" y="182" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:1" x="109" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:1" x="117" y="182" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:1" x="109" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:1" x="117" y="205" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:2" x="126" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:2" x="134" y="182" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:2" x="126" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:2" x="134" y="205" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:2" x="126" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:2" x="134" y="228" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="182" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="205" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="228" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:3" x="143" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:3" x="151" y="251" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:4" x="160" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:4" x="176" y="251" text-anchor="middle">0</text>
<rect class="build f-acc-22" style="--t:6" x="194" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:6" x="202" y="251" text-anchor="middle">1</text>
<rect class="build f-loud-25" style="--t:6" x="194" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:6" x="210" y="228" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:7" x="211" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:7" x="228" y="251" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:8" x="228" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:8" x="236" y="228" text-anchor="middle">2</text>
<rect class="build f-loud-25" style="--t:8" x="228" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:8" x="244" y="205" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:9" x="245" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:9" x="262" y="228" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:9" x="245" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:9" x="253" y="251" text-anchor="middle">2</text>
<rect class="build f-loud-25" style="--t:10" x="262" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:10" x="278" y="251" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:10" x="262" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:10" x="270" y="205" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:10" x="262" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:10" x="278" y="182" text-anchor="middle">0</text>
<rect class="build f-loud-25" style="--t:11" x="279" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:11" x="296" y="205" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:11" x="279" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:11" x="287" y="228" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:12" x="296" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:12" x="312" y="228" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:12" x="296" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:12" x="304" y="251" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:12" x="296" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:12" x="304" y="182" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:13" x="313" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:13" x="330" y="251" text-anchor="middle">3</text>
<rect class="build f-loud-25" style="--t:13" x="313" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:13" x="330" y="182" text-anchor="middle">1</text>
<rect class="build f-acc-22" style="--t:13" x="313" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:13" x="321" y="205" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:14" x="330" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:14" x="346" y="205" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:14" x="330" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:14" x="338" y="228" text-anchor="middle">4</text>
<rect class="build f-acc-22" style="--t:15" x="347" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:15" x="355" y="182" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:15" x="347" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:15" x="364" y="228" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:15" x="347" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:15" x="355" y="251" text-anchor="middle">4</text>
<rect class="build f-acc-22" style="--t:16" x="364" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:16" x="372" y="205" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:16" x="364" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:16" x="380" y="251" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:16" x="364" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:16" x="380" y="182" text-anchor="middle">2</text>
<rect class="build f-acc-22" style="--t:17" x="381" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:17" x="389" y="228" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:17" x="381" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:17" x="398" y="205" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:18" x="398" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:18" x="406" y="251" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:18" x="398" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:18" x="406" y="182" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:18" x="398" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:18" x="414" y="228" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:19" x="415" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="432" y="251" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:19" x="415" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="423" y="205" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:19" x="415" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:19" x="432" y="182" text-anchor="middle">3</text>
<rect class="build f-acc-22" style="--t:20" x="432" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:20" x="440" y="228" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:20" x="432" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:20" x="448" y="205" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:21" x="449" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="466" y="228" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:21" x="449" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="457" y="251" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:21" x="449" y="168" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:21" x="457" y="182" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:22" x="466" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:22" x="482" y="251" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:22" x="466" y="191" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:22" x="474" y="205" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:22" x="466" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:22" x="482" y="182" text-anchor="middle">4</text>
<rect class="build f-loud-25" style="--t:23" x="483" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:23" x="500" y="205" text-anchor="middle">5</text>
<rect class="build f-acc-22" style="--t:23" x="483" y="214" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:23" x="491" y="228" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:24" x="500" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:24" x="516" y="228" text-anchor="middle">6</text>
<rect class="build f-acc-22" style="--t:24" x="500" y="237" width="16" height="20" rx="2"/>
<text class="t-s9 build" style="--t:24" x="508" y="251" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:25" x="517" y="237" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:25" x="534" y="251" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:25" x="517" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:25" x="534" y="182" text-anchor="middle">5</text>
<rect class="build f-loud-25" style="--t:26" x="534" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:26" x="550" y="205" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:27" x="551" y="214" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:27" x="568" y="228" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:28" x="568" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:28" x="584" y="182" text-anchor="middle">6</text>
<rect class="build f-loud-25" style="--t:29" x="585" y="191" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:29" x="602" y="205" text-anchor="middle">7</text>
<rect class="build f-loud-25" style="--t:31" x="619" y="168" width="33" height="20" rx="2"/>
<text class="t-s9 build" style="--t:31" x="636" y="182" text-anchor="middle">7</text>
<text class="t-mut" x="92" y="274">peak activations in flight per stage: p = 4 — the schedule&#8217;s real win</text>
<rect class="f-acc-22" x="92" y="288" width="14" height="12" rx="2"/><text class="t-mut" x="112" y="298">forward (1 unit)</text>
<rect class="f-loud-25" x="222" y="288" width="14" height="12" rx="2"/><text class="t-mut" x="242" y="298">backward (2 units)</text>
<rect class="f-hair" x="382" y="288" width="14" height="12" rx="2"/><text class="t-mut" x="402" y="298">bubble (stage idle)</text>
</g>
</svg></div>
<p class="vz-cap">Four stages, eight microbatches, backward drawn at twice forward's cost — both schedules simulated exactly. The two pay the same warmup-and-drain bubble, (p−1)(F+B); 1F1B's win is memory, holding at most p microbatches of activations in flight against GPipe's m. What shrinks the bubble is interleaving: r = (p−1)/(v·m), which is why Ch18's Layout 2 (p=8, v=4) needs m≥18 microbatches to hold r under 10% — and the steady B–F cadence in the lower panel is the compute window Megatron-Core's 1F1B-based all-to-all overlap scheme (§4.2.3) hides EP dispatch behind.</p>
</figure>

**Dropless MoE and CUDA Graphs — the least-publicized section.** Dropless MoE has per-expert token counts known only on-device; host-initiated kernels need a device→host sync to read shapes, blocking CUDA Graph capture. Three mechanisms close this gap, none widely discussed elsewhere (§4.3.7). Device-initiated Grouped GEMM (cuBLASLt with device-array shapes, CUDA 13.1+; cuteDSL with SwiGLU/FP8 fused into the epilogue) plus a sync-free HybridEP dispatch mode. **ECHO** (Elastic Cloning for Hot Experts, PR #2368, merged Feb 2026): a bin-packing planner identifies overloaded experts per forward pass, clones their weights onto underutilized ranks via sync-free comm, redirects overflow tokens, and reduces clone gradients back to the home expert on backward — architecturally distinct from DeepSeek's EPLB (inference-time replica redistribution); ECHO is training-time and gradient-aware. **Paged Stashing** (PR #2690): replaces per-layer worst-case CUDA-Graph buffers (O(layers×worst_case), often an order of magnitude above actual usage) with one shared `tmp` buffer plus a paged buffer storing only actual per-layer token counts, cutting memory to O(worst_case+actual_total), overlapped on dedicated streams. Both PRs are merged into `dev`, but the 2026-Q2 roadmap (issue #4815) still lists "sync-free, full-iteration CUDA-Graph MoE training" as long-term — full integration is not yet default as of July 2026.

**Memory and compute-efficiency.** At PP4×VPP4×EP64 on 256 GPUs, BF16, DeepSeek-V3 measures 199.5 GB/GPU (36.4 weights+grads, 32.1 optimizer, 131.0 activations); the full optimization stack (FP8, fine-grained recompute, memory-efficient permutation, offloading) claims to bring this under 80 GB (Table 3, §4.1.1, §11). GEMMs account for ~70% of layer time in dense Llama-3-405B but under 50% in DeepSeek-V3 — MoE structurally halves GEMM's share of the timeline before communication is even counted (§1.2).

**Distributed optimizer, FSDP, checkpointing.** The distributed optimizer shards MoE optimizer state only among replicas of the same expert via EDP, avoiding cross-EP-group duplication. **Megatron-FSDP** is "the practical fully sharded data parallel path in Megatron Bridge today" per NVIDIA's docs, using a Dual DeviceMesh design for EP compatibility; "FSDP + A2A overlap" is still roadmapped, i.e. not yet mature. Distributed checkpointing uses a `ShardedTensor` abstraction enabling fully-parallel save and any-to-any reshard on load — save at TP=2/EP=4, load at TP=4/EP=8, no offline conversion — backed by Zarr or PyTorch Distributed (§7.4). Two features worth flagging: **Flexible Asymmetric VPP** (arbitrary per-stage layer type/count — DeepSeek-V3's PP=16/VPP=2 layout individually places dense, MoE, MTP, and loss layers; Table 10, §7.5); and **LatentMoE** (compresses routed-expert computation through a shared low-rank projection around dispatch, reducing both all-to-all volume and per-expert weight size — already production-deployed in Nemotron-3 Super/Ultra; §7.3). LatentMoE compresses the dispatch payload itself, a distinct axis from anything in the DeepSeek or MAI-1 disclosures, and easy to miss buried in a systems-paper appendix.

**Ecosystem restructuring.** As of July 2026: **Megatron-Core** (the engine, a library not a trainer), **Megatron-Bridge** (submodule dependency, HF↔Megatron checkpoint conversion, explicitly "a refactor of the previous NeMo training stack that adopts a PyTorch-native training loop"), and **NeMo AutoModel** (DTensor-based, bypasses Megatron-Core entirely). NeMo 2.0 is deprecated as of the 25.11 release. Current Megatron-Bridge release is 0.5.0 (June 22, 2026); NeMo RL runs Megatron-Core inside Ray workers and also backs veRL and Slime via the same interop layer. The net read: Megatron-Core is the engine to build on directly — it's what "Megatron-LM" colloquially means now — while Megatron-Bridge is the right layer only for HF interop; at 100k-GPU scale the team will be patching Megatron-Core directly regardless (open issue #4590, "Reduce downstream Megatron patching for RL use cases," is explicitly about how much even NVIDIA's own RL frameworks have to patch it).

**What actually breaks first, from open issues, not vibes.** The paper's own benchmarks top out at EP64/1,024 GPUs (H100) and 256 GPUs (GB200/GB300) — the live tracker surfaces a sharper ceiling signal:

1. **NCCL collective-order deadlocks in EP overlap with uneven pipeline layouts.** #5749 (open): a 128-rank deadlock after 92-852 steps, EP-disabled the only workaround (2.2x step-time cost). #1810 (open) root-causes it: a single shared CUDA event synchronizing both compute and comm streams lets different ranks enqueue the same two collectives in different orders, violating NCCL's same-order invariant — a currently open *correctness* bug (silent hang, not wrong numbers) in the flagship `--overlap-moe-expert-parallel-comm` feature, triggered exactly by the EP + comm-overlap + uneven-VPP combination a heterogeneous dense/MoE/MTP 10T model would use.
2. **Hardware protocol limits becoming training-time constraints.** #3999 (fixed): HybridEP's token-count computation used flattened microbatch size, pushing RDMA queue-pair depth past InfiniBand's 65,535 hardware limit at 8,192-seqlen×8-microbatch, aborting all ranks — fixed, but the class of bug will recur at 10T-scale batch sizes.
3. **Fault-tolerance/checkpoint-integrity gaps compound at scale.** #5281 (open): local replica-based checkpointing can silently restore a corrupt replica (deterministic NaN 3 iterations after load); with no validation and no automatic fallback to the global checkpoint, `ft_launcher` enters an infinite restart loop (44 identical ~90s cycles observed). At 100k-GPU scale, where local checkpointing is the only practical fast-restart mechanism, this is a real availability risk (see Ch21).
4. **Megatron-FSDP + optimizer CPU offload checkpoint hangs.** #4910 (open): save hangs during optimizer-state preprocessing specifically under `optimizer_cpu_offload=True` — a gap between two headline memory-saving features the paper itself recommends composing.
5. Two closed-but-instructive issues (#1707, checkpoint-validation OOM at scale; #1053, Zarr losing distributed-optimizer state) confirm checkpointing-at-scale is a recurring soft spot, not a one-off.

Raw throughput does not look like a near-term ceiling — the dispatch/compute/combine pipeline scales cleanly through EP64 and thousands of GPUs when the stack cooperates. The bottleneck at 100k-GPU/10T-parameter scale is the *intersection* of features (EP + comm-overlap + uneven-VPP + FSDP + CPU-offload + fault-tolerance, all active together), consistent with the paper's own thesis that "addressing walls in isolation leads to suboptimal solutions" (§4).

The Megatron-Bridge 26.06-container performance-summary page publishes exact TP/PP/CP/VP/EP configs across eight model families and four hardware generations; a representative slice:

| Model | Hardware | GPUs | Precision | Config | tok/s/GPU |
|---|---|---|---|---|---|
| DeepSeek-V3 | DGX-GB300 | 256 | MXFP8 | TP1/PP2/EP32/VP8 | 6,338–6,422 |
| DeepSeek-V3 | DGX-GB200 | 256 | MXFP8 | TP1/PP4/EP64/VP4 | 4,969 |
| Qwen3-235B-A22B | DGX-GB300 | 256 | MXFP8 | TP1/PP4/EP32/VP12 | 9,015 |
| Kimi-K2 (Muon) | DGX-GB300 | 256 | MXFP8 | TP1/PP4/EP64/VP4 | 5,372 |
| LLAMA3.1-405B (dense) | DGX-GB300 | 256 | NVFP4 | TP4/PP8/CP1/VP4 | 1,413 |

Source: the Megatron-Bridge 26.06 performance summary (July 2026).

## TorchTitan: the PyTorch-native challenger, honestly assessed

TorchTitan has grown far past its October 2024 paper (arXiv:2410.06511, Llama-3-only). The current tree (checked July 2026) natively supports **DeepSeek-V3** (real MLA — `kv_lora_rank=512`, `qk_nope_head_dim=128`, matching the public architecture — and a `_671b()` config registering the real 256 experts), gpt-oss, Qwen3/Qwen3.5, and Llama3/4. Its bet, unchanged since 2024, is that `DTensor`/`DeviceMesh` are mature enough to build FSDP2/TP/CP/EP/PP on top of, inside roughly 7-9K lines versus Megatron-LM's 93-269K — auditability and PyTorch-nativeness, not comprehensiveness. The MoE stack is newer and moving fast rather than battle-tested.

**Five dispatcher backends.** `LocalTokenDispatcher` (EP=1), `AllToAllTokenDispatcher` (plain collectives), `TorchAOTokenDispatcher` (16/32-element alignment for FP8/MXFP8 grouped-GEMM), `DeepEPTokenDispatcher` (native DeepEP v2), and `HybridEPTokenDispatcher` (GB200/NVLink72-specific, with a non-blocking dispatch mode trading bounded token-drop risk for CPU-sync-free dispatch). A sixth, `MinimalAsyncEPTokenDispatcher`, is explicitly scoped EP-with-SP=1-only per its own docstring. A dedicated `HybridEPTokenDispatcher` living in core code (not experiments) is a concrete signal the PyTorch team is targeting exactly the NVL72 topology a 100k-GPU GB200 build would use.

**MXFP8 real-training numbers.** The richest datapoint in this space: PyTorch+Nebius, DeepSeek-V3-671B on 256×B200, TP=2/PP=2/EP=32: BF16 baseline 651 tok/s → BF16+DeepEP 859 tok/s (+32%) → MXFP8 grouped-GEMM+DeepEP 918 tok/s (+41% over baseline), loss convergence matching BF16 over 1,500 steps (the TorchTitan blog, Mar 2026, updated May 2026). MXFP8 was applied only to grouped GEMMs, not dense linears, "because torch.compile does not yet support MXFP8 linear layers when combined with tensor parallelism" — 41% is not the ceiling.

**Real frontier-adjacent usage.** Arcee AI's Trinity Large (400B total/13B active, top-4 of 256+1 experts, 17T tokens) trained on 2,048 B300 GPUs on a modified torchtitan (arXiv:2602.17004, Feb 2026) — the single best "production grade" data point, unusually candid about failure modes: the team fell back from MXFP8 to BF16 for stability and needed six concurrent interventions (a new load-balancing scheme "SMEBU," z-loss, sequence-wise aux loss, more dense layers, intra-doc attention masking, BF16 fallback) to eliminate router-collapse loss spikes. AMD/TensorWave ran DeepSeek-V3-671B on 1,024 MI325X via a modified torchtitan plus Primus-Turbo kernels, 96-97% scaling efficiency 256→1024 GPUs, 2.77x cumulative speedup (AMD blog, Dec 2025). Every public benchmark tops out in the 1K-2K GPU range.

**Fault tolerance is thin.** `torchft` (DiLoCo/semi-sync via a Rust "Lighthouse" quorum service) is a torchtitan *experiment*, its own README calling it "an ongoing development effort, subject to change." The most recent validation (AMD/ROCm, Feb 2026) is 32-GPU-only, no throughput-overhead numbers published. Its architectural ceiling matters independent of scale: torchft tolerates losing an *entire* HSDP replica group, not single-GPU failures inside a 3D/4D-parallel job — the more relevant primitive for a 100k-GPU EP+PP+FSDP job is NCCL-level process-group reconfiguration, which torchft doesn't address (see Ch21).

**GraphTrainer — the compiler-driven bet, explicitly still experimental.** `experiments/graph_trainer/` captures an entire training step as one flat FX graph, running activation checkpointing, CUDAGraph, CPU offload, and comm/compute overlap as composable graph passes rather than competing eager-mode hooks. Its own MANIFESTO.md: "each combination [of FSDP2 + AC + compile + CUDAGraph] needs its own workaround, and workarounds for one pair can break another." Sub-components include **SimpleFSDP** (arXiv:2411.00284, FSDP collectives as traceable graph nodes), **AutoParallel** (GSPMD-style automatic sharding search), and a quotable GB200 hard number: NUMA-aware CPU offload documents ~350 GB/s NUMA-local versus ~120 GB/s cross-NUMA D2H/H2D over NVLink-C2C. The July 2026 composability table is the honesty check: TP/CP/DCP/CUDAGraph/EP done; **Float8/MXFP8 training, EP+AC, EP+PP, graph-based PP, and micro-batch overlap all still in-progress**. GraphTrainer is explicitly barred from leaking into core torchtitan by repo policy.

Corroborating context: Ed Yang (PyTorch core/compiler) gave an unusually candid August 2025 assessment — secondhand but high-trust — that DTensor lacks a "jagged sharding" concept needed for imbalanced-routing EP, and `torch.compile` is "not SPMD-first." The code confirms this directly, with EP dispatch/combine dropping out of DTensor into plain local tensors at the `GroupedExperts` boundary, an explicit comment noting the shape "cannot be easily expressed as DTensors."

**Bottom line:** torchtitan's parallelism composition and MoE/quantization engineering look genuinely state-of-the-art and GB200-aware, but the operational/reliability layer at 50-100x today's largest demonstrated scale is unproven in public, and Megatron-LM (plus NVIDIA's internal tooling) remains the safer bet purely on "has anyone run this at anything approaching 100k GPUs" grounds — a caveat that, fairly, applies to every open framework.

## MaxText and AI-Hypercomputer: compiler-first, and honest about not being GB200-ready

MaxText is Google's reference framework — pure Python/JAX + XLA + Pallas — betting the opposite of Megatron: "trusts the underlying compiler to handle the complexities of hardware optimization... only for the most performance-critical operations... does MaxText use custom kernels" (its own docs, June 2026). It ships native configs for DeepSeek V3/V3.1/V3.2, a DeepSeek-V4-shaped config, Kimi-K2 (1T, 384 experts), Qwen3.5-397B-A17B (GatedDeltaNet every 4 layers), and Qwen3-480B-A35B across roughly 85 model YAMLs.

**The GB200 gap is the single most consequential, checkable finding here.** `accelerator_to_spec_map.py` — defining every valid compile-topology for AOT compilation — has roughly 200 TPU entries and exactly one GPU entry: `"a3"` (H100, single-host only), zero A4/A4X/GB200/B200 entries anywhere (checked June 2026). GPU attention doesn't use MaxText's own Pallas kernel at all — the docs state "Flash does not work on GPUs" in MaxText, so GPU training depends on NVIDIA Transformer Engine's cuDNN kernels instead, an entirely separate code path. The only GPU benchmarks shipped are Llama2-7B on 1-16x A3 nodes; no MoE-at-scale GPU numbers exist. A `sharding.md` doc misdescribes GB200 NVL72 as "72 hosts or 576 chips" (canonical is 72 GPUs across 18 hosts) — unedited aspirational text, not a validated recipe. Google Cloud markets A4X (GB200 NVL72) GA with "4x LLM training perf vs A3" (2025), but that optimization hasn't been upstreamed into MaxText's config files; JAX itself only added Blackwell support in the April 2026 release — the compiler layer for GB200 is roughly three months mature as of mid-2026. MaxText-as-shipped is not the tool for a 100k-GB200 cluster today; its proven scale story (9,216-chip Ironwood pods, a 50,944-chip v5e job) is entirely TPU/Pathways-native.

The compiler-driven architecture is still worth studying for design principles. The **3-layer sharding system** (physical mesh → logical axes → per-tensor rules, with XLA deriving collectives from hints) and **arithmetic-intensity-first culture** generalize: EP's arithmetic intensity reduces to simply the MLP dimension, independent of batch size — EP communication cost is set by model width, not batch, exactly why EP-heavy MoE training is comms-bound differently than dense-model FSDP. Five pre-built mesh configs specifically target large-MoE cases, including `ep-as-cp`/`cp-as-ep` (repurposing the otherwise-idle expert axis for context-parallel duty outside MoE FFN blocks — a "free lunch" a hand-coded Megatron-style stack would need bespoke code to replicate) and **`pipeline-large-moe`**, built explicitly for "extremely large-scale MoE jobs (such as DeepSeek models)." A "Sharding Dump" regression harness with golden JSON files per model/topology protects these configs against silent regressions — engineering culture worth borrowing regardless of framework choice.

MaxText's MoE kernel story is two-speed: **Megablox** (mature, Pallas-native grouped-matmul) versus **Tokamax Ragged Dot** (the officially "recommended" dropless path, but BF16 support is still "in progress" — only FP8 is complete). Anyone benchmarking MaxText MoE in BF16 today is effectively still on Megablox.

The goodput culture is the part most worth stealing regardless of vendor: **93% goodput for Gemini 2.5** across multiple 8,960-chip TPU v5p pods spanning data centers, **97% for Gemini 1.0** on TPU v4 (arXiv:2606.15870, Jun 2026, citing the Gemini technical reports). A July 2026 Google Cloud demo killed a TPU mid-run and recovered a 48-chip job in under two minutes total — 13s detection, 50s Kubernetes reschedule, 5.39s checkpoint restore, 12.7s to first post-recovery step — losing only 88 steps. That granularity in a public elastic-recovery breakdown is rare (see Ch21).

## The rest of the field

**InternEvo's successor, XTuner V1, makes a claim worth stress-testing.** InternEvo (arXiv:2401.09149) used simulator-searched 4D parallelism for up to 4.8x MFU on 7-65B dense models. Its successor, `InternLM/xtuner`, inverts industry consensus: claiming **pure FSDP with zero expert parallelism up to 200B-scale MoE**, only intra-node EP at 600B — "the first framework where FSDP throughput beats traditional 3D-parallel (TP+PP+EP) schemes for MoE above 200B parameters" — with no absolute tokens/s or MFU given, only an unlabeled chart, and no backing paper as of July 2026. This directly contradicts the assumption underneath DeepSeek's DualPipe+DeepEP, Megatron-Core's Parallel Folding, and MegaScale-MoE's SP+EP hybrid, all of which treat non-trivial EP as structurally necessary at fine-grained, multi-hundred-billion-parameter scale. If it replicates independently it's consequential; if it's an artifact of Ascend interconnect specifics or a weak baseline, it's a cautionary tale — treat it as unverified, not a design input.

**veScale has pivoted to FSDP-specific work.** ByteDance/Volcengine's `volcengine/veScale` is explicitly "an internal ByteDance library, of which this repo open-sources a small piece." 2026 energy has moved to **veScale-FSDP** (arXiv:2602.22437, 5-66% higher throughput, 16-30% lower memory than existing FSDP systems, scaling to tens of thousands of GPUs) and a sibling multimodal project, VeOmni.

**PanGu Ultra MoE trains on MindSpeed** (Huawei's Megatron-derived platform), not the separately branded MindSpore — no independent MindSpore training-framework detail appears to exist publicly beyond this, a real gap worth flagging rather than papering over. The paper (arXiv:2505.04519) discloses 718B total/39B active on 6,000+ Ascend 910B NPUs at TP=8/PP=16/VPP=2/EP=4 — notably *small* EP leaning on deep PP, the inverse of the DeepSeek/Megatron playbook. Its most reusable contribution is a **bottom-up simulator** sweeping ~10,000 configs before committing GPU time (88.9-90.1% simulated-vs-measured accuracy) — the most rigorous public "simulate before you burn GPU-hours" methodology outside internal lab tooling, and a direct template for a 10T/200B config search. Bottom line: 30.0% MFU/1.46M tok/s at 6K NPUs. A conflation trap worth flagging: secondary sources routinely cite ">52% MFU" for "Pangu Ultra" as the MoE number — that figure is the 135B *dense* sibling model; the MoE model gets 30.0%.

**AMD's Primus is the real Megatron-compatible AMD stack** — a three-layer system (Primus-LM orchestrating Megatron-LM/Megatron-Bridge/TorchTitan backends, Primus-Turbo kernels, Primus-SaFE cluster management) with CLI 1.0 shipping November 2025 and AMD's own docs describing it as "battle-tested... with hundreds of GPUs" — capping validated scale at hundreds, not tens-of-thousands. The Zyphra/IBM/AMD ZAYA1 paper (arXiv:2511.17127, Dec 2025), a modified Megatron-LM fork integrated with Primus on 2,048 MI300X GPUs (8.3B-total MoE, three orders of magnitude below this run's target), is the richest fresh AMD systems source: MI300X HBM bandwidth saturates at 4.330 TB/s versus H100's 3.036 TB/s, and xGMI requires all GPUs in a node to participate in a collective to reach full bandwidth (`B_per-GPU = (n-1)·B_link`) — a structural difference from NVSwitch, where partial-node collectives aren't bandwidth-starved. MLPerf v6.0 (June 2026) shows MI355X at parity-to-1.16x versus B200 single-node but falling to 0.96x at 8-node scale — the AMD advantage compresses as scale-out grows, a much smaller gap than SemiAnalysis found in December 2024 but the same direction. None of this makes AMD a candidate for a GB200-based build; it's the right comparison set for vendor-risk hedging.

**DeepSpeed is not dead but is not where frontier pretraining happens** — and it received a substantive commit the day before the July 13 evidence snapshot closed: "Add AutoEP + AutoTP parallel folding" (#8064, merged July 12, 2026, verified via `git log`), letting EP groups span TP lanes and dense-DP ranks without EP nested inside DP. Conceptually adjacent to, though independently engineered from, Megatron-Core's Parallel Folding — two teams converging on the same abstraction in the same window. AutoEP notably borrows **TorchTitan's grouped-GEMM kernels** rather than building its own. The verdict across everything surveyed here: Megatron-Core/TorchTitan own frontier-scale pretraining; DeepSpeed's remaining edge is ZeRO-3 CPU/NVMe offload for memory-constrained fine-tuning plus HF-ecosystem AutoTP/AutoEP convenience. No 2025-2026 frontier pretraining disclosure we can find — DeepSeek, MegaScale-MoE, PanGu Ultra MoE, ZAYA1 — uses DeepSpeed as its core engine.

## The MoE-systems research the frameworks are absorbing

Megatron-Core and torchtitan's dispatcher zoos are absorbing a wave of MoE-systems papers, several already shipped at scale exceeding either open framework's public demonstrations. **COMET** (arXiv:2502.19811, ByteDance+SJTU, MLSys 2025) attacks a problem distinct from DeepEP's: instead of optimizing the all-to-all kernel itself, it does thread-block-level decomposition of the tensors between MoE sub-operations, splitting GPU thread blocks into concurrent "communication" and "computation" blocks — 1.96x single-layer, 1.71x end-to-end speedup, hiding 86.5% of communication latency versus 68.6% (Tutel) and 29.2% (FasterMoE). It runs in ByteDance's ten-thousand-GPU production clusters since early 2025, open-sourced inside `bytedance/flux` — a materially different mechanism from DeepEP that could in principle layer on top of either NVSHMEM-based EP, though no public source discusses combining them, and **Flux/COMET have no Blackwell backend as of their last commit (August 2025)**, verified via the GitHub API — a real, checkable gap for anyone assuming ByteDance's stack is GB200-ready.

**MegaScale-MoE** (arXiv:2505.11432, ByteDance/PKU, EuroSys '26) is the fullest disclosed production MoE systems paper after DeepSeek's own. It inverts standard practice: sequence parallelism instead of TP for the attention path (~25% of TP's communication latency intra-node), and for the FFN path, when top-k exceeds the EP world size, replaces standard all-to-all dispatch with allgather+local-scatter+reduce-scatter — a different trick from both DeepEP and HybridEP. Measured: 352B MoE, 1,440 H800, 1.41M tok/s, 1.88x over Megatron-LM; a Mixtral-8x7B ablation across H800/H20/A100 shows a consistent 1.58x MFU gain, indicating the mechanism generalizes across GPU generations. Production use confirmed training a 200B-total/20B-active MoE across 10,000+ GPUs for months.

**Triton-distributed** (ByteDance-Seed, arXiv:2504.19442) is the compiler-level answer to Flux/COMET's hand-written kernels — compiling comm-overlap kernels from a tile-centric Triton DSL, up to 44.97x versus raw NCCL/RCCL on microbenchmarks, and it's the only one of these libraries with **working AMD ROCm backends today**, independently confirmed by AMD's own ROCm blog. For any build hedging vendors or anticipating Blackwell gaps in ByteDance's Hopper-era libraries, this is the more portable bet.

Newer academic ideas worth flagging as the visible edge of what comes after EPLB, none yet adopted in a frontier lab's public stack: **ScMoE** (arXiv:2404.05019, decouples communication from layer-sequential ordering, 1.49x train/1.82x inference, already production-adopted by Meituan's LongCat-Flash — one of the few academic MoE-systems ideas that has actually shipped); **FlowMoE** (arXiv:2510.00207, NeurIPS 2025, unified pipeline scheduler with joint expert-relayout/routing optimization, 1.13-1.82x over prior schedulers); and two 2026 EPLB-successors, **UltraEP** (arXiv:2606.04101) and **LAER-MoE** (arXiv:2602.11686), both targeting dynamic training-time expert re-layout on rack-scale nodes but not yet cleanly quantified in public — low-confidence entries worth a follow-up if expert-rebalancing becomes critical-path.

## Fork-vs-build guidance

One recommendation, stated plainly: **fork Megatron-Core, do not build from a clean-sheet PyTorch-native or JAX stack, and do not treat the fork as configuration-only — budget for sustained upstream patching.**

Megatron-Core is the only framework here with (a) a primary-source paper detailing its MoE stack down to the dispatcher class and CUDA-Graph memory layout, (b) validated throughput on the exact target hardware generation for the exact model class, and (c) an open issue tracker letting a team pre-identify correctness bugs — not performance bugs — before committing capital. TorchTitan's GB200-aware MoE/quantization engineering is genuinely comparable or better in places, but every public validation point tops out at 1-2K GPUs and its fault-tolerance layer is validated only at 32 GPUs with a replica-group failure model that doesn't match a 100k-GPU job's failure granularity. MaxText's compiler-driven sharding and goodput culture are worth studying for principles, but the framework has zero production GB200 support at the code level — not a close call. AMD's Primus and PanGu Ultra MoE's MindSpeed are real, working systems built for non-NVIDIA silicon (ROCm/MI-series, Ascend NPUs respectively); Baidu's PaddlePaddle-Fleet stack is actually NVIDIA-native (ERNIE 4.5's disclosed 47%-MFU run used 2,016 H800s) but validated three orders of magnitude below the target GPU count here — none of the three is a candidate for a 100k-GPU GB200 build. DeepSpeed has structurally exited frontier pretraining.

The concrete modification list to scope before committing to a Megatron-Core fork:

- **Fix or work around the EP+comm-overlap+uneven-VPP deadlock** (#5749/#1810) before it's hit at scale — patch the shared-CUDA-event synchronization for a consistent same-order invariant, or restrict initial de-risking runs (Ch9) to uniform VPP layouts until the upstream fix lands.
- **Validate the local-checkpoint corruption path** (#5281) end-to-end before relying on `ft_launcher`'s fast-restart mechanism; add a checksum step on local-checkpoint restore with automatic fallback to the last-known-good global checkpoint.
- **Test Megatron-FSDP + optimizer CPU offload checkpointing under real load** (#4910) rather than assuming the combination works because both features are individually documented as production-ready.
- **Budget engineering time to mature ECHO and Paged Stashing** beyond their current merged-but-not-default state; plan de-risking runs both with and without full CUDA-Graph capture, since sync-free full-iteration graphs remain roadmapped, not default.
- **Evaluate HybridEP against DeepEP directly on target GB200/GB300 hardware early** — HybridEP's Table 7 latency advantage is specifically a GB200 NVL72 story; DeepEP alone, tuned for Hopper, won't capture it. Run the comparison at the per-microbatch token counts the real batch plan produces, not the fat-microbatch shapes the published tables use — Ch18's expert-batch starvation and Ch20's message-size analysis show the production regime sits at far smaller messages, where the published latency numbers do not extrapolate.
- **Treat Megatron-Bridge as checkpoint-interop and recipe-scaffolding only**, not the training loop, given #4590's own acknowledgment that downstream frameworks still patch Megatron-Core directly.
- **Consider layering COMET/Flux-style overlap or Triton-distributed on top of the chosen EP dispatcher** as a research track, not a day-one dependency, given the unresolved Blackwell-backend gap in ByteDance's Hopper-era libraries — Triton-distributed's compiler-first, cross-vendor approach is the safer bet if pursued.
- **Prototype LatentMoE** (already production in Nemotron-3 Super/Ultra) as a dispatch-payload-compression lever independent of the EP-degree decisions in Ch18, since it addresses a different point in the design space than either large-EP (DeepSeek/Megatron) or small-EP-plus-deep-PP (PanGu Ultra MoE).

## Implications for the 10T/200B run

Frameworks are not a solved problem at this scale in mid-2026 — they are a set of specific, named, currently-open correctness bugs sitting exactly at the intersection of the features a 10T/200B-active MoE will use simultaneously (EP, communication overlap, uneven pipeline layouts, FSDP, CPU offload, fault tolerance). Megatron-Core is the right base precisely because its maturity means these gaps are visible and filed, several already partially fixed, rather than undiscovered. A team should treat the framework decision as settled early and redirect the saved decision-cycles toward the modification list above — particularly the EP-comm-overlap deadlock and the local-checkpoint corruption path, both silent-failure modes far more expensive to discover mid-run at 100,000 GPUs than in a pre-flight validation pass (Ch22). The dropless-MoE CUDA-Graph story and the HybridEP-versus-DeepEP choice are the two levers most likely to move measured MFU on the actual target hardware, and both are new enough — merged within months of the snapshot — that no public source yet demonstrates them composed together at anything near 100k-GPU scale. That gap, more than any single missing feature, is the honest state of frameworks for a run of this size in July 2026.
