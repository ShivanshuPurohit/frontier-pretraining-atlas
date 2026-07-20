# Chapter 23 — Benchmarks and the Money Math

Every other chapter here reasons from first principles — rooflines, formulas, architectural argument. This one exists because first principles alone cannot tell you what a training run actually costs: that requires audited throughput numbers from clusters that exist, not clusters that could exist. The public record for that is thinner than the industry's marketing implies — one benchmark suite, a handful of vendor-published recipe tables, and a small set of primary-source cost analyses — but it is enough to build a defensible number. This chapter builds one, shows every step, and ends with the table you would need to defend simultaneously to a CFO asking "how much" and a chief scientist asking "why that MFU."

## MLPerf Training as the only audited ground truth

MLPerf Training's large-language-model benchmark has been rebased twice in the window that matters here. The **v3.0** round (June 2023) debuted GPT-3-175B, with NVIDIA and CoreWeave setting the marquee record at 10.9 minutes on 3,584 H100 GPUs. **v4.0** (June 2024) tripled submission scale to 11,616 H100 GPUs and cut time-to-train to 3.4 minutes. **v4.1** (November 2024) brought the first Blackwell appearance, but only as a 64-GPU air-cooled HGX B200 preview entry — not the rack-scale GB200 NVL72 — reporting roughly 2x per-GPU gains over H100 on GPT-3-175B and 2.2x on Llama-2-70B-LoRA. The pivotal change came in **v5.0** (announced May 2025): GPT-3-175B was retired in favor of **Llama-3.1-405B pretraining** as the flagship large-model benchmark — AllenAI C4-en data, 8,192-token context (4x GPT-3's), a fixed log-perplexity target of 5.6 rather than a fixed token count, the Mixtral-8x22B 32k-vocab tokenizer, NeMo Framework as reference implementation, median of three runs (MLCommons, "MLCommons MLPerf Training Expands with Llama 3.1 405B," May 2025). **v5.1** (November 2025) added GB300 NVL72 (Blackwell Ultra), the first NVFP4 MLPerf Training submission (with FP8 "healing" on final iterations to hit the accuracy target), and the first 800 Gb/s NIC platform (Quantum-X800 InfiniBand) to appear in the benchmark. **v6.0** (June 16, 2026) matters most here: it added **DeepSeek-V3-671B pretraining as MLPerf's first audited large-MoE benchmark** — NVIDIA/CoreWeave trained it to target on **8,192 GB300 NVL72 GPUs in 2.02 minutes**, with GB300 delivering roughly 1.6x GB200 per-GPU throughput on the new benchmark, and — the competitive tell — zero AMD, Google, or AWS large-MoE submissions. An audited MoE throughput number at 8k-GPU rack scale is precisely the class of evidence the rest of this chapter previously had to approximate from vendor recipe tables; the dense-era framing that Llama-3.1-405B was the only audited large-model number in existence is, as of that round, obsolete.

One methodological wrinkle matters for everything that follows: because the benchmark trains to a fixed perplexity target rather than a fixed token count, tokens processed is not identical across submissions, and faster-converging recipes reach the target with fewer tokens. Any MFU back-calculated from an MLPerf wall-clock time inherits this ambiguity — it is a throughput number, not a pure FLOPs-utilization number, unless the submitter separately discloses tokens processed.

| Round | Date | Flagship benchmark | Record | Notes |
|---|---|---|---|---|
| v3.0 | Jun 2023 | GPT-3-175B | 10.9 min / 3,584 H100 | debut |
| v4.0 | Jun 2024 | GPT-3-175B | 3.4 min / 11,616 H100 | 3.4x scale-up |
| v4.1 | Nov 2024 | GPT-3-175B | B200 preview, 64 GPU, ~2x/GPU vs H100 | not GB200 rack |
| v5.0 | May–Jun 2025 | Llama-3.1-405B (new) | 27.33 min / 2,496 GB200 | CoreWeave+NVIDIA+IBM |
| v5.1 | Nov 2025 | Llama-3.1-405B | 10 min / 5,120+ GB300 | first NVFP4 submission |
| v6.0 | Jun 2026 | + DeepSeek-V3-671B MoE (new) | 2.02 min / 8,192 GB300 | first audited large-MoE benchmark |

Sources: MLCommons results pages, NVIDIA's developer blogs, and CoreWeave's blog, per round.

## GB200 NVL72 submissions: a reproducible weak-scaling curve

The largest verified GB200 Llama-3.1-405B submission is CoreWeave + NVIDIA + IBM on the "Carina" cluster (39 racks, IBM-built): **2,496 GB200 GPUs, 27.33 minutes**, v5.0, described as "34x larger than the only other cloud provider GB200 submission" (CoreWeave's blog, Jun 2025). CoreWeave published the full scaling ladder, which is rare enough to be worth reproducing — most vendor scaling claims arrive as a single ratio, not a curve an outside reader can recompute:

| GPUs | Time-to-train | Ideal time (linear from 512) | Computed efficiency |
|---|---|---|---|
| 512 | 121.1 min | 121.1 min (baseline) | 100% |
| 1,024 | 62.1 min | 60.55 min | 97.5% |
| 1,536 | 42.46 min | 40.37 min | 95.1% |
| 2,048 | 32.6 min | 30.28 min | 92.9% |
| 2,496 | 27.33 min | 24.84 min | **90.9%** |

```
efficiency(N) = ideal_time(N) / actual_time(N),  ideal_time(N) = T(512) × (512/N)
```

This reproduces CoreWeave's stated "91% scaling efficiency from 512 to 2,496 GPUs" almost exactly (the recomputation is ours) — a useful confirmation that the underlying numbers are internally consistent, and a genuine data point (not a marketing ratio) for what weak-scaling efficiency looks like on real NVL72 hardware in the 512–2,500 GPU range. At the same round, NVIDIA's own 512-GPU Blackwell entry hit 121.09 minutes against Hopper's 269.12 minutes at the same GPU count — a 2.2x generational gain — and NVIDIA separately states GB200 NVL72 delivers "up to 1,960 TFLOPS of training throughput" per GPU on this benchmark at 512-GPU scale. Dividing 1,960 by the dense-FP8 peak established below (5,000 TFLOPS/GPU) gives an implied **39.2% MFU** for this specific submission — already a useful anchor before the more systematic derivation in the next section.

Software-only gains between rounds were large: the *same* GB200 hardware class went from 27.33 minutes (2,496 GPUs, May 2025) to 18.79 minutes (2,560 GPUs, Nov 2025) — 45% *faster*, i.e., a 31% wall-clock reduction (27.33 → 18.79 min), in six months with almost no change in GPU count. This mirrors the H100-generation software-maturation curve documented independently by SemiAnalysis (below) and is the single best evidence available that a "bet-your-run" MFU number measured on day one of a new SKU understates what the same hardware will deliver a year later. v5.1's max GB300 submission reached 5,120+ GPUs at 10 minutes — a new record — though the specific "85% scaling efficiency, 512→5,120" figure NVIDIA quotes for that round could not be independently re-derived, since no full intermediate ladder was published; treat it as vendor-reported, not re-verified. Lambda's own v5.1 submission directly compared GB300 vs GB200 vs 64×B200 on Llama-2-70B-LoRA: 1.26 min (GB300) vs 1.598 min (best GB200, Oracle) vs 2.019 min (64×B200) — a 27% GB300-over-GB200 speedup, decomposed by Lambda as roughly 1.12x hardware, 1.13x software-stack, 1.13x FP4-vs-FP8 precision (these three factors multiply to ~1.43x, not the stated 1.27x — Lambda's own attribution is reported here as stated, without reconciling the arithmetic).

Two absences round out the picture. AMD closed materially in v5.1: MI355X progressed from MI300X's 27.95-minute Llama-2-70B-LoRA result through MI325X's 21.05 minutes to **10.18 minutes**, landing within 3–6% of NVIDIA's B200/B300 on the same benchmark and within 5–6% on Llama-3.1-8B pretraining — the closest AMD has come to NVIDIA parity on a from-scratch MLPerf Training submission (AMD's ROCm blogs, Nov 2025). AMD has never submitted a Llama-3.1-405B result. Google has never submitted a TPU result on Llama-3.1-405B through v5.1 — its competitive perf/$ claims against Blackwell (2.1–2.5x vs v5e/v5p, >50% MFU improvement from Trillium "host-offloading") are self-reported MaxText benchmarks, not MLPerf-audited numbers, and no on-the-record explanation for the absence exists in the public record.

## Deriving MFU: the vendor-published recipe tables, and a SKU trap worth naming

The most load-bearing MFU evidence here is not MLPerf at all — it is NVIDIA's own Megatron-Bridge performance-summary documentation, which publishes "Model TFLOP/sec/GPU" (actual forward+backward FLOPs, directly comparable to peak hardware FLOPs) across DGX-H100, DGX-B200, and DGX-GB200 for a battery of recipes including dense Llama-3/3.1 and MoE models (DeepSeek-V3, Qwen3-30B-A3B, Qwen3-235B-A22B). Computing MFU from this table requires the correct dense-FP8 peak denominator per SKU, and this is where a genuine trap lives in the public literature: NVIDIA's own GB200 NVL72 product page and datasheet report **2:4 structured-sparse** FLOPs without always labeling them as such, and a separate SKU — HGX B200 (air-cooled, 1,000W max TDP) — is routinely conflated with GB200 NVL72 (liquid-cooled, 1,200W max TDP) even though they are the same die at different power envelopes and genuinely different dense-FP8 peaks: **4,500 TFLOPS/GPU for HGX B200, 5,000 TFLOPS/GPU for GB200 NVL72** — a clean, uniform +11.1% across every tensor-core precision (5000/4500 = 2500/2250 = 10000/9000 = 10/9 exactly), confirmed against three independent NVIDIA properties plus a third-party technical wiki (datasheet #3384703; Ch16 carries the full reconciliation). The trap is not hypothetical — exactly this substitution circulates: dividing the Megatron-Core-reported 1,048 TFLOPS/GPU sustained figure for DeepSeek-V3-685B-scale training (arXiv:2603.07685, Mar 2026) by the HGX 2,250-TFLOPS dense-BF16 denominator yields a claimed 46.6% "GB200 BF16 MFU," when the corrected GB200-specific denominator (2,500 TFLOPS dense BF16) yields **41.9%** — a ~10% relative overstatement from a single wrong SKU assumption. And the report's own Table 11 marks the underlying measurement as MXFP8, so the honest figure is in fact **21.0%** against the 5,000 TFLOPS FP8/MXFP8 denominator (see Ch17). Every MFU number in this chapter uses the correct 5,000 TFLOPS/GPU dense-FP8 (2,500 TFLOPS dense-BF16) GB200 NVL72 peak; anyone importing a "B200" FLOPs figure from elsewhere should confirm which SKU it actually describes before trusting it — see Ch18 for a second instance of the same hazard.

With that denominator fixed, NVIDIA's own best-tuned recipes produce the single clearest empirical fact in this chapter:

| Model | H100 (peak 1,979 TFLOPS) | B200 (peak 4,500 TFLOPS) | GB200 (peak 5,000 TFLOPS) |
|---|---|---|---|
| Llama3-8B (dense) | 36.6% | 34.3% | 32.3% |
| Llama3-70B (dense) | 36.7% | — | 35.8% |
| Llama3.1-405B (dense) | 38.6% | 37.2% | 36.8% |
| DeepSeek-V3 (MoE, 37B/671B active, EP64) | 17.1% | 13.1% | 18.0% |
| Qwen3-30B-A3B (MoE) | 12.2% | 9.2% | 10.5% |
| Qwen3-235B-A22B (MoE, EP64) | 9.0% | 13.5% | 13.2% |

Source: NVIDIA's Megatron-Bridge performance-summary docs, **versioned 0.1.0 snapshot** — see the container correction below. The MFU columns are our own division by the corrected peaks.

In this 0.1.0-era snapshot, dense models cluster tightly at **32–39% MFU** across three hardware generations while MoE models fall to **9–18%** — roughly half to a third — and the gap holds regardless of accelerator generation, which rules out "the chip isn't mature yet" as the explanation for its *existence*. Its *size*, however, turns out to be a software artifact we caught in our own audit: the table above came from a versioned `megatron-bridge/0.1.0` documentation page, and the **live** performance summary — re-fetched July 13, 2026, reporting the NeMo **26.06 container** — moves every MoE number materially. DeepSeek-V3 on GB200 now sustains 1,292 TFLOP/s (**25.8% MFU**, up from 18.0%, a +43% relative gain in roughly one container generation) and 1,648–1,670 TFLOP/s on GB300 (**≈33%**); Qwen3-235B-A22B on GB200 reaches 1,092 TFLOP/s (**21.8%**, up from 13.2%); dense Llama-3.1-405B rises to 2,129 TFLOP/s on GB200 (**42.6%**) and 2,646 on GB300 FP8 (**52.9%**, with an NVFP4 entry at 3,575 TFLOP/s). Two consequences follow. First, the honest current reading of the dense-vs-MoE gap is **dense 32–53% / MoE 18–33%, a ratio of roughly 1.65x** — the sparsity penalty is real but shrinking fast, which is the same software-maturation curve this chapter documents on Hopper, now caught in the act on Blackwell. Second, a citation doctrine worth adopting from the incident: **any number pulled from a living vendor documentation page must carry a container or release version, not just an access date** — the single largest numeric drift we found anywhere came from a versioned doc URL the vendor had quietly superseded. The mechanism behind the gap (our interpretation, not a directly sourced claim) is arithmetic intensity: per-token compute in a top-k MoE layer scales with `top_k × expert_hidden_dim`, while per-token expert-parallel all-to-all traffic (dispatch + combine) scales with `d_model × bytes_per_element`, largely independent of expert count or width — so compute-to-communication ratio falls as tokens fan out across more, smaller experts, even holding total active parameters fixed. This is the single most load-bearing empirical fact for sizing a 2%-active 10T/200B MoE run, because DeepSeek-V3 (5.5% active) and Qwen3-235B-A22B (9.4% active) are both denser-active than the strawman in Ch7 — extrapolating the trend, a 2%-active model should land *below* both, not between them.

SemiAnalysis's independently measured production economics corroborate the same gap from a different methodology entirely (per "H100 vs GB200 NVL72 Training Benchmarks," Aug 2025, paywalled): DeepSeek-670B (MoE) achieved only **16.6% BF16 MFU on H100**, against 54.5% for dense Llama3-405B on the identical hardware and framework family — attributed to massive dispatch/combine traffic and to MoE communication volume scaling with `top_k` (routing to 4 experts instead of 1 roughly quadruples comm volume). On GB200 specifically, the same document tracked DeepSeek-670B's MFU trajectory improving fast: GB200 delivered only 1.3x H100's tok/s/GPU in May 2025 (not compelling — GB200 was actually 7% *more expensive* per token at that point, $0.684 vs $0.626/M tokens), reaching 2.5x by July 2025 ($0.307 vs $0.468/M tokens, a 1.5x perf/TCO advantage), with NVIDIA's own software roadmap projecting 4.7x throughput and ~42% BF16 MFU by December 2025. The mechanism is GB200's 72-GPU NVLink scale-up domain: 64-GPU all-to-all collectives run **18x faster** than the same collective on an 8-GPU-domain H100 cluster — this is the entire reason GB200 wins specifically for MoE and is comparatively unremarkable for dense models. Full DeepSeek-670B pretrain cost on GB200: $4.5M as of July 2025, projected to fall to $2.5M by December 2025 as software matures.

## DeepSeek-V3: reconstructing MFU from the paper's own numbers, and debunking a myth

DeepSeek-V3's technical report states 14.8T pretraining tokens, 37B active/671B total parameters, and 2.664M H800-GPU-hours specifically for the pretraining phase, on a 2,048-GPU cluster (arXiv:2412.19437, Dec 2024). Using the standard `FLOPs ≈ 6 × N_active × D_tokens` approximation:

```
FLOPs_needed   = 6 × 37e9 × 14.8e12 = 3.286e24 FLOPs
FLOPs_available = 2,664,000 GPU-hrs × 3,600 s/hr × 1.979e15 FLOP/s  (H800 SXM FP8 dense peak)
               = 1.898e25 FLOPs
MFU = 3.286e24 / 1.898e25 = 17.3%
```

This lands within a point of the SemiAnalysis-measured 16.6% figure for DeepSeek-670B on H100 above, despite being a wholly independent reconstruction from a different primary source — real convergent evidence that MoE MFU in the mid-teens is the correct anchor for a ~5%-active model on Hopper-class hardware, not an artifact of one methodology. A widely circulating "~23%" figure for DeepSeek-V3 implicitly uses the PCIe H800 variant's lower ~1.5 PFLOPS peak rather than the SXM 1.979 PFLOPS figure the cluster's NVLink-domain topology almost certainly used; a separately-circulating **"58% MFU"** claim found in casual write-ups does not survive reconstruction from the paper's own stated GPU-hours and token count and should be treated as a misreading, not a fact — worth flagging explicitly since it would badly distort any DeepSeek-V3-anchored Hopper-class MFU baseline if laundered forward. The 17.3% here uses the standard `6×N_active×D` cross-model approximation for direct comparability against the dense models in the table above; Ch4's own reconstruction, built from DeepSeek's disclosed 250-GFLOPs/token hardware-reflections figure rather than the 6ND shortcut, lands slightly higher at FP8-normalized ≈19.5% (BF16-normalized ≈39.0%). Both are mid-teens-to-twenties FP8-MFU reconstructions, an order of magnitude from the debunked 58% figure; read 17.3% here as the standardized-methodology number used for cross-model comparison and Ch4's 19.5% as the more precise single-model figure, not as competing claims.

## Hopper-era dense baselines and the software-maturation curve

Dense-model MFU references anchor the optimistic end of the sensitivity bands used below. MegaScale (ByteDance, 175B dense, 12,288 Ampere-class GPUs) reported **55.2% MFU** (arXiv:2402.15627, NSDI'24). Llama-3-405B's own paper states 38–43% overall BF16 MFU, with a Meta engineer separately tweeting 43% peak at 8K-GPU scale (arXiv:2407.21783, Jul 2024). SemiAnalysis's own re-benchmark of the identical Llama-3-405B recipe roughly a year later, on NVIDIA's own DGXC scripts across 576–2,304 H100 GPUs, found **FP8 MFU ~43%, BF16 MFU ~54%** — meaning software maturation pushed real-world MFU well above the as-shipped figure within about a year, at $1.95/M tokens and $29.1M total cost for the full 15T-token pretrain. GPT-3-175B on a fixed 128×H100 cluster tracked the same curve even more starkly: BF16 MFU rose 34%→54% and FP8 29.5%→39.5% purely from software changes between January and December 2024, cutting FP8 cost per million tokens from 72¢ to 54.2¢ — no hardware change at all. One scale-sensitivity wrinkle worth carrying into any 100k-GPU extrapolation: Llama3-70B's FP8 MFU *degrades* with scale (38.1% at 64 GPUs → 35.5% at 2,048 GPUs) while BF16 stays comparatively flat (54.5%→53.7%) — small-scale FP8 benchmarks should not be extrapolated naively to frontier cluster sizes.

| Reference | Config | MFU | Source date |
|---|---|---|---|
| MegaScale | 175B dense, 12,288 GPU | 55.2% | Feb 2024 |
| Llama-3-405B (as-shipped) | 405B dense, up to 16K H100 | 38–43% BF16 | Jul 2024 |
| Llama-3-405B (re-benchmarked) | same recipe, 576–2,304 H100 | 43% FP8 / 54% BF16 | Aug 2025 |
| GPT-3-175B (software delta) | 128 H100, Jan→Dec 2024 | 34%→54% BF16, 29.5%→39.5% FP8 | Aug 2025 |
| Llama3-70B (scale sensitivity) | 64→2,048 H100 | 38.1%→35.5% FP8; 54.5%→53.7% BF16 | Aug 2025 |
| DeepSeek-V3 (own calc) | 37B active, 2,048 H800 | 17.3% | this chapter |
| DeepSeek-670B (SemiAnalysis) | H100 | 16.6% BF16 | Aug 2025 |
| NVIDIA table, DeepSeek-V3 recipe | GB200, EP64 | 18.0% | accessed Jul 2026 |

Sources per row as cited above.

## GB200's reliability tax on realized economics

Raw MFU and its dollar translation are not the whole story: GB200 NVL72 is measurably *less* reliable per rack than H100, not more. Mean time between interruption runs 1,000–3,000 GPU-days on GB200 versus 2,000–5,000 GPU-days on H100, and the failure blast radius is 72 GPUs (a whole rack drained for a switch-tray or backplane-cartridge swap) against 8 GPUs on an H100/B200 HGX server; backplane cartridge swaps take 8–24 hours, and the Paladin HD copper connectors carry only ~200 mating cycles before replacement (per SemiAnalysis, Aug 2025 — the full failure taxonomy and reliability-adjusted goodput math is Ch21's territory). Priced into a real workload, this materially erodes the headline perf/TCO advantage: SemiAnalysis's own worked example for Llama4-Maverick-400B-MoE-style training found GB200's raw 50%-cheaper-per-token advantage over H100 shrinks to **only ~30% cheaper** once the opportunity cost of 8-of-72 hot-spare GPUs and dedicated spare racks is charged against the run, and their explicit verdict — "the vanilla B200 is better performance-TCO than GB200 NVL72 for pretraining normal-sized models; as model sizes scale or reinforcement learning comes in, GB200 NVL72 will take a lead again" — is a load-bearing caveat for this specific case: a 10T/200B MoE at 100k-GPU scale is exactly the regime where NVL72's reliability tax is worth paying, because it is precisely the scale-up domain that makes expert parallelism tractable at all (Ch18). GB200's all-in capex-per-GPU runs 1.6–1.7x H100's across buyer tiers, and total TCO ~1.6x — meaning GB200 needs at least that much aggregate throughput advantage just to break even before reliability is considered, a threshold SemiAnalysis dates as first crossed around July 2025.

## What Epoch AI's cross-checks add

Epoch AI's public compute-estimation methodology gives useful outside calibration on how conservative or aggressive an assumed MFU should be: their historical compilation spans single-GPU CNNs at 30–75%, GPT-3's original run at just 25%, GSPMD at 62%, Google LaMDA at 56.5%, with a stated default assumption of 0.3 for LLMs when utilization is unreported (Epoch AI, "Estimating training compute of deep learning models"). Their frontier-run FLOP estimates for the current generation — GPT-5 pretrain ~3e25 FLOP, Grok 4 ~5e26 FLOP (their largest confirmed run as of the most recent count), broad 2026 frontier runs in the 1e26–1e27 FLOP range — bracket the strawman's own compute budget (computed below: 3.06–6.12e25 FLOPs) rather than dwarfing or being dwarfed by it: GPT-5's estimated pretrain is comparable to the 25T-token case, while Grok 4 and the broad 2026 frontier band sit roughly 2–16x *above* the strawman — consistent with a 10T/200B design being squarely inside the current frontier envelope, neither an incremental run nor an unprecedented one. Separately, and more structurally relevant to why the plan here is a 100,000-GPU *instantaneous* cluster rather than a smaller cluster run longer: Epoch AI's data-insight on training-run duration finds frontier LLM training time has grown ~1.4x/year since 2020 across 38 frontier LLMs, and argues runs longer than roughly 8.6–9 months become economically suboptimal — missing out on hardware and algorithmic improvements a later, shorter run could capture — projecting the industry hits this ceiling around May 2027 (wide confidence interval, Aug 2025–Sep 2029). Since training-run duration growth has historically contributed roughly a third of total compute-scaling progress since 2018, hitting that ceiling forces future scaling to come disproportionately from simultaneous GPU count rather than run length — a direct structural argument for the 100k-GPU target assumed throughout, independent of any MFU or cost argument. Worth noting for calendar-planning purposes: Epoch AI separately finds that a single final pretraining run's GPU-hour cost systematically *understates* total program cost, since experimentation, ablations, and failed runs consume the majority of total compute budget in practice (Epoch AI's Gradient Updates).

## The money table

**FLOPs per token**, following the Ch7 strawman (N_active = 200B) and folding in the DeepSeek-V3-style single-MTP-module overhead (Ch9), conservatively estimated at +2% for a 60–120-layer model (MTP overhead scales roughly as 1/n_layers, so this shrinks further for a deeper design):

```
FLOPs/token ≈ 6 × N_active × (1 + MTP_overhead)
            ≈ 6 × 200e9 × 1.02
            ≈ 1.224e12 FLOPs/token
```

This omits the attention quadratic term (`≈ 6 × n_layers × d_model × seq_len` per token for causal attention, forward+backward — the `2×2×L×d×s` form sometimes quoted undercounts it by 1.5x), a few percent of the total at 4–8K training sequence lengths but growing materially during any long-context curriculum stage (Ch14) — add it explicitly when modeling that phase.

**Total FLOPs**: 25T tokens → `1.224e12 × 25e12 = 3.06e25 FLOPs`; 50T tokens → `6.12e25 FLOPs`.

**Cluster peak**, 100,000 GB200 GPUs at the reconciled 5,000 TFLOPS/GPU dense-FP8 peak: `100,000 × 5e15 = 5e20 FLOP/s`.

Two MFU bands frame the sensitivity: an "optimistic/dense-analog" band (25/35/45%) matching what NVIDIA's own tables show for *dense* models, and a "MoE-realistic" band (10/15/20%) anchored to the measured MoE MFU above — 9–18% in the 0.1.0-era snapshot, 18–33% on the 26.06 container — kept conservative because 200B/10T = 2% active is sparser than any measured data point. The 10% row deserves a sharper interpretation than "pessimism": Ch18's gradient-sync floor shows that a layout which under-buys tokens-per-replica (as the flagship EP64×PP8×DP192 layout does at a 40M global batch) has its expert-gradient AllReduce exposed rather than overlapped, capping achievable MFU near **10–13% as a calculable structural ceiling** that kernel maturation cannot lift — so the spread between the 10% and 20% rows is largely a *layout design choice* worth $40–85M on the 50T row, not exogenous uncertainty. The table below shows both days-to-token-target and dollar cost, bracketed across a $2.40–$5.00/GPU-hr range spanning a 25th-percentile-neocloud spot price ($2.40) through an illustrative GB300-equivalent-pricing upper bound ($5.00), per SemiAnalysis's "How Much Do GPU Clusters Really Cost?". A dating note on that report: secondhand references describe it as an Aug-2025 pricing snapshot (the origin of the $2.40 figure used here), while the archived PDF itself is dated April 20, 2026, with matching goodput-expense figures (6.14%/10.53%/20.91%, used below). The Apr-2026 primary-source dating is used for citations here, but the conflict is flagged rather than silently resolved, since the $2.40 figure's own within-document date could not be independently re-confirmed.

| MFU | FLOPs/day | Days: 25T | Days: 50T | Cost: 25T ($2.40–$5.00/hr) | Cost: 50T ($2.40–$5.00/hr) |
|---|---|---|---|---|---|
| 45% (dense-analog ceiling) | 1.944e25 | 1.57 | 3.14 | $9.0M–$18.8M | $18.1M–$37.7M |
| 35% (dense-analog) | 1.512e25 | 2.02 | 4.03 | $11.6M–$24.2M | $23.2M–$48.4M |
| 25% (dense-analog, lower) | 1.08e25 | 2.83 | 5.67 | $16.3M–$34.0M | $32.7M–$68.0M |
| 20% (MoE-realistic, upper) | 8.64e24 | 3.54 | 7.08 | $20.4M–$42.5M | $40.8M–$85.0M |
| 15% (MoE-realistic, mid) | 6.48e24 | 4.72 | 9.44 | $27.2M–$56.6M | $54.4M–$113.3M |
| 10% (MoE-realistic, lower/2%-sparsity penalty) | 4.32e24 | 7.08 | 14.17 | $40.8M–$85.0M | $81.6M–$170.0M |

```
daily_$ = 100,000 GPUs × 24 hr × $/GPU-hr    (=$5.76M/day at $2.40/hr, $12.0M/day at $5.00/hr)
cost    = days(MFU, tokens) × daily_$
```

(The table is our own arithmetic, consistent with independently published anchor figures to within rounding.)

A cross-check the table's $2.40 lower bound deserves, because it determines whether that bound is a real economic floor or a promotional artifact (our own cross-check, from Ch16's capex figures): build the same number from ownership economics instead of rental pricing. At Ch16's ~$3.9M all-in per GB200 NVL72 rack, a 100,000-GPU fleet is ~1,389 racks ≈ **$5.4B of capex**; straight-line over a three-year useful life gives ~$4.95M/day of depreciation. Power adds surprisingly little: ~208 MW of IT load at ~150 kW/rack, ~260 MW at a 1.25 PUE, is ~$0.37M/day at $0.06/kWh. The owner's all-in daily cost, ~$5.3M/day, lands within ~8% of the $2.40/GPU-hr rental floor's $5.76M/day — meaning the 25th-percentile neocloud price is very close to bare cost-recovery at three-year depreciation, the table's lower bound is an economic floor rather than a loss-leader, and there is no dramatically cheaper number available to an owner-operator that the rental framing hides. Two corollaries worth stating: energy is only ~7% of the daily cost — this business is silicon amortization, not electricity, so a cheap-power site helps the *feasibility* of the interconnect (Ch16's oscillation envelope) far more than the budget; and the single most cost-sensitive assumption in the entire table is not the $/hr at all but the depreciation horizon — stretch three years to five and the owner's floor drops ~35%, which is exactly the bet embedded in every hyperscaler's accounting-lifetime extension announcements of 2024–2026.

<figure class="vz">
<div class="scroll"><svg style="min-width:620px" viewBox="0 0 748 268" role="img" aria-label="Daily cost of 100,000 GB200 GPUs: owner economics converge within 8 percent of the 25th-percentile rental floor">
<text class="t-ttl" x="0" y="16">100,000 GPUs, per day</text>
<text x="0" y="56">own, 3-yr straight-line</text>
<rect class="build f-acc-22" style="--t:0" x="180" y="42" width="228" height="20" rx="2"/>
<rect class="build f-loud-25" style="--t:0" x="408" y="42" width="17" height="20" rx="2"/>
<text class="t-num build" style="--t:0" x="433" y="56">$5.3M</text>
<text class="t-mut" x="180" y="80">depreciation $4.95M &#183; power $0.37M (260 MW at 1.25 PUE, $0.06/kWh)</text>
<text x="0" y="118">rent at the $2.40/hr floor</text>
<rect class="build f-ink-12" style="--t:1" x="180" y="104" width="265" height="20" rx="2"/>
<text class="t-num build" style="--t:1" x="453" y="118">$5.76M &#8212; within 8% of owning</text>
<text x="0" y="164">rent at $5.00/hr</text>
<rect class="build f-hair s-line" style="--t:2" x="180" y="150" width="552" height="20" rx="2" stroke-width="1"/>
<text class="t-num build" style="--t:2" x="186" y="164">$12.0M &#8212; the illustrative GB300-pricing ceiling</text>
<line class="s-line" x1="180" y1="182" x2="180" y2="188" stroke-width="1"/>
<text class="t-mut" x="174" y="202">0</text>
<line class="s-line" x1="272" y1="182" x2="272" y2="188" stroke-width="1"/>
<text class="t-mut" x="266" y="202">2</text>
<line class="s-line" x1="364" y1="182" x2="364" y2="188" stroke-width="1"/>
<text class="t-mut" x="358" y="202">4</text>
<line class="s-line" x1="456" y1="182" x2="456" y2="188" stroke-width="1"/>
<text class="t-mut" x="450" y="202">6</text>
<line class="s-line" x1="548" y1="182" x2="548" y2="188" stroke-width="1"/>
<text class="t-mut" x="542" y="202">8</text>
<line class="s-line" x1="640" y1="182" x2="640" y2="188" stroke-width="1"/>
<text class="t-mut" x="634" y="202">10</text>
<line class="s-line" x1="732" y1="182" x2="732" y2="188" stroke-width="1"/>
<text class="t-mut" x="726" y="202">12</text>
<text class="t-mut" x="404" y="220">$M per day</text>
<text class="t-mut" x="0" y="244">energy is ~7% of the owner&#8217;s day &#8212; the business is silicon amortization, not electricity;</text>
<text class="t-mut" x="0" y="258">a 5-year depreciation bet drops the floor ~35%</text>
</svg></div>
<p class="vz-cap">The cross-check that makes the cost table&#8217;s lower bound trustworthy: build the owner&#8217;s number from Ch16&#8217;s capex instead of rental pricing. ~1,389 racks at ~$3.9M all-in is ~$5.4B; straight-line over three years is $4.95M/day, plus $0.37M of power — landing within 8% of what the 25th-percentile neocloud charges per day. The $2.40/hr floor is bare cost-recovery at three-year depreciation, not a loss-leader, and no dramatically cheaper number is available to an owner-operator. The most sensitive assumption in the whole money model is the depreciation horizon, not the hourly rate.</p>
</figure>

**Calendar-vs-compute adjustment.** These are pure-compute floors; the calendar schedule is longer once goodput is accounted for. Two independent goodput analyses give complementary views. SemiAnalysis's ClusterMAX "Goodput Expense" framework (based on a 5,184×GB300 NVL72 large-pretrain scenario — not identical scale/hardware to the 100k-GB200 target here, but the best available quantified real-world figure) gives Gold-tier 6.14%, Hyperscaler-tier 10.53%, Silver-tier 20.91% goodput expense — equivalently, multipliers of 1.065x, 1.118x, and 1.264x on pure-compute days:

| Tier | Goodput expense | Calendar multiplier | 15%-MFU/50T case (9.44 pure-compute days) |
|---|---|---|---|
| Gold | 6.14% | 1.065x | ≈10.1 days |
| Hyperscaler | 10.53% | 1.118x | ≈10.6 days |
| Silver | 20.91% | 1.264x | ≈11.9 days |

Ch21's independent, more granular reliability analysis — built from MTBI field data, a 100K-GPU HSDP production paper, and Google's own Gemini goodput disclosures — targets **85–93% goodput** for a well-engineered 100k-GB200, 10T-MoE run specifically (equivalently, 1.08x–1.18x calendar multipliers), which brackets the ClusterMAX Gold/Hyperscaler figures closely and treats Silver-tier's 79%-effective outcome as below the bar a serious 100k-GPU operator should accept. The two methodologies converge: at realistic goodput, expect roughly **8–20% calendar overhead** on top of the pure-compute floor, not the 2–3x sometimes assumed from early GB200-generation horror stories.

## Implications for the 10T/200B run

The pure-FLOPs floor at 100k GB200 GPUs is genuinely not the binding constraint: even at the comm-bound 10% floor, 50T tokens costs 14.2 pure-compute days and $82–170M; at the 15–20% a sound layout on current software should deliver (Ch17–Ch19), that falls to 7–9.4 days and $41–113M. Three numbers should anchor any budget conversation. First, the MFU band to plan against is **15–25% on 26.06-era software** — not the 32–53% dense figures a naive reading of vendor tables would suggest, and no longer the 10–20% the first Blackwell container generation implied — with roughly 10–13% reinterpreted as the *diagnosable signature of a comm-bound layout* (Ch18's gradient-sync floor), a design failure with a calculable cause, not hardware fate; a 2%-active design should still be modeled toward the low end of the band until EP all-to-all bandwidth (Ch18, Ch20) is proven out at 100k-GPU scale. Second, software maturation is worth a full MFU tier over the run's lifetime — the GB200 generation got 45% faster (a 31% wall-clock cut) in six months on identical hardware, DeepSeek-V3's GB200 recipe gained +43% relative MFU in one container generation, and DeepSeek-670B's GB200 economics improved from marginally-worse-than-H100 to a 2.5x-and-climbing advantage in the same window — so the de-risking ladder (Ch9) should assume day-one MFU numbers are a floor, not a target. Third, the reliability tax is not a rounding error: GB200's larger blast radius and lower MTBI relative to H100 erode a naive cost advantage to only ~30% in SemiAnalysis's own worked example, and that erosion is the direct reason Ch21's checkpoint and fault-tolerance architecture is not optional infrastructure but a load-bearing part of the money math itself. A defensible number for this specific run — the one to bring to both the CFO and the chief scientist — is: **6.5–10 calendar days and $41–113M in pure GPU-hour cost for a 50T-token run at a 15–20% blended MFU with Gold-tier goodput**, with the explicit caveat that this is the *training-compute* line item alone, not the total program cost, which Epoch AI's own findings (final runs are 9.6–22.6% of lab R&D compute, i.e., a 3.4–9.4x experimentation multiplier) suggest the full program will multiply several-fold.
