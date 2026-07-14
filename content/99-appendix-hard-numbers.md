# Appendix A: Master Hard-Numbers Table

Compiled from the "Hard numbers" sections of all 42 research-note files in `pretraining-2026-research/notes/` (01–53). Each row carries: the value, its context, the citing source, a provenance tag (how confident/primary the sourcing is), and a date. Where the same fact appears in multiple notes, the best-sourced instance is kept and duplicates are dropped silently. Where notes genuinely disagree — different numbers for what should be the same quantity — both rows are kept and marked **[CONFLICT]**.

**Silicon note:** `notes/30-gb200-nvl72-peak-flops-reconciliation.md` is treated as authoritative for all GB200 NVL72 / HGX B200 per-GPU peak-FLOPS figures; it reconciles the NVIDIA Blackwell datasheet against every other note's derived numbers and finds them consistent to rounding except where flagged. Downstream MFU calculations elsewhere in the corpus that used a slightly-off denominator (e.g., 4,500 vs. the correct 5,000 TFLOPS/GPU dense FP8 for GB200 NVL72) are noted where material.

**Provenance tags:** PAPER (arXiv/peer-reviewed), OFFICIAL (vendor/lab blog, docs, product page, statement), CODE (repo/README/benchmark script), BENCHMARK (independent measured benchmark), DERIVED (calculated by a note's author from other primary numbers), HEARSAY (secondary press/aggregation/paywalled analysis), SPOKEN (podcast/interview claim, unverified), LOGBOOK (primary training chronicle), THESIS, UNVERIFIED (explicitly flagged unconfirmed by the sourcing note).

---

## A.1 Silicon & Systems

### A.1.1 NVIDIA Hopper / pre-Blackwell

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 989–990 TFLOPS/GPU BF16, 1,979–2,000 TFLOPS/GPU FP8 dense | H100 SXM5 peak Tensor Core | scaling-book gpus.md; multiple NVIDIA-spec cross-refs | OFFICIAL | 2025-08 |
| 700 W | H100 SXM5 TDP | SemiAnalysis PDF; Stas repo | OFFICIAL | 2025-08 |
| 794.5/989 = 80.3% BF16 | H100 SXM measured-achievable matmul FLOPS (MAMF, `mamf-finder.py`) | Stas `compute/accelerator/README.md` | BENCHMARK | 2026-07 |
| 271.2/312 = 86.9% BF16 | A100 SXM MAMF | Stas repo | BENCHMARK | 2026-07 |
| 19.5 → 156 TFLOPS (8×) | A100 FP32 → TF32 via `allow_tf32` flag | Stas `training/dtype.md` | CODE | 2026-07 |
| 700 W | B100 max TDP (lowest-power Blackwell SKU, drop-in H100 replacement, excluded from most benchmarking) | glennklockwood.com; JAX scaling book | OFFICIAL | 2026-04 |
| 22 µs | Meta CTSW (VOQ) fabric unloaded RTT, H100-era RoCE backend | SIGCOMM'24 Gangidi et al. | PAPER | 2024-08 |

### A.1.2 NVIDIA Blackwell (B200 / GB200 NVL72) — authoritative table

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| Dense FP8/FP6 **5,000** TFLOPS/GPU; sparse (2:4) 10,000 | GB200 NVL72 per-GPU Tensor Core | NVIDIA Blackwell datasheet #3384703, cross-validated live gb200-nvl72 page | OFFICIAL | 2024-12 / 2026-07 |
| Dense BF16/FP16 2,500 TFLOPS/GPU; sparse 5,000 | GB200 NVL72 | same | OFFICIAL | 2024-12 |
| Dense NVFP4 10,000 TFLOPS/GPU; sparse 20,000 | GB200 NVL72 | same | OFFICIAL | 2024-12 |
| Dense TF32 1,250 TFLOPS/GPU; sparse 2,500 | GB200 NVL72 | same | OFFICIAL | 2024-12 |
| 80 TFLOPS/GPU | GB200 NVL72 FP32 (vector/CUDA-core, no sparse variant) | same | OFFICIAL | 2024-12 |
| 40 TFLOPS/GPU (Dec-2024 NVIDIA spec) vs. 45 TFLOPS/GPU (Apr-2026 3rd-party page) | GB200 NVL72 FP64/FP64-TC | NVIDIA datasheet vs. glennklockwood.com | OFFICIAL vs HEARSAY | **[CONFLICT]** 2024-12 / 2026-04 |
| 186 GB HBM3e @ 8.0 TB/s per GPU (372 GB/16 TB/s per superchip) | GB200 NVL72 memory | NVIDIA datasheet + live product page | OFFICIAL | 2024-12 / 2026-07 |
| 1,200 W/GPU (liquid-cooled, configurable) | GB200 NVL72 max TDP | NVIDIA datasheet; SemiAnalysis independently confirms | OFFICIAL | 2024-12 / 2025-08 |
| Dense FP8/FP6 4,500 TFLOPS/GPU; sparse 9,000 | HGX B200 (air-cooled) | NVIDIA datasheet | OFFICIAL | 2024-12 |
| Dense BF16/FP16 2,250 TFLOPS/GPU; sparse 4,500 | HGX B200 | same | OFFICIAL | 2024-12 |
| Dense NVFP4 9,000 TFLOPS/GPU; sparse 18,000 | HGX B200 | same | OFFICIAL | 2024-12 |
| 75 TFLOPS/GPU FP32; 37 TFLOPS/GPU FP64 | HGX B200 | same | OFFICIAL | 2024-12 |
| 180 GB HBM3e @ 7.7 TB/s | HGX B200 memory | same | OFFICIAL | 2024-12 |
| 1,000 W/GPU (air-cooled, configurable) | HGX B200 max TDP | same | OFFICIAL | 2024-12 |
| 360 PFLOPS (72-GPU rack, dense FP8) | GB200 NVL72 rack-level = 5,000×72 | derived from datasheet | DERIVED | 2024-12 |
| 1,822/2,500 = 72.9% BF16 MAMF | GB200 SXM measured-achievable matmul FLOPS | Stas repo (torch 2.10.0.dev, cu130) | BENCHMARK | 2026-07 |
| 3,615.6/5,000 = 72.3% FP8 MAMF | GB200 SXM | Stas repo | BENCHMARK | 2026-07 |
| 1,745/2,250 = 77.6% BF16 MAMF | B200 SXM | Stas repo (torch 2.7.1+cu128) | BENCHMARK | 2026-07 |
| 1,769/2,250 = 78.6% BF16 MAMF | B300 SXM | Stas repo (torch 2.9.1+cu130) | BENCHMARK | 2026-07 |
| 288 GiB HBM3e @ 8.00 TB/s | B300 SXM (vendor-advertised, usable-capacity-adjusted) | Stas repo | OFFICIAL | 2026-07 |
| 80% unidirectional / 38% duplex efficiency of 450/900 GB/s spec | GB200 NVLink-C2C measured (`nvbandwidth`, DGX Station, half GB300 module) | Stas repo | BENCHMARK | 2026-07 |
| 96.3%/96.0%/96.2%/96.3%/98.2% of peak | B200 measured tensor-core efficiency at realistic tile shape, FP8/FP6/FP4/BF16/INT8 | arXiv:2512.02189v2 | BENCHMARK | 2025-12 |
| 3,850.6/5,134.4/7,700.2/1,926.4 TFLOPS; 3,928.5 TOPS | B200 measured (not peak-spec) at m64n8k16 tile, FP8/FP6/FP4/BF16/INT8 | arXiv:2512.02189v2 | BENCHMARK | 2025-12 |
| Implied theoretical peak ≈4,000 TFLOPS FP8 (from 3,850.6/96.3%) | Below even HGX B200's 4,500 dense spec — internal inconsistency flagged in the paper's own baseline | arXiv:2512.02189v2, cross-checked by notes/30 | **[CONFLICT — likely erroneous 3rd-party benchmark baseline]** | 2025-12 |
| ~1.27× | B200 vs. H200 speedup, consistent across precisions at raw tensor-core level | arXiv:2512.02189v2 | BENCHMARK | 2025-12 |
| 148 total SMs (74/die active of 80 physical, 6 disabled/die) | B200 two-die chiplet | Chips and Cheese teardown | HEARSAY | 2025 |
| 21 TB/s same-partition vs. 16.8 TB/s cross-partition L2 (126 MB total) | B200 L2 cache | Chips and Cheese teardown | HEARSAY | 2025 |
| 16 TB/s read / 8 TB/s write per SM; 420-cycle miss latency | B200 TMEM | arXiv:2512.02189v2 | BENCHMARK | 2025-12 |
| 7.48 TB/s (93.5% of theoretical) vs. 3.8 TB/s plateau | B200 HBM3e bandwidth, STREAM-Triad-optimized vs. conventional access path | arXiv:2512.02189v2 | BENCHMARK | 2025-12 |
| 190.49 TFLOPS | CUTLASS FP8 GEMM, H800, realistic small-M MoE shape (M=128,N=4096,K=7168) — vs. ~1,550 TFLOPS peak-shape number | vllm-project PR #13917 | CODE | 2025 |
| $2.69/GPU-hr (GB300 NVL72) → $4.18/GPU-hr (Vera Rubin NVL72) | Total-system-cost basis | SemiAnalysis | HEARSAY | 2026-05 |
| 2,500→4,000 BF16 / 5,000→17,500 FP8 / 15,000→35,000 FP4 dense TFLOPS | GB300 NVL72 → Vera Rubin NVL72 marketed per-GPU compute | SemiAnalysis | HEARSAY | 2026-05 |
| 8→22 TB/s per-logical-GPU | GB300 NVL72 → VR NVL72 memory bandwidth | SemiAnalysis | HEARSAY | 2026-05 |

### A.1.3 NVIDIA Rubin (pre-launch specs)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 4,000 BF16 / 17,500 FP8 / 35,000 FP4-NVFP4 TFLOPS dense | Rubin SXM pre-launch spec | Stas repo | OFFICIAL | 2026-07 |
| 35 PFLOPS dense FP4 / 50 PFLOPS "compressed" | Rubin GPU | SemiAnalysis | HEARSAY | 2026-02 |
| 22 TB/s target / ~20 TB/s realistic | Rubin HBM4 bandwidth | SemiAnalysis | HEARSAY | 2026-02 |
| 2,300 W (Max-P) / 1,800 W (Max-Q) | Rubin GPU TDP | SemiAnalysis | HEARSAY | 2026-02 |
| 336B transistors | Rubin GPU | SemiAnalysis | HEARSAY | 2026-02 |
| 100 PFLOPS dense FP4, 1,024 GB HBM (16 stacks), 365 TB total fast memory | Rubin Ultra system-level | SemiAnalysis, GTC2025 coverage | HEARSAY | 2025-03 |
| 14.4 Tbit/s uni-directional/GPU | Rubin Ultra Kyber scale-up bandwidth | SemiAnalysis | HEARSAY | 2026-03 |
| 180–220 kW/rack (roadmap to 1 MW) | Vera Rubin NVL72 rack TDP | SemiAnalysis | HEARSAY | 2026-02 |
| 60 exaflops, 10 PB/s, 1.2 quadrillion transistors, ~20,000 dies, 1,100+ GPUs, 40 racks | Per "Vera Rubin pod," Jensen Huang claim | Lex Fridman #494 | SPOKEN | 2026-03 |
| ~200 pods/week manufacturing target | Vera Rubin | Jensen Huang, Lex Fridman #494 | SPOKEN | 2026-03 |
| 1.3M components, 200 suppliers/rack | Rack supply-chain complexity | Jensen Huang, Lex Fridman #494 | SPOKEN | 2026-03 |
| "Blackwell = 50× Hopper" | Stated generational efficiency jump | Jensen Huang, Dwarkesh | SPOKEN | 2026-04 |

### A.1.4 Google TPU

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| BF16 46→123→275→459→2,307 TFLOPS/chip | v2→v3→v4→v5p→Ironwood generational peak | arXiv:2606.15870 Table 1 | PAPER | 2026-06 |
| FP8 peak: n/a→n/a→n/a→459→4,614 TFLOPS | v5p first with FP8 at BF16 parity; Ironwood FP8 | arXiv:2606.15870 | PAPER | 2026-06 |
| HBM 16→32→32→96→192 GiB/chip | v2→v3→v4→v5p→Ironwood | arXiv:2606.15870 | PAPER | 2026-06 |
| HBM BW 700→900→1,200→2,765→7,300 GB/s | v2→v3→v4→v5p→Ironwood | arXiv:2606.15870 | PAPER | 2026-06 |
| Pod size 256→1,024→4,096→8,960→9,216 chips | v2→v3→v4→v5p→Ironwood | arXiv:2606.15870 | PAPER | 2026-06 |
| Pod peak BF16 0.01→0.13→1.1→4.1→21.3 ExaFLOPS | v2→v3→v4→v5p→Ironwood | arXiv:2606.15870 | PAPER | 2026-06 |
| Pod peak FP8 4.1→42.5 ExaFLOPS | v5p→Ironwood | arXiv:2606.15870 | PAPER | 2026-06 |
| Pod HBM 4TB→33TB→131TB→851TB→1.77PB | v2→v3→v4→v5p→Ironwood aggregate | arXiv:2606.15870 | PAPER | 2026-06 |
| Relative TFLOPS/W 1→1.8→4.9→5.2→29.3 | v2→v3→v4→v5p→Ironwood, normalized | arXiv:2606.15870 Fig.5 | PAPER | 2026-06 |
| 4,614 FP8 TFLOPS, 192 GB, 7.37 TB/s | Ironwood, independent press confirmation | Google/SemiAnalysis | HEARSAY | 2025-11 |
| 9,216 chips, 42.5 EFLOPs max superpod | Ironwood | Google/press | OFFICIAL | 2025-11 |
| 4 chips/host, 64 chips/cube (4×4×4), 144 cubes/pod, 2,304 CPU hosts/pod | Ironwood physical building blocks | arXiv:2606.15870 | PAPER | 2026-06 |
| 64 MiB/TensorCore VMEM (2 cores/chip) | Ironwood | arXiv:2606.15870 + Google Cloud TPU docs | PAPER | 2026 |
| ~11,500 flops/byte | Ironwood hardware arithmetic intensity, drives TP-viability threshold | Google Cloud TPU docs | OFFICIAL | 2026 |
| 1.3× end-to-end speedup | Ironwood FP8 vs. BF16 (not the naive 2× theoretical MXU ratio) | Google Cloud TPU docs | OFFICIAL | 2026 |
| 9,216 chips (ICI) / 147,456 chips (DCN) | TPUv7 max scale-up / max data-center-network world size | SemiAnalysis | HEARSAY | 2025-11 |
| 9,600 chips/pod, 2 PB pod HBM, 3× Ironwood perf, 2× perf/W | "TPU 8t" Gen-8 training TPU | Google Cloud Next | OFFICIAL | 2026-04-22 |
| 12.6 PFLOPs FP4/chip, 216 GB HBM, 6,528 GB/s | TPU 8t chip specs | Google Cloud blog | OFFICIAL | 2026-04 |
| 10.1 PFLOPs FP4/chip, 288 GB HBM, 8,601 GB/s | TPU 8i (Gen-8 inference) chip specs | Google Cloud blog | OFFICIAL | 2026-04 |
| 1,152 chips/pod, 3× on-chip SRAM | TPU 8i pod | Google Cloud Next | OFFICIAL | 2026-04-22 |
| 134,000 chips/site, 47 Pb/s bisectional BW | Google Virgo Network single-site fabric | Google Cloud blog | OFFICIAL | 2026-04 |
| >1,000,000 TPU chips claimed single training cluster (Virgo + JAX/Pathways) | Multi-site scaling claim | Google Cloud blog | OFFICIAL | 2026-04 |
| 100,000 chips / 13 Pb/s bisection | TPU v6e Trillium Jupiter fabric | Google Cloud Blog | OFFICIAL | 2024 |
| 91 exaFLOPS | Single TPU v6e cluster peak (4× largest v5p cluster) | Google Cloud Blog | OFFICIAL | 2024 |
| 9,216 chips/pod, 1,200 GB/s/chip bidirectional ICI | TPU7x (Ironwood) pod ceiling | docs.cloud.google.com/tpu/tpu7x | OFFICIAL | GA 2025-11 |
| 8,960 chips/pod, 4,800 Gbps/chip ICI | TPU v5p pod ceiling | Google Cloud Docs | OFFICIAL | — |
| 4,096 chips/SuperPod | Gemini TPUv4 SuperPod unit | Gemini 1.5 Technical Report | PAPER | 2024-03 |
| ~10 seconds | OCS reconfiguration time, 4×4×4 cube → arbitrary 3D torus | Gemini 1.5 report | PAPER | 2024-03 |

### A.1.5 AWS Trainium

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 96 GiB HBM3 @ 2.9 TB/s; 1,299 FP8 / 667 BF16 / 181 FP32 TFLOPS | Trainium2 per-chip | AWS Neuron docs | OFFICIAL | 2026 |
| NeuronLink-v3 1.28 TB/s/chip | Trainium2 interconnect | AWS Neuron docs | OFFICIAL | 2026 |
| 64-chip UltraServer, 4×4×4 3D torus, 83.2 FP8 PFLOPS, 6 TB HBM, 185 TB/s | Trn2 UltraServer | AWS product docs | OFFICIAL | 2026 |
| Z-axis 64 GB/s vs. X/Y 128 GB/s | Trn2 torus asymmetry | SemiAnalysis | HEARSAY | 2025 |
| 667 vs. 2,500 TFLOP/s FP16 (3.85×); 2,900 vs. 8,000 GB/s mem BW (2.75×); 186 vs. 576 TB/s rack-aggregate (3.1×) | Trainium2 vs. GB200 head-to-head | SemiAnalysis | HEARSAY | 2025-09 |
| 144 GiB HBM3e @ 4.9 TB/s | Trainium3 per-chip | AWS Neuron docs | OFFICIAL | 2025-12 |
| NeuronLink-v4 2.56 TB/s/chip | Trainium3 interconnect | AWS Neuron docs | OFFICIAL | 2025-12 |
| 2,517–2,520 MXFP8/MXFP4 TFLOPS/chip | Trainium3 compute | AWS Neuron docs / AWS re:Invent | OFFICIAL | 2025-12 |
| 144 chips/UltraServer, 362 PFLOPS | Trainium3 UltraServer | AWS re:Invent | OFFICIAL | 2025-12 |
| 40% more energy-efficient | Trainium3 vs. Trainium2 | AWS re:Invent | OFFICIAL | 2025-12 |
| ~705.6 TB/s aggregate fabric BW | Trainium3 NL72×2 topology | SemiAnalysis | HEARSAY | 2026-Q1 |
| ~30% better TCO/marketed FP8 PFLOPS vs. GB300 NVL72 (worse on FP4) | Trainium3 UltraServer | SemiAnalysis | HEARSAY | 2026 |
| 500 W → 1,000 W | Trn2 → Trainium3 chip TDP | SemiAnalysis | HEARSAY | 2025-12 |
| 43% BF16 MFU (dense) / 20–30% (MoE) | Trainium3 Day-0 PyTorch-native/torch.compile | SemiAnalysis | HEARSAY | 2025-12 |
| ~60% BF16 MFU (dense) / ~40% (DeepSeek-670B-class MoE) | Trainium3, hand-written NKI kernels | SemiAnalysis | HEARSAY | 2025-12 |
| ≈2 TB/s @ ≈1 µs/hop (AWS exec claim) vs. 1.28 TB/s/chip (AWS Neuron docs) | NeuronLink bandwidth, conflicting AWS-internal figures | SDxCentral interview vs. AWS Neuron docs | **[CONFLICT]** | 2025-11 / 2026 |

### A.1.6 AMD Instinct

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 668.4/1,300 = 51.4% BF16 MAMF | MI300X | Stas repo (ROCm 6.3.42131) | BENCHMARK | 2026-07 |
| 784.9/1,300 = 60.4% BF16 MAMF | MI325X | Stas repo (ROCm 6.2.4) | BENCHMARK | 2026-07 |
| 1,565/2,300 = 68.0% BF16 MAMF | MI355X | Stas repo (ROCm 7.0.2) | BENCHMARK | 2026-07 |
| 80.5 PFLOPS FP8/8-GPU platform, 1,400 W/GPU | MI355X | AMD official | OFFICIAL | 2025 |
| 432 GB HBM4, 19.6 TB/s, 40 PFLOPS FP4/20 PFLOPS FP8, 300 GB/s fabric | MI400 | AMD official | OFFICIAL | 2025-2026 |
| 20,000 FP8 / 40,000 FP4 TFLOPS | MI450X pre-launch spec | Stas repo | OFFICIAL | 2026-07 |
| 72× MI400 rack, ~3.1 ExaFLOPS, 31 TB HBM4 | AMD "Helios" rack-scale | AMD official | OFFICIAL | 2025-2026 |
| 4.330 TB/s vs. 3.036 TB/s | MI300X vs. H100 HBM bandwidth saturation, measured | arXiv:2511.17127 | PAPER | 2025-12 |
| ~200 GFLOPs | GEMM problem size needed to reach peak MI300X TFLOPS | arXiv:2511.17127 | PAPER | 2025-12 |
| ~720 vs. ~620 TFLOP/s | Achieved BF16 GEMM, H100/H200 vs. MI300X | SemiAnalysis benchmark | BENCHMARK | 2024-12 |
| <20 TFLOP/s | MI300X public-release Flash Attention bug, persisted for months | SemiAnalysis | HEARSAY | 2024-12 |

### A.1.7 Chinese silicon & non-NVIDIA misc.

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 128 GB HBM2e @ 6.4 TB/s | Ascend 910C per-chip | Multiple teardown/analysis sources | HEARSAY | 2025 |
| ~2.9M TSMC-fab'd dies accumulated (export-control violation) | Huawei Ascend die bank | SemiAnalysis | HEARSAY | 2025-2026 |
| SMIC yield 20–40% vs. TSMC 60%+ | Ascend post-TSMC-bank production | SemiAnalysis | HEARSAY | 2025-2026 |
| ~13M HBM stacks stockpiled (China total) | Huawei Ascend feedstock | SemiAnalysis | HEARSAY | 2025-09 |
| ~2M CXMT HBM stacks/yr → ~250–300K (or up to ~5M) chip-equivalent | China domestic HBM ceiling on Ascend output (range across two notes) | SemiAnalysis | HEARSAY | 2025-2026 |
| 600,000 Ascend 910C units targeted 2026 | Huawei production target | Industry reporting | HEARSAY | 2025-2026 |
| ~300 PFLOPS dense BF16, 48 TB HBM (384× Ascend 910C) | CloudMatrix 384 supernode | SemiAnalysis | HEARSAY | 2025 |
| 4.1× power, 2.5× worse power/FLOP vs. GB200 NVL72 | CloudMatrix 384 | SemiAnalysis | HEARSAY | 2025 |
| 6,912 optical 400G LPO transceivers | CloudMatrix 384 | qsfptek.com | HEARSAY | 2025-2026 |
| 68.2% of all component failures attributed to optics (10K-scale cluster, 1yr) | Cluster reliability | qsfptek.com | HEARSAY | 2025-2026 |
| 520,000+ chips, 10,000+ racks, 524 EFLOPS FP8, 1,152 TB memory, 16.3 PB/s interconnect | Ascend 950PR/950DT + Atlas 950 SuperCluster roadmap | Huawei Connect/Convequity tracking | HEARSAY | 2025-2026 |
| 30 EFLOPS FP8/60 EFLOPS FP4 SuperNode (4Q26); 2 ZFLOPS FP8 SuperCluster (4Q27) | Ascend 960 future roadmap | Convequity tracking | HEARSAY | 2025-2026 |
| 3/1.5/0.8 POPS at 6-bit/9-bit/BF16 | Microsoft Maia 100 peak dense tensor throughput | Hot Chips 2024 slides | OFFICIAL | 2024-08 |
| 1.8 TB/s, 64 GB HBM2E | Maia 100 | Hot Chips 2024 | OFFICIAL | 2024-08 |
| 216 GB HBM3e @ 7 TB/s, 272 MB SRAM, TSMC 3nm | Microsoft Maia 200 (inference chip) | Microsoft official | OFFICIAL | 2026-01 |
| 4 exaFLOPs, 54M cores, 64 nodes | Cerebras CG-1 | LLM360 paper | PAPER | 2023-12 |
| 64× CS-3, 8 ExaFLOPS/site, 16 ExaFLOPS network (3 sites w/ G42) | Cerebras Condor Galaxy fleet | Cerebras official | OFFICIAL | 2025-2026 |
| ~$20B Nvidia IP-license + acqui-hire | Groq market exit | Industry reporting | HEARSAY | 2025-12-24 |

---

## A.2 Networks & Fabrics

### A.2.1 Scale-up interconnect (NVLink / NeuronLink / ICI)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 900 GB/s unidirectional / 1,800 GB/s duplex per GPU | NVLink5, B200 and GB200 alike | scaling-book + NVIDIA + Stas repo | OFFICIAL | 2025-08 |
| 8 GPUs (HGX B200) vs. 72 GPUs (GB200 NVL72) | NVLink domain size | scaling-book gpus.md | OFFICIAL | 2025-08 |
| ~130 TB/s aggregate NVL72 domain bandwidth | GB200 NVL72 | NVIDIA docs; cluster-architecture notes | OFFICIAL | 2025 |
| 400 GB/s (H100) vs. 3,600 GB/s (GB200 NVL72) | Node/domain egress to scale-out fabric | scaling-book gpus.md | OFFICIAL | 2025-08 |
| ~50 GB/s/GPU (CX-7, 400 Gbps baseline) vs. ~100 GB/s/GPU (ConnectX-8, 800 Gbps XDR) | Scale-out per-GPU bandwidth, 2025 vs. 2025–26 GB200 racks | scaling-book + industry reporting | OFFICIAL/HEARSAY | 2025 / 2025-2026 |
| 5,184 copper cables | GB200 NVL72 rack backplane | Vendor technical blogs | HEARSAY | 2025 |
| 1,800 GB/s unidirectional / 3,600 duplex, 36 links | NVLink6 (Rubin) | Stas repo | OFFICIAL | 2026-07 |
| NVLink-over-PCIe ratio: Hopper ~7×, Blackwell ~7×, Rubin ~15× | Cross-gen scale-up:scale-out ratio | Stas repo | DERIVED | 2026-07 |
| NV12, 2 NUMA domains × 24 cores | 8× A100 node topology | Stas repo (`nvidia-smi topo -m`) | CODE | 2026-07 |
| NV18, 2 NUMA domains × 52 cores | 8× H100 node topology | Stas repo | CODE | 2026-07 |
| NeuronLink-v3 1.28 TB/s/chip; NeuronLink-v4 2.56 TB/s/chip | Trainium2 / Trainium3 | AWS Neuron docs | OFFICIAL | 2026 |

### A.2.2 Collective communication libraries & measured comms

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| ~370 GB/s realized of 450 GB/s theoretical (SHARP disabled) | H100 intra-node AllReduce, empirical | scaling-book (measured) | BENCHMARK | 2025-08 |
| ~30% actual gain of 75–100% theoretical | NVIDIA SHARP in-network reduction, empirical | scaling-book (NCCL 2.27.5) | BENCHMARK | 2025-08 |
| +30% intra-node / +25% inter-node all-reduce | NVLS/SHARP gain | GTC25 session S72583, corroborated by Stas's own benchmark | OFFICIAL/BENCHMARK | 2026-07 |
| 480 GB/s busbw vs. 450 GB/s theoretical | H100 NVLink4 all-reduce (`all_reduce_bench.py`, 4 GiB payload) | Stas repo | BENCHMARK | 2026-07 |
| 47.671 GB/s (2-GPU) vs. 312.912 GB/s (8-GPU), 6.5× not 4× | MI300X busbw scaling, 4 GiB payload | Stas repo | BENCHMARK | 2026-07 |
| Up to 9× | NCCL 2.27 symmetric-memory latency reduction, small messages | NVIDIA official blog | OFFICIAL | 2025 |
| 16→6 SMs (2.7×) reduction | NCCL 2.27 SHARP AllGather/ReduceScatter SM usage | NVIDIA official blog | OFFICIAL | 2025 |
| ~1–2% gap | NCCL 2.28 GIN vs. NVSHMEM device-initiated performance | arXiv:2606.05951 | PAPER | 2026-06 |
| Up to 12% per-step latency reduction (Llama4); up to 11× cluster-init speedup @96K GPU; 15–80% inference decode-latency gain | NCCLX (Meta) | arXiv:2510.20171 | PAPER | 2026-01 |
| 7× / 15× / 30× | NCCLX latency multiplier: same-rack → cross-AI-Zone → cross-DC-building | arXiv:2510.20171 | PAPER | 2026-01 |
| Up to 3.8× / 15% | MSCCL++ speedup vs. NCCL (collectives / real inference workloads) | arXiv:2504.09014 | PAPER | 2026-03 |
| 200 Gbps, 120 Mops/s | Google Falcon first hardware implementation | Google SIGCOMM'25 | PAPER | 2025 |
| Up to 8× lower completion time, 65% higher goodput vs. CX-7 RoCE under congestion | Falcon vs. standard RoCE | Google SIGCOMM'25 | PAPER | 2025 |
| ~3 µs one-way latency; 5–6% die / 3–4% power HW cost | Falcon NIC integration | Google | PAPER | 2025 |
| busbw curve: 0.92 GBps@32KiB → 98.34 GBps@32MiB (peak) → 84.90 GBps@64MiB (dip) → ~91–94 GBps plateau | `all_reduce_bench.py` sample, 4-node run | Stas repo | BENCHMARK | 2026-07 |
| `all_gather_object`→`all_reduce` for status sync: 23× speedup | Micro-benchmark | Stas repo | BENCHMARK | 2026-07 |

### A.2.3 Switch/fabric vendor hardware

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 144 ports × 800 Gb/s, 14.4 TFLOPS in-network SHARP, <100 ns port-to-port, 10,368-NIC 2-tier fat-tree ceiling | NVIDIA Quantum-X800 XDR switch | NVIDIA official | OFFICIAL | 2025 |
| 102.4 Tbps, 512-XPU single-hop reach | Broadcom Tomahawk 6 "Davisson" | Broadcom official | OFFICIAL | 2025-10 (shipping 2026-03) |
| 250 ns latency | Broadcom Tomahawk Ultra | Broadcom official | OFFICIAL | 2025-2026 |
| 51.2 Tbps, 36,000 HyperPorts, 100km+ lossless-RoCE reach, 160× HBM buffering vs. on-chip SRAM | Broadcom Jericho4 | Broadcom official | OFFICIAL | 2025-08 |
| 51.2 Tbps single-chip switch capacity | Used in Alibaba HPN 7.0 / NVIDIA Colossus | SIGCOMM'24; NVIDIA | PAPER/OFFICIAL | 2024-08 / 2025 |
| 1M link-hours, 0 flaps; 2.5M-hr MTBF | Broadcom TH5-Bailly CPO, Meta-tested | Broadcom (vendor-reported) | OFFICIAL | 2025-10 |
| ~70% / >3.5× optical-power reduction vs. pluggables | Broadcom TH6-Davisson | Broadcom official | OFFICIAL | 2025-10 |
| 2.6M device-hr (CPO) vs. 0.5–1M device-hr (2×FR4 pluggable) | CPO vs. transceiver MTBF | SemiAnalysis, citing Meta/Broadcom ECOC 2025 | HEARSAY | 2026-01 |
| 2–4% / 3–7% | Total cluster power / cost savings from full CPO scale-out adoption | SemiAnalysis | HEARSAY | 2026-01 |

### A.2.4 Production cluster fabrics

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 24,000 GPUs | Meta production RoCE backend cluster size | SIGCOMM'24 Gangidi et al. | PAPER | 2024-08 |
| 4K→24K→129K GPUs in <2 years | Meta RoCE fabric scaling timeline | @Scale Networking 2025 recap | OFFICIAL | 2025-09 |
| +40% AllReduce gain, QP=16 round-robin vs. QP=1 | Meta RoCE, E-ECMP baseline | SIGCOMM'24 | PAPER | 2024-08 |
| 15,360 GPUs/Pod, 2-tier dual-plane | Alibaba HPN 7.0 | SIGCOMM'24 Qian et al. | PAPER | 2024-08 |
| 0.057%/month NIC-ToR link failure; 0.051%/month ToR critical failure; 5K–60K link-flaps/day; 1–2 crashes/month at 15K-Pod scale | Alibaba HPN reliability | SIGCOMM'24 | PAPER | 2024-08 |
| +14.9% e2e training throughput; up to 59.3%/158.2% AllReduce/multi-AllReduce BusBw gain | HPN vs. 3-tier DCN+ | SIGCOMM'24 | PAPER | 2024-08 |
| 15:1 | Alibaba Aggregation-Core oversubscription ratio | SIGCOMM'24 | PAPER | 2024-08 |
| 131,072 GPUs at full bisection BW; 8×100 Gb/s planes/800 Gb/s NIC; 128–256 Entropy Values/QP | OpenAI MRC (Multi-Rack Cluster) two-tier fabric | OpenAI/OCP paper, via MarkTechPost | OFFICIAL | 2026-05 |
| 1,000–30,000× faster failover convergence claim; ~30% conventional single-path slowdown avoided | MRC failover | OpenAI MRC blog | OFFICIAL | 2026-05 |
| 75K-GPU "Cluster A" production pretraining job | OpenAI MRC deployment | arXiv:2605.04333 | PAPER | 2026-05 |
| 92 GB/s NCCL send/recv at 42K-GPU MRC test scale; ~770 Gb/s (96% theoretical) peak link BW | MRC measured | arXiv:2605.04333 | PAPER | 2026-05 |
| 512 T0 switches × 256 NICs = 131,072 NICs, 2-tier | Azure MRC worked example | Microsoft techcommunity | OFFICIAL | 2025-11 |
| 100,000 → 555,000 GPUs, ~2 GW | xAI Colossus growth trajectory (mixed H100/H200/GB200) | Multiple secondary trackers | HEARSAY | 2025-2026 |
| ~110,000 GPUs ×2 clusters, ~210–220 MW ×2 | xAI Colossus 2 (GB200+GB300) | Secondary press | HEARSAY | 2026-01 |
| 0→200 MW in 6 months (vs. ~15mo peers) | xAI Colossus 2 build speed | SemiAnalysis | HEARSAY | 2025-09 |
| 95% vs. ~60% data throughput | Spectrum-X vs. standard Ethernet claim | NVIDIA/xAI | OFFICIAL (vendor) | 2024-2025 |
| 22 Pb/s bidirectional | Meta "AI-Backbone" bandwidth, Prometheus cluster | SemiAnalysis | HEARSAY | 2026 |
| >3 GW (from ~1 GW plan) over 2 years; 27 datacenters/6 campuses up to 80km apart | Meta Prometheus cluster | SemiAnalysis | HEARSAY | 2026 |
| Up to 2,000 km apart | Meta "Titan-cluster" inter-campus distance (non-Prometheus) | SemiAnalysis | HEARSAY | 2026 |
| 1–10 µs intra-region / 500 µs+ inter-region (~100km+) | Meta latency tiers driving sync-vs-async training choice | SemiAnalysis | HEARSAY | 2026 |
| 800/200 Gb/s per chip (Trn2 / Trn2-Ultra) scale-out BW | EFAv3 | SemiAnalysis/AWS | HEARSAY/OFFICIAL | 2025 |
| 6.5 µs packet latency (from 9 µs) | EFAv3 improvement over EFAv2 | SemiAnalysis | HEARSAY | 2025 |
| "10P10U": tens of PB/s bisection, 10 µs any-to-any | AWS EFA Gen3 internal spec | SDxCentral interview, Ron Diamant | OFFICIAL (interview) | 2025-11 |
| 140 kW/rack, 1,360 kW/row | Microsoft Fairwater (GB200 NVL72) rack/row power | Microsoft official blog | OFFICIAL | 2025-11 |
| 524,288 GPUs, 24,576 switches, rail-only 2-layer | Fairwater 2 Atlanta network | SemiAnalysis | HEARSAY | 2025-11 |
| >120,000 new fiber miles (1 yr, ~25% increase) | Microsoft AI WAN buildout, Fairwater superfactory | Microsoft official | OFFICIAL | 2025-11 |
| 28.8 Tbps (72×400G) | GCP A4X / Azure ND GB200 v6 per-rack scale-out (both vendors) | Google Cloud Blog; Microsoft Learn | OFFICIAL | 2025 |
| 13 Pb/s bisectional bandwidth | GCP Jupiter fabric | Google Cloud Blog | OFFICIAL | 2025 |
| 131,072 GPUs; 52 Pbps aggregate; 2.4 zettaFLOPS | OCI Supercluster ceiling | Oracle official blog | OFFICIAL | 2025-10 |
| Tier-0 ≤2 µs (256 GPU), Tier-1 ≤5 µs (2,048 GPU), Tier-2 ≤8 µs (131,072 GPU) | OCI 3-tier Clos latency budget | Oracle official | OFFICIAL | 2025-10 |
| 16 zettaFLOPS peak | OCI Zettascale10 (multi-DC follow-on) | Oracle news release | OFFICIAL | 2025-10-14 |

### A.2.5 Long-haul / multi-datacenter

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| ~5 Pbit/s (intra-region) / ~1 Pbit/s (inter-region) | Estimated needed inter-campus bandwidth for 100T+-param models | SemiAnalysis | HEARSAY | 2025-2026 |
| 0.64 s | Time to exchange 400 TB weights (100T-param-class) at estimated line rate | SemiAnalysis | HEARSAY | 2025-2026 |
| 43.2 ms | Physical speed-of-light-in-fiber RTT floor, US east-west coast | SemiAnalysis | HEARSAY | 2025-2026 |
| 121.6 Tbps | Max DWDM long-haul capacity per fiber pair (96 wavelengths) | SemiAnalysis | HEARSAY | 2025-2026 |
| 40%→30% MFU (25% relative drop) | ByteDance straggler-accumulation case, steps 7,500–19,000 | SemiAnalysis | HEARSAY | 2025-2026 |
| ~400× less bandwidth needed | Streaming DiLoCo vs. data-parallel, comparable quality | arXiv:2501.18512 | PAPER | 2025-01 |
| 88% goodput | Decoupled DiLoCo under high simulated failure rate | DeepMind (secondary-sourced) | HEARSAY | 2026-04 |
| 106B MoE, 90.8% AIME24 | Prime Intellect INTELLECT-3, trained on centralized GPUs (context for decentralized-training claims) | Secondary press | HEARSAY | 2025 (late) |

---

## A.3 Measured Training Throughput & MFU

### A.3.1 MLPerf Training records

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 10.9 min / 3,584 H100 | GPT-3-175B time-to-train, MLPerf v3.0 | NVIDIA blog | OFFICIAL | 2023-06 |
| 3.4 min / 11,616 H100 | GPT-3-175B, v4.0 | NVIDIA blog | OFFICIAL | 2024-06 |
| 64 GPU B200 preview, ~2×/GPU vs. H100 | GPT-3-175B, v4.1 preview | NVIDIA blog | OFFICIAL | 2024-11 |
| 2,496 GB200 GPUs / 27.33 min | Largest Llama-3.1-405B submission, v5.0 (CoreWeave/NVIDIA/IBM) | CoreWeave blog | OFFICIAL | 2025-06 |
| 512→2,496 GPUs: 121.1→62.1→42.46→32.6→27.33 min | Full v5.0 scaling ladder, 90.9% weak-scaling efficiency | CoreWeave blog | OFFICIAL | 2025-06 |
| 121.09 min (GB200) vs. 269.12 min (H100), 512 GPU (2.2× gen-on-gen) | v5.0 | NVIDIA dev blog | OFFICIAL | 2025-06 |
| 1,960 TFLOPS/GPU → 39.2% implied MFU | GB200 NVL72, Llama-3.1-405B, 512 GPU, v5.0 | NVIDIA dev blog + calc | OFFICIAL/DERIVED | 2025-06 |
| 5,120+ GB300 GPUs / 10 min | New Llama-3.1-405B record, v5.1 | NVIDIA blog | OFFICIAL | 2025-11 |
| 2,560 GB200 GPUs / 18.79 min (45% faster than v5.0 on same hardware) | v5.1, same-hardware software gain over 6 months | NVIDIA dev blog | OFFICIAL | 2025-11 |
| GB300 1.26 min vs. GB200 1.598 min vs. 64×B200 2.019 min | Llama-2-70B-LoRA, v5.1 | Lambda blog | OFFICIAL | 2025-11 |
| MI300X 27.95 min → MI325X 21.05 min → MI355X 10.18 min | AMD Llama-2-70B-LoRA generational progression | AMD/ROCm blogs | OFFICIAL | 2025 |
| MI355X within 3–6% of B200/B300 | Llama-2-70B-LoRA and Llama-3.1-8B, v5.1 | ROCm blog | OFFICIAL | 2025-11 |
| DeepSeek-V3-671B: GB300 NVL72, 8,192 GPUs, 2.02 min to target quality; GB300 1.6× faster than GB200 at same scale; 1,298→1,648 TFLOPS/GPU in 3 months | MLPerf v6.0, first audited large-MoE benchmark, zero non-NVIDIA submissions | MLCommons/NVIDIA/CoreWeave | OFFICIAL | 2026-06-16 |
| 95 systems, 24 orgs, 13 accelerator types, 60% multi-node | MLPerf Training v6.0 overall round | MLCommons | OFFICIAL | 2026-06-16 |
| Llama-3.1-405B: GB200 NVL72, 8,192 GPUs, 7.07 min; GB300 ~1.5× faster | MLPerf v6.0 companion dense benchmark | NVIDIA/MLCommons | OFFICIAL | 2026-06 |
| MI355X within ~5–6% of B200 on Llama3.1-8B pretrain / Llama2-70B LoRA (first Primus-based submission) | MLPerf v6.0 | AMD/ROCm blogs | OFFICIAL | 2026-06 |
| MI355X 8.27 min vs. MI300X 28.65 min (3.46× gen leap) | Llama2-70B LoRA, 8 GPU | AMD/ROCm blogs | OFFICIAL | 2026-06 |
| 512/64 GPUs largest AMD submissions | MLPerf v6.0 (OCI collab / pure-AMD) | MLCommons via AMD blog | OFFICIAL | 2026-06 |

### A.3.2 NVIDIA Megatron-Core / Megatron-Bridge published numbers

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 1,233 / 1,048 / 857 TFLOPS/GPU | DeepSeek-V3-685B: GB300-MXFP8 / GB200-MXFP8 / GB200-BF16, 256 GPUs, seqlen 4096 | arXiv:2603.07685 Table 11 | PAPER | 2026-03 |
| 368 TFLOPS/GPU | DeepSeek-V3-685B, H100 FP8-blockwise, 1,024 GPUs | arXiv:2603.07685 Table 11 | PAPER | 2026-03 |
| 974 / 919 TFLOPS/GPU | Qwen3-235B, GB300/GB200 MXFP8, 256 GPUs | arXiv:2603.07685 Table 11 | PAPER | 2026-03 |
| 1,150 TFLOPS/GPU | Qwen3-235B, GB300 MXFP8, 128 GPUs, seqlen 131,072 (long-context stress test) | arXiv:2603.07685 Table 11 | PAPER | 2026-03 |
| 4,730 / 4,020 / 3,298 / 1,412 tokens/s/GPU | DeepSeek-V3: GB300-MXFP8 / GB200-MXFP8 / GB200-BF16 / H100-FP8 respectively | arXiv:2603.07685 Table 11 | PAPER | 2026-03 |
| <50% (MoE, DeepSeek-V3) vs. ~70% (dense, Llama-3-405B) | Share of layer time in GEMMs | arXiv:2603.07685 §1.2 | PAPER | 2026-03 |
| ~9% | Routing+permutation overhead as share of layer time, post-fusion | arXiv:2603.07685 | PAPER | 2026-03 |
| 88% / 129% | Long-context (256K) MFU as % of short-context MFU: DeepSeek-V3 / Qwen3-235B-A22B, 256 H100 | arXiv:2603.07685 §6.5 | PAPER | 2026-03 |
| 1048 tok/s/GPU, 2,646 TFLOP/s/GPU | Llama3.1-405B, DGX-GB300, 256 GPU, FP8, TP4/PP8/CP1/VP4 | Megatron-Bridge perf-summary (26.06 container) | OFFICIAL/CODE | 2026-07 |
| 1,413 tok/s/GPU, 3,575 TFLOP/s/GPU | Llama3.1-405B, DGX-GB300, 256 GPU, NVFP4 (fastest dense config in table) | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 326 tok/s/GPU, 822 TFLOP/s/GPU | Llama3.1-405B, DGX-H100, 1024 GPU, FP8 | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 6,338–6,422 tok/s/GPU, 1,648–1,670 TFLOP/s/GPU | DeepSeek-V3, DGX-GB300, 256 GPU, MXFP8 | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 4,969 tok/s/GPU, 1,292 TFLOP/s/GPU | DeepSeek-V3, DGX-GB200, 256 GPU, MXFP8 | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 3,541 tok/s/GPU, 920 TFLOP/s/GPU | DeepSeek-V3, DGX-B300, 256 GPU, MXFP8 | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 45,275 tok/s/GPU | Qwen3-30B-A3B, DGX-GB300, 8 GPU, MXFP8 (highest tok/s/GPU in the whole table) | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 9,015 tok/s/GPU, 1,335 TFLOP/s/GPU | Qwen3-235B-A22B, DGX-GB300, 256 GPU, MXFP8 | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 5,372 tok/s/GPU, 1,099 TFLOP/s/GPU | Kimi-K2 (Muon optimizer), DGX-GB300, 256 GPU, MXFP8 | Megatron-Bridge perf-summary | CODE | 2026-07 |
| 199.5 GB/GPU unoptimized → <80 GB/GPU fully optimized | DeepSeek-V3 BF16 memory, PP4×VPP4×EP64, 256 GPUs | arXiv:2603.07685 | PAPER | 2026-03 |
| 20–60% training time lost to EP all-to-all pre-optimization; 30–40% post-dispatcher-optimization | DeepSeek-V3 EP64 | arXiv:2603.07685 §4.2 | PAPER | 2026-03 |
| 49.3% / 39.0% MFU | Mixtral-8x22B / Qwen2-57B-A14B, H100, MoE Parallel Folding | arXiv:2504.14960 | PAPER | 2025-04 |
| 502 PFLOP/s aggregate, 52% of peak/GPU | 1T-param model, 3,072 A100, interleaved pipeline schedule | arXiv:2104.04473 | PAPER | 2021 |
| 54.2% MFU (vs. 42.1% baseline, +29%) | 530B GPT-3-style, 2,240 A100, sequence-parallel + selective recompute | arXiv:2205.05198 | PAPER | 2022 |
| 76% scaling efficiency, 15.1 PFLOP/s aggregate | 8.3B model, 512 GPU (V100-era), original Megatron-LM | arXiv:1909.08053 | PAPER | 2019 |

### A.3.3 DeepSeek measured throughput / MFU

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 385.8 TFLOPS/GPU avg (recompute) / 424.3 TFLOPS/GPU (3rd-party profiler) | DeepSeek-V3 achieved compute, two independent reconstructions | DeepSeek paper Table 2 recompute vs. Medium/dlrover | DERIVED / HEARSAY | 2026-07 / ~2025 |
| ~19.5–21.4% FP8-MFU (peak 1,979 TFLOPS) / ~39.0–42.9% BF16-MFU (peak 989.5 TFLOPS) | DeepSeek-V3, two normalizations of the same achieved-compute figure | Derived from arXiv:2412.19437 | DERIVED | 2026-07 |
| ~23% independent MFU estimate | DeepSeek-V3 | SemiAnalysis | HEARSAY | 2025-01 |
| 17.3% (SXM H800 peak) vs. "~23%" (alt. PCIe H800 peak assumption) | Same DeepSeek-V3 run, MFU differs by choice of H800 SKU peak (1.5 vs. 1.979 PFLOPS) | Notes' own calc vs. secondary analyses | **[CONFLICT — methodological, not factual]** | 2026-07 / 2025 |
| 17.1% (H100) / 13.1% (B200) / 18.0% (GB200) | DeepSeek-V3 MoE MFU per NVIDIA's own published TFLOP/GPU tables ÷ dense-FP8 peak | NVIDIA docs + calc | DERIVED | accessed 2026-07 |
| 16.6% BF16 MFU (H100) | DeepSeek-670B MoE (vs. 54.5% dense Llama3-405B) | SemiAnalysis | HEARSAY | 2025-08 |
| 1.3×→2.5×→4.7× (proj.) tok/s/GPU | GB200 vs. H100, DeepSeek-670B, May→Jul→Dec-2025 | SemiAnalysis | HEARSAY | 2025-08 |
| 18× faster | 64-GPU all-to-all, GB200 NVL72 domain vs. H100 | SemiAnalysis | HEARSAY | 2025-08 |
| Beat-12 GB200 MFU (Megatron-Core, DeepSeek-V3-685B, 1,048 TFLOPS/GPU measured): 46.6% (naive) → **41.9%** corrected BF16 denominator (2,500 not 2,250) | Correction using notes/30's authoritative GB200 dense-BF16 peak | Notes/30 self-correction of notes/16 | DERIVED | 2026-07 |
| 21.0% (if FP8, 1,048/5,000) vs. naive 23.3% (1,048/4,500) | Same measured figure, corrected FP8 denominator | Notes/30 | DERIVED | 2026-07 |
| "38–46% sustained FP8 training MFU on NVL72" | Cited elsewhere as NVIDIA official; **could not be independently verified** against local SemiAnalysis PDF, NVIDIA dev blog, or a fresh web source | Notes/30's own audit | **[UNVERIFIED]** | n.d. |
| 651→859 tok/s (DeepEP, +32%) → 918 tok/s (DeepEP+MXFP8, +41%) | DeepSeek-V3-671B, 256×B200 (TP2/PP2/EP32) | PyTorch official blog | OFFICIAL | 2026-03 (upd. 2026-05) |
| 20 of 132 SMs | Cross-node all-to-all comm budget, H800 | arXiv:2412.19437 §3.2.2 | PAPER | 2024-12 |
| 160 GB/s NVLink vs. 50 GB/s IB (3.2:1) | H800 intra- vs. inter-node bandwidth | arXiv:2412.19437 | PAPER | 2024-12 |

### A.3.4 Llama measured throughput / MFU

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 44 days | Llama3-70B training time, full v5p pod, 40% MFU | scaling-book applied-training.md | HEARSAY (book estimate) | 2025-02 |
| 38.6% (H100) / 37.2% (B200) / 36.8% (GB200) | Llama3.1-405B MFU per NVIDIA Megatron-Bridge tables ÷ dense-FP8 peak | NVIDIA docs + calc | DERIVED | accessed 2026-07 |
| 38–43% BF16 MFU overall; 43% peak @8K-GPU | Llama3-405B, Meta's own reported figures | Meta / arXiv:2407.21783 | PAPER | 2024-07 |
| FP8 ~43%, BF16 ~54% MFU, 576–2,304 H100 | Llama3-405B, re-benchmarked ~1 year after Meta's original figures | SemiAnalysis | HEARSAY | 2025-08 |
| 41% BF16 MFU (16K H100), 38% at 131K seqlen | Llama3-405B, published Herd-of-Models pretrain/mid-train, CP=16 | SemiAnalysis | HEARSAY | 2025-08 |
| $1.95/M tokens, $29.1M total | Llama3-405B 15T-token BF16 pretrain cost, 2,304× H100 | SemiAnalysis | HEARSAY | 2025-08 |
| 38.1%→35.5% FP8 / 54.5%→53.7% BF16 MFU | Llama3-70B weak scaling, 64→2,048× H100 | SemiAnalysis | HEARSAY | 2025-08 |
| 2.4× tok/s/GPU, 50% cheaper ($0.061 vs. $0.093/M tok) | Llama4 Maverick 400B MoE, GB200 vs. H100 | SemiAnalysis | HEARSAY | 2025-05 (data) |
| 390 TFLOPs/GPU | Llama4 Behemoth pretraining, FP8, 32K GPUs | Meta AI blog | OFFICIAL | 2025-04 |
| 1–2% diurnal throughput variation | Llama3-405B, temperature-driven voltage/frequency effect | arXiv:2407.21783 | PAPER | 2024-07 |

### A.3.5 TPU / MaxText measured throughput

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 66.86% MFU | Single TPU v5e pod, BF16, MaxText 128B model | Google Cloud Blog | OFFICIAL | 2023-11 |
| 5.32 exa-OP/s observed, 1.2×–1.4× steps/s speedup | Full 199-pod v5e cluster (50,944 chips), INT8 quantized (AQT) vs. BF16 | Google Cloud Blog | OFFICIAL | 2023-11 |
| Llama2-70B: 65.4% MFU, 692 tok/s/dev | v5p-512 pod, BF16, SL=4096 | MaxText tiering docs | OFFICIAL | 2026-06 |
| Mixtral 8x7B: 52.56% MFU, 2,909 tok/s/dev | v5p-256, BF16, SL=4096 | MaxText tiering docs | OFFICIAL | 2026-06 |
| Llama3.1-8B: 45.46% MFU, 7,207 tok/s/dev | Trillium v6e-256, BF16, SL=8192 | MaxText tiering docs | OFFICIAL | 2026-06 |
| Llama3.1-70B: 50.33% MFU, 960 tok/s/dev | Trillium v6e-256, BF16, SL=8192 | MaxText tiering docs | OFFICIAL | 2026-06 |
| Llama3.1-405B: 38.55% MFU, 123 tok/s/dev | Trillium v6e-256, BF16, SL=8192 | MaxText tiering docs | OFFICIAL | 2026-06 |
| Mixtral 8x22B: 36.2% MFU, 1,326 tok/s/dev | Trillium v6e-256, BF16, SL=4096 | MaxText tiering docs | OFFICIAL | 2026-06 |
| Gemini 2.5: 93% goodput | Sync data-parallel across multiple 8,960-chip v5p pods, multi-datacenter | Gemini 2.5 report, cited in arXiv:2606.15870 | PAPER | 2026-06 (citing 2025) |
| Gemini 1.0: 97% goodput | Smaller scale, TPU v4 | Gemini23 report | PAPER | 2023 |
| Gemini 1.5: 85%→97% goodput | Largest-scale training job | arXiv:2403.05530 | PAPER | 2024-03 |
| ~97% throughput retained | Gemini 2.5, slice-granularity elasticity during localized TPU failure | Gemini 2.5 report | PAPER | 2025 |

### A.3.6 Chinese-lab / ByteDance measured

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 47% MFU | ERNIE-4.5-300B-A47B pretraining, 2016 H800, RoCE, seq4096, batch15120 | ERNIE 4.5 Technical Report | PAPER | 2025-06 |
| 55.2% MFU, 175B model, 12,288 GPUs (1.34× over Megatron-LM baseline) | ByteDance MegaScale | arXiv:2402.15627 | PAPER | 2024-02 |
| 1,984.0k vs. 1,466.8k tok/s | MegaScale vs. Megatron-LM throughput, 175B/12,288 GPU | arXiv:2402.15627 Table 2 | PAPER | 2024-02 |
| 1.41M tokens/s, 27.89–32.48% MFU | MegaScale-MoE, 352B-param MoE, 1,440 Hopper GPUs (1.65–1.88× vs. Megatron-LM) | arXiv:2505.11432 (EuroSys'26) | PAPER | 2025-05 |
| 30.0% MFU, 1.46M tok/s (vs. 18.9% MFU/0.61M tok/s prior config) | PanGu Ultra MoE, 718B/39B-active, 6K vs. 4K Ascend 910B | arXiv:2505.04519 | PAPER | 2025-05 |
| >52% MFU | Pangu Ultra (135B dense), 8,192 NPUs, 13.2T tokens | arXiv:2504.07866 via secondary aggregation | PAPER/HEARSAY | 2025 (Apr–May) |
| 1.96×/1.71× | COMET single-MoE-layer / end-to-end speedup | arXiv:2502.19811 | PAPER | 2025-02 |
| 2.87×/3.08× | ByteDance Flux AG+GEMM / GEMM+RS speedup vs. unfused PyTorch, L20, TP=8 | github.com/bytedance/flux | CODE | 2025 |
| Up to 44.97× | Triton-distributed vs. raw NCCL/RCCL, targeted microbenchmarks | arXiv:2504.19442 | PAPER | 2025-04 |
| Up to 4.8× MFU | InternEvo vs. baseline, 7–65B dense, 256K seq, 128 GPU | arXiv:2401.09149 | PAPER | 2024-01 |
| 1.8× | NVIDIA Wide-EP (TensorRT-LLM) EP32 vs. EP8 tok/s, GB200 NVL72, inference (not pretraining) | NVIDIA dev blog | OFFICIAL | ~2026-01 |

### A.3.7 torchtitan / PyTorch-ecosystem measured

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 65.08%/12.59%/30% speedups | 1D/2D/3D parallelism stacking, Llama3.1 8B/70B/405B, 128/256/512 H100 | arXiv:2410.06511 | PAPER | 2024-10 |
| 1.22×–1.28× speedup, ~5% delta scaling 4→188 nodes (47×) | MXFP8, Llama3-70B, up to 1,856×B200, Crusoe | PyTorch official blog | OFFICIAL | 2025-09 |
| 651→859 (DeepEP, +32%) → 918 tok/s (DeepEP+MXFP8, +41%) | DeepSeek-V3-671B, 256×B200 (TP2/PP2/EP32) | PyTorch official blog | OFFICIAL | 2026-03 (upd. 2026-05) |
| 2.77× cumulative speedup (AITER +15%, FP8 GEMM +102%, FP8 grouped GEMM +60%) | DeepSeek-V3-671B, 1024×MI325X, TensorWave | PyTorch official blog | OFFICIAL | 2025-12 |
| 96–97% scaling efficiency 256→512→1024 GPU (MI325X); 100–102% Llama4-Scout 32→512 | AMD MoE pretraining | PyTorch official blog | OFFICIAL | 2025-12 |
| 400B total/13B active, 256+1 experts top-4; 2048×B300 GPUs; 17T tokens | Arcee Trinity Large | arXiv:2602.17004 | PAPER | 2026-02 |

---

## A.4 Failure & Reliability Rates

### A.4.1 Interruption rates, MTBI/MTBF

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 419 unexpected + 47 planned = 466 interruptions, 54 days, ≤16,384 H100 | Llama 3 405B run | arXiv:2407.21783 Table 5 | PAPER | 2024-07 |
| GPU 30.1%, HBM3 17.2%, SRAM 4.5%, host CPU 4.1%, thermal 1.4%, NIC 1.7%, switch/cable 8.4%, SW bugs 12.9%, NCCL watchdog 1.7%, SDC 1.4%, unplanned maint. 7.6% | Breakdown of the 419 unexpected interruptions above | arXiv:2407.21783 Table 5 | PAPER | 2024-07 |
| >90% effective training time | Llama 3 405B run, despite the failure rate | arXiv:2407.21783 | PAPER | 2024-07 |
| 2,111 H100-days implied MTBF | Llama 3 405B (54-day run, 466 interruptions/419 HW failures) | SemiAnalysis | HEARSAY | 2025-11 |
| H100 MTBI 2,000–5,000 GPU-days; GB200 NVL72 MTBI 1,000–3,000 GPU-days | Field-derived reliability ranges | SemiAnalysis (primary PDF) | HEARSAY | 2025-08 |
| 72 GPUs (whole rack) vs. 8 GPUs (HGX board) | GB200 vs. H100 "blast radius" | SemiAnalysis | HEARSAY | 2025-08 |
| Backplane swap 8–24h; switch-tray swap 1–4h (rack drain); compute-tray swap 1–4h (no drain) | GB200 NVL72 service times | SemiAnalysis | HEARSAY | 2025-08 |
| ≤200 mating cycles | GB200 NVL72 Paladin HD backplane connector wear (Amphenol) | SemiAnalysis | HEARSAY | 2025-08 |
| 6–7 months | GB200 NVL72 backplane firmware-bug resolution time (ES/PS to stable v1.3) | SemiAnalysis | HEARSAY | 2025-11 |
| DeepSeek-670B perf/TCO: 2.76× raw → ~1.5–2.1× reliability-adjusted; Llama4-400B: 50% cheaper raw → 20–30% cheaper adjusted | GB200 NVL72 vs. H100 economics | SemiAnalysis | HEARSAY | 2025-08 |
| Machines 0.08%, ICI cables 0.005%, OCS 0.04% daily failure rate | TPU v4 resiliency | NSDI'24 (Zu et al.) | PAPER | 2024-04 |
| >99.9% host availability required without OCS; 2,304 CPU hosts (Ironwood) | TPU supercomputer architecture | arXiv:2606.15870 | PAPER | 2026-06 |
| 2.3 interruptions/1,000 servers/day (32K-GPU ref.); failure every 18 min at 100K GPUs; 44%→80% effective training time (naive sync → FT-HSDP) | FT-HSDP | arXiv:2602.00277 | PAPER | 2026-02 |
| 43,264 explicit + 5,948 implicit failures / 778,135 jobs over 3 months; fleet >200,000 GPUs | ByteRobust production | arXiv:2509.16293 | PAPER | 2025-09 |
| 97% ETTR (dense, 3mo, 9,600 GPUs); 91% ETTR (MoE, 1mo) | ByteRobust | arXiv:2509.16293 | PAPER | 2025-09 |
| ~2 machines/day attrition | OPT-175B cloud-provider hardware baseline | metaseq chronicles logbook | LOGBOOK | 2021-12/2022-01 |
| ~90 restarts / 148 pages of notes, 33 days, 4.30e23 FLOPs | OPT-175B full run, 1,024× A100 80GB | metaseq chronicles logbook | LOGBOOK | 2022-01-07 |
| 149–150 TFLOPs/GPU steady state | BLOOM-176B, Jean Zay, post-ramp (day 7) | bigscience chronicles | LOGBOOK | 2022-03-18 |
| Up to 14 nodes, 3 continents, 30 providers; 83–96% util, 36.2–41.4% MFU | INTELLECT-1 (10B, 1T tokens), decentralized training | arXiv:2412.01152 | PAPER | 2024-12 |
| 97% (99% excl. dev work) uptime; 3m42s median / 38m54s mean recovery | OLMo Hybrid (Ai2 × Lambda) reliability | Lambda official blog | OFFICIAL | ~2025-12/2026-01 |
| 10.36% of Azure GPU nodes flagged defective | SuperBench, USENIX ATC'24 Best Paper | arXiv:2402.06194 | PAPER | 2024-07 |
| 22.61× | SuperBench mean-time-between-incidents improvement | arXiv:2402.06194 | PAPER | 2024-07 |
| 0.23 hours (~14 min) projected MTTF | At 131,072-GPU job scale | arXiv:2410.21680 | PAPER | 2024-10 |
| 14%→4% | Large-job (512+ GPU) failure rate, before/after proactive lemon-node detection | arXiv:2410.21680 | PAPER | 2024-10 |
| 31.19%→1.16% (~30× reduction) | Alibaba C4 error-induced training downtime, before/after | arXiv:2406.04594 | PAPER | 2024-06 |
| 3–10% at delivery, absent adequate burn-in | New accelerator batch failure rate | SemiAnalysis-sourced burn-in data | HEARSAY | 2024-10 (cited) |
| 95% of Stas's own observed hardware failures | Accelerators as share of total hardware failures, personal experience | Stas repo | LOGBOOK | 2026-07 |
| <400 GPU-days typical MTBF | Gold/silver-tier neocloud (5+ failures/day, customer reports) | SemiAnalysis ClusterMAX 2.0 | HEARSAY | 2025-11 |

### A.4.2 Silent data corruption (SDC)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| ~1 per 1,000 devices (up from ~1 per million historically) | Meta hardware reliability | engineering.fb.com | OFFICIAL | 2025-07 |
| Weekly-biweekly (Gemini scale); 6/419 = 1.4% (Llama 3) | SDC incidence, two labs directly compared | arXiv:2502.12340 | PAPER | 2025-02 |
| Up to 5.1% worst-node gradient noise; mismatch freq up to 2.89e-2 | SDC impact quantification | arXiv:2502.12340 | PAPER | 2025-02 |
| FP8: <1% PPL deviation (safe) or crash (binary); BF16/FP16: up to 400–600% PPL deviation, silent | Numeric-format SDC resilience | arXiv:2604.10390 | PAPER | 2026 |
| Nullification 50.68%, non-special bit-flip 48.31%, NaN/Inf only 1.01% | GPU fault-injection corruption breakdown, 3M+ sim-hours | arXiv:2605.04213 | PAPER | 2026 |
| ~once/week | SDC loss-spike frequency traced to a single faulty GPU | ZAYA1 cluster, arXiv:2511.17127 | PAPER | 2025-12 |
| "Every week or two" | Expected SDC frequency at Gemini Ultra scale | Gemini 1.5 Technical Report | PAPER | 2024-03 |

### A.4.3 Fault-tolerance systems: measured gains

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 99.11–99.71% vs. 39.84–43.07% MFU | ByteRobust in-memory checkpointing vs. sync-to-remote-FS | arXiv:2509.16293 | PAPER | 2025-09 |
| 80–93% recovery-time reduction; up to 95% goodput | AWS checkpointless training (peer-to-peer, EFA) vs. traditional restore | AWS official blog | OFFICIAL | 2025-12 |
| 1min45s vs. 15min recovery; 5% memory overhead | AWS checkpointless training | SemiAnalysis | HEARSAY | 2026-04 |
| 82.3% step-commit @60s MTBF; 30.2% @~15s MTBF; <1s replica recovery (Gloo) | torchft/Crusoe fault-tolerance demo, 300 L40S GPUs | PyTorch official blog | OFFICIAL | mid-2025 |
| 1.35–1.60× throughput vs. ReCycle/torchft; <1s communicator recovery (up to 82×/3.6× faster); 51% MTTR reduction | ElasWave, 96 NPUs | arXiv:2510.00606 | PAPER | 2025-10 |
| 2.63–5.28× | Malleus efficiency gain vs. hybrid-parallel baselines under stragglers, up to 110B models | arXiv:2410.13333 | PAPER | 2024-10 |
| <100ms MRO expert-placement compute @1,024-GPU scale; 20–40s total failure-to-resume | Lazarus | arXiv:2407.04656 | PAPER | 2024-07 |
| 5.7×/3.4×/2.3× vs. DeepSpeed-MoE (high-freq injected / real spot-trace / infrequent failures) | Lazarus | arXiv:2407.04656 | PAPER | 2024-07 |
| 0.3–5.6s error-detection latency range across 4 tiers; 1.2–1.9×/3.7–3.8×/4.8–5.8× efficiency vs. Megatron/Oobleck/Varuna | Unicron | arXiv:2401.00134 | PAPER | 2023-12 |
| 3.6–58.7× | Acme async-checkpoint restart speedup vs. sync baseline | arXiv:2403.07648 | PAPER | 2024-03 |
| ~2% min. spare-to-production GPU ratio (NVL72) | Vendor guidance | Lenovo NVL72 documentation | OFFICIAL | 2025-2026 |

### A.4.4 Historical training-run postmortems

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 40+ restarts in 2 weeks | OPT-175B, Nov 17–Dec 3 2021 window | metaseq chronicles | LOGBOOK | 2021-12-03 |
| 16 machines failed health check post-restore | OPT-175B "Omicron SEV" cluster deletion | metaseq chronicles | LOGBOOK | 2021-12-21 |
| 7.3h lost | BLOOM-176B CUDA crash, 2 iterations before checkpoint | bigscience chronicles | LOGBOOK | 2022-03-21 |
| grad norm 0.24→960.35 in 2 steps, recovered in ~30 iterations | BLOOM-176B's one lm-loss spike | bigscience chronicles | LOGBOOK | 2022-04-12 |
| 149→140 TFLOPs (~6%) drop, traced to 1 bad node via binary search | BLOOM-176B mystery throughput regression | bigscience chronicles | LOGBOOK | 2022-04-28 |
| 18h, then 6h, then 11h lost (3 incidents) | BLOOM-176B eval-hang saga (DeepSpeed race + PyTorch DataLoader bug) | bigscience chronicles | LOGBOOK | 2022-04/05 |
| ~20 spikes/run; rewind ~100 steps, skip 200–500 batches | PaLM 540B spike-handling protocol | arXiv:2204.02311 | PAPER | 2022 |
| 154 checkpoints/model; full-suite v0→v1 rerun | Pythia reproducibility correction | EleutherAI GitHub | CODE | 2023 |
| 35 days → ~3 days | OLMo 7B-scale training-time evolution, 2024→2025 | interconnects.ai interview | HEARSAY | ~2025 |
| ~80,000 steps: 3 failed numerics fixes → 1 architecture swap | Marin 32B "Necromancy Restart" → QK-Norm switch | marin-community GitHub logbook | LOGBOOK | 2025 |
| 4-of-12 nodes lost simultaneously → checkpoint resume | INTELLECT-1 worst disclosed incident | arXiv:2412.01152 | PAPER | 2024-12 |
| 46,080 GPU-hrs restart cost; 437,760 total; 161,280 ablation/debug (>50% of 276,480 main-run hrs) | SmolLM3 compute breakdown | HuggingFace Smol Training Playbook | LOGBOOK | 2025-10 |

---

## A.5 Checkpoint & Storage

### A.5.1 Checkpoint state sizing

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 140 TB total (20TB BF16 + 40TB FP32 master + 80TB Adam m,v) | Full Adam-style checkpoint state, 10T-param model | Derived (VAST Data's "14 bytes/param" convention) | DERIVED | 2026 |
| ~100 TB (Muon, FP32 momentum) / ~80 TB (Muon, BF16 momentum) | Same model, Muon replacing Adam | Derived | DERIVED | 2026 |
| ~2.33 TB/s aggregate write needed for <60s full-140TB drain | Derived from checkpoint math | Derived | DERIVED | 2026 |
| 74–86% optimizer-state reduction, 8-bit Muon vs. Adam-32/Muon-32 | Quantized Muon, small-model ablation | arXiv:2509.23106 | PAPER | 2025-09 |
| Params × 12 bytes | Checkpoint-size rule of thumb (full training state), used for ramdisk sizing | MaxText checkpointing docs | OFFICIAL | 2026-06 |
| `params_B * 18 * 1.25 / gpu_GB` (train) vs. `params_B * 2 * 1.25 / gpu_GB` (infer) | GPU-count back-of-envelope formula | Stas repo | DERIVED | 2026-07 |

### A.5.2 Checkpoint system performance

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 0.01–0.04 s per-step blocking time | ByteRobust in-memory peer replication | arXiv:2509.16293 | PAPER | 2025-09 |
| 50–200 GB/s actual measured *global* checkpoint-storage BW | Across 40 real production runs, 85,000+ checkpoints, 18 clusters, 45B–1T+ params — "well below 1TB/s even for trillion-parameter models" | blog.vastdata.com | OFFICIAL | 2026 |
| 3.6 min median checkpoint duration | 800B-param example run, 40-min interval, ~9% overlap | VAST Data blog | OFFICIAL | 2026 |
| 150 min MTTI, 405B/16,000 H100; recommended interval ~15 min | Mean time to interrupt drives checkpoint-frequency guidance (~1/10 MTTI) | VAST Data blog | OFFICIAL | 2026 |
| Up to 529.22× (10B dense/FSDP) / up to 87.80× (28B MoE/Megatron) | Checkpoint-stall reduction, ByteCheckpoint vs. native | arXiv:2407.20143 | PAPER | 2024/2025 |
| 54.20× avg stall reduction; up to 9.96× save / 8.80× load speedup | ByteCheckpoint vs. open-source baselines | arXiv:2407.20143 | PAPER | 2024-07 (rev. 2025-04) |
| 173.0s (Orbax) vs. 595.0s (PyTorch DCP), 3.4× save; 117.2s vs. 169.5s, 1.4× load | 405B model checkpoint | arXiv:2605.23066 | PAPER | 2026 |
| Sync ckpt overhead ~50s/100 iters; async <0.5s (19×); zero-overhead 0.06s staging | DCP, Llama3-8B, 512×H100 | PyTorch official forum | OFFICIAL | 2024-10 |
| ~16s checkpoint load (vs. ~2,400s naive), ~150× speedup | 300TB-scale, Google Cloud | Google Cloud Blog | OFFICIAL | 2023-11 |
| <2 min total elastic recovery (13s detect + 50s reschedule + 5.4s restore + 12.7s to next step) | 48× TPU v5e-16, Qwen3-0.6B, SIGKILL demo | Google Developers Blog | OFFICIAL | 2026-07-06 |
| 7 GiB checkpoint restored in 5.39s; 88 steps lost (3388→3300) | Same demo | Google Developers Blog | OFFICIAL | 2026-07-06 |
| 62s | Llama 3 405B checkpoint-save planning, 8,960 GPUs | Secondary technical summary of Llama 3 herd paper | HEARSAY | 2024-2025 |
| 1MB–4GB per-GPU checkpoint shard; 7,500 SSD servers | Llama 3 checkpoint → Meta Tectonic FS | arXiv:2407.21783 | PAPER | 2024-07 |
| 2.3TB saved in 40s across 384 processes, GPFS-over-NVMe | BLOOM-176B checkpoint | Stas repo (JeanZay HPC retrospective) | LOGBOOK | 2022 (repo 2026-07) |
| ~2.6TB/day checkpoint volume (every 3h); overhead ≈0.37% of training time | BLOOM-176B | Stas repo | LOGBOOK | 2022 (repo 2026-07) |
| ~60s → <1s | Checkpoint-save blocking time, async + distributed-optimizer sharding | Nemotron 3 Ultra, arXiv:2606.15007 | PAPER | 2026-06 |
| 269s → 30s | Checkpoint save time optimization | Ant Ling-1T, arXiv:2510.22115 §5.4 | PAPER | 2025-10 |

### A.5.3 Storage systems (throughput)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 6.6 TiB/s aggregate read (180-node, w/ training traffic) | DeepSeek 3FS (open-sourced) | github.com/deepseek-ai/3fs | CODE | 2025-02 |
| 3.66 TiB/min GraySort (110.5 TiB / 8,192 partitions) | 3FS via smallpond | github.com/deepseek-ai/3fs | CODE | 2025-02 |
| 40 GiB/s peak | 3FS KVCache read throughput | github.com/deepseek-ai/3fs | CODE | 2025 |
| 9 TB/s outbound, 8 TB/s achieved read | FF2-era 3FS (180 nodes), pre-open-source | arXiv:2408.14158 | PAPER | 2024-08 |
| 6.3–8.1 GB/s (HFReduce) vs. 1.6–4.8 GB/s (NCCL) | Inter-node allreduce BW, Fire-Flyer 2, 16–1,440 GPUs | arXiv:2408.14158 | PAPER | 2024-08 |
| 2 TB/s sustained, 7 TB/s peak, 240 PB capacity | Meta Tectonic FS (Llama 3) | Secondary summary of Llama 3 herd paper | HEARSAY | 2024-2025 |
| 250 GB/s | B200 + NVMe-oF + Magnum IO sustained checkpoint-workload throughput | Vendor benchmark aggregation | HEARSAY | 2025-2026 |
| 190 GB/s per appliance; "10s of TB/s" aggregate claimed | DDN AI400X3M, deployed at xAI Colossus | ddn.com | HEARSAY (vendor) | 2026 |
| Up to 10 TB/s throughput, 80 PB capacity | GCP Managed Lustre | Google Cloud Blog | OFFICIAL | 2025-2026 |
| 1,006.3 MBps / 257,614 IOPS (16KiB read); 2,762.0 MBps / 707,062 IOPS (1GiB read); 1,943.9 MBps / 497,638 IOPS (1GiB write) | `fio-scan` reference run, Samsung 980 PRO 2TB NVMe | Stas repo | BENCHMARK | 2023-12 (repo 2026-07) |
| 125/62 GBps SU-agg read/write (Enhanced tier, 576-GPU SU) | NVIDIA GB200 Reference Architecture storage guidance | docs.nvidia.com | OFFICIAL | 2025-11 |
| 125/62 GBps SU-agg read/write (Best tier, 256-GPU SU) | NVIDIA H100 RA storage guidance | docs.nvidia.com | OFFICIAL | 2025-11 |
| 89% of provisioned capacity reliably delivered | GCP FileStore | Google Cloud docs | OFFICIAL | 2026-07 |

### A.5.4 Data-loading pipeline performance

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 1.96T tokens deduped in 30 min on 32 H100s | NeMo Curator fuzzy dedup | NVIDIA docs/blog | OFFICIAL | 2024-2026 |
| 1.1T tokens (RedPajama) deduped in 1.8h on 64 A100s | NeMo Curator (RAPIDS/Dask) | NVIDIA docs/blog | OFFICIAL | 2023-2024 |
| 4.5× e2e throughput, 13.5× CPU memory reduction | MegaScale-Data actor-model loader vs. baselines, multimodal VLM training | arXiv:2504.09844 (EuroSys) | PAPER | 2026 |
| 31.7× / 26.2× | Throughput / cost gains, tf.data service disaggregation, input-bound jobs | arXiv:2210.14826 (SoCC'23) | PAPER | 2023 |
| +4.7%/+9.3%/+16.8%/+15.0%/-58.3% | Downstream gains from Best-fit Packing (reading comp/NLI/context-following/HumanEval/hallucination) | arXiv:2404.10830 | PAPER | 2024 |
| O(N log L), 60% runtime cut | OLMo-core Optimized-BFD packing vs. naive BFD, 1B-doc scale | olmo-core docs | OFFICIAL | 2025/2026 |
| Up to 6× faster to target accuracy | Apple Dataset Decomposition (VSL curriculum) | arXiv:2405.13226 | PAPER | 2024 |
| ~2% (parity) / 0.25% (viable at 100s-TB scale) | Shuffle-buffer-to-dataset-size ratio for near-full-shuffle quality | arXiv:2309.01640 | PAPER | 2023 |
| ~50 MB/s/core | HF "fast" tokenizer throughput (1GB text <20s, single CPU core) | HuggingFace docs | OFFICIAL | 2025/2026 |
| ~75–76M tokens/s | Derived compute-ceiling ingestion rate, 100k-GPU cluster, 200B-active MoE (cross-checked two ways) | Derived (this report) | DERIVED | 2026 |

---

## A.6 Data

### A.6.1 Crawler traffic & robots.txt economics

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| GPTBot 7.7% / ClaudeBot 5.4% / Bytespider 2.9% of all crawler requests | Cloudflare-observed request share | Cloudflare blog | OFFICIAL | 2025-06 (data thru 2025-05) |
| GPTBot +305% YoY, ClaudeBot -46% YoY, Bytespider -85% YoY | Crawler volume growth, May'24→May'25 | Cloudflare blog | OFFICIAL | 2025-06 |
| ClaudeBot ~19.8–20% of AI-bot traffic, #2 overall (+66% MoM surge) | June 2026 | Cloudflare Radar (secondary aggregation) | HEARSAY | 2026-06/07 |
| Googlebot 24.9–27.5% of AI-bot-classified traffic (down from 57.2% a year earlier) | Q2 2026 vs. Q2 2025 | Cloudflare Radar (secondary aggregation) | HEARSAY | 2026-07 |
| 52.3% of AI crawler requests = training purpose; 2.6% = user fetch | 28 days to 2026-06-22 | Cloudflare Radar | OFFICIAL | 2026-06 |
| Training crawl 79–82% of AI crawl volume (up from 72% in 2024) | Jan–Jul 2025 | Cloudflare blog | OFFICIAL | 2025-08 |
| Anthropic crawl-to-refer ratio 286,930:1 → 38,065:1 | Jan 2025 → Jul 2025 | Cloudflare blog | OFFICIAL | 2025-08 |
| OpenAI crawl-to-refer ratio 1,217:1 → 1,091:1 | Jan 2025 → Jul 2025 | Cloudflare blog | OFFICIAL | 2025-08 |
| GPTBot 569M fetches/mo; Googlebot 4.5B; Claude 370M; AppleBot 314M | Vercel network, one month | vercel.com blog | OFFICIAL | 2024-12 |
| ChatGPT/Claude 404 rate ~35%; Googlebot 404 rate 8.22% | Vercel network | vercel.com blog | OFFICIAL | 2024-12 |
| >28% of top C4 source domains fully robots.txt-disallowed | Domain audit, n=14,000 | arXiv:2407.14933 "Consent in Crisis" | PAPER | 2024-07 |
| ~45% of C4 data restricted by Terms of Service | Same audit | arXiv:2407.14933 | PAPER | 2024-07 |
| AI-blocking among reputable sites: 23%→~60% | Sept 2023 → May 2025 | Secondary aggregation of Cloudflare robots.txt scans | HEARSAY | 2025 |
| >2.5M sites fully disallow AI training; >1M Cloudflare customers activated AI blocking | As of Aug 2025 | Secondary aggregation | HEARSAY | 2025-08 |
| 2xx crawl success 80.5%→49.2%; 4xx blocks 10.2%→35.8% | Q2 2025 → Q2 2026 | Secondary aggregation of Cloudflare data | HEARSAY | 2026-07 |
| >1 billion HTTP 402 responses/day to AI crawlers | Cloudflare network-wide, Pay Per Crawl | Cloudflare / TechCrunch | OFFICIAL | 2026-07 |
| >3.8 million domains on Content Signals managed robots.txt | Cloudflare | Cloudflare blog | OFFICIAL | 2026 |
| >50% of AI crawler traffic re-fetches unchanged pages | Cited as rationale for Sept 2026 policy | Cloudflare / TechCrunch | OFFICIAL | 2026-07 |
| Common Crawl monthly snapshot: ~2.0–2.2B pages, ~350–400 TiB uncompressed | CC-MAIN-2026-21/25 | commoncrawl.org | OFFICIAL | 2026-05/06 |
| Cloudflare handles traffic for ~21.3% of all websites | Market-share context | Press | HEARSAY | 2026-01 |

### A.6.2 Data-stock scarcity & economics

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| ~300T tokens effective human-text stock (90% CI 100T–1000T) | Quality/repetition-adjusted | Epoch AI | HEARSAY | 2024-06 |
| Stock exhaustion 2026–2032 (80% CI) | Compute-optimal training trend continuation | Epoch AI | HEARSAY | 2024-06 |
| Multi-epoch repetition expands effective stock 2–5× | "Several epochs without significant degradation" | Epoch AI | HEARSAY | 2024-06 |
| 5×10^28 FLOP compute-optimal ceiling, ~2028 | Given data-stock constraint | Epoch AI | HEARSAY | 2024-06 |

### A.6.3 Licensing deals

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| Reddit–Google: $60M/yr | Feb 2024 deal | CBS News / press | HEARSAY | 2024-02 |
| Reddit–OpenAI: ~$70M/yr (estimate) | 2024 deal | Press (CJR) | HEARSAY | 2024 |
| News Corp–OpenAI: ~$250M/5yr, ~$50M/yr | May 2024 deal | OpenAI blog (existence) + press (terms) | OFFICIAL/HEARSAY | 2024-05 |
| Axel Springer–OpenAI: ~$13M/yr, 3yr | Dec 2023 | Press | HEARSAY | 2023-12 |
| Shutterstock Big Tech deals: $25–50M each | Per Shutterstock CFO | Press | HEARSAY | 2023 |
| T&F–Microsoft: $10M initial, ~$75M total (2 deals) | Informa subsidiary | Bookseller / Inside Higher Ed | HEARSAY | 2024-05/07 |
| Wiley: $23M one-time (deal 1), ~$44M total expected | Wiley AI partnerships | Books+Publishing | HEARSAY | 2024-09 |
| Anthropic book-scan throughput: up to 8,000 books / 2M pages per day | "Project Panama" internal docs (litigation discovery) | WaPo / Bookseller | HEARSAY | 2026-01 |
| Anthropic settlement: $1.5B, ~500,000 works, ~$3,000/work | Bartz v. Anthropic | NPR / multiple legal press | HEARSAY | 2025-08/09 |
| Meta (Kadrey record): ~82TB pirated books torrented | Discovery record | Court filings via press | HEARSAY | 2025 |
| Meta (new 2026 suit allegation): 267TB+ pirated material | Elsevier et al. v. Meta | Complaint via press | HEARSAY | 2026-05 |
| Whisper transcribed 1M+ hours of YouTube video for GPT-4-era training | NYT reporting | NYT (secondary aggregation) | HEARSAY | ~2024 |
| Whisper large-v3 training data: 1M hrs weak-labeled + 4M hrs pseudo-labeled | Whisper model card | OpenAI | OFFICIAL | 2023-11 |
| Mercor expert network: ~300,000 vetted experts | Company claim | Mercor / press | HEARSAY | 2025-2026 |
| Frontier labs: ~$1B/yr each on human-provided training data | Aggregate industry estimate | Multiple secondary sources | HEARSAY | 2026 |
| Non-commercial license (initial release) | Harvard Institutional Books 1.0 access terms | arXiv:2506.08300 | PAPER | 2025-06 |
| CC0 license | HPLT corpus licensing | arXiv:2503.10267 | PAPER | 2025-03 |

### A.6.4 Extraction & filtering pipelines

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| resiliparse/trafilatura beat WET extraction by ≥2.5 CORE points; resiliparse 8× faster | DCLM-Baseline extractor ablation | arXiv:2406.11794 (NeurIPS 2024) | PAPER | 2024-06 |
| Extractor union increases DCLM-Baseline token yield by up to 71% (58% post-dedup) | resiliparse+trafilatura+jusText union | arXiv:2602.19548 | PAPER | 2026-02 |
| Only 39% of pages survive >1 extractor; 61% unique-to-one | Same paper | arXiv:2602.19548 | PAPER | 2026-02 |
| AICC: MinerU-HTML ROUGE-F1 81.8% vs. trafilatura 63.6%; code preservation 90.9%, formula 94.0% | Model-based HTML extraction | arXiv:2511.16397 | PAPER | 2025-11 |
| AICC corpus: 7.3T tokens from 2 CC snapshots; +1.08pp avg benchmark gain vs. trafilatura baseline | Same paper | arXiv:2511.16397 | PAPER | 2025-11 |
| 96 CommonCrawl dumps, 15T tokens/44TB, 5–7% raw-to-final pass rate | FineWeb build via datatrove | HuggingFace FineWeb blogpost | OFFICIAL | 2024-05 |
| ~30% tokens removed | FineWeb terminal-punctuation filter alone | HuggingFace FineWeb blogpost | OFFICIAL | 2024-05 |
| 5.17 pages/sec (H100, LMDeploy+repetition-detection) | FinePDFs RolmOCR throughput | HuggingFace FinePDFs blog | OFFICIAL | 2025-09 |
| 66%/96% removed (base/EDU variant) | FinePDFs filtering yield | HuggingFace FinePDFs blog | OFFICIAL | 2025-09 |
| ~5.3k char median (2× typical), ~68k 95th-pctile | FinePDFs document length vs. HTML corpora | HuggingFace FinePDFs blog | OFFICIAL | 2025-09 |
| 110 H100-GPU-hrs verification vs. ~1,200 for full 1B training run; 80 CPUs×1,000 hrs to filter 15T tokens | Ultra-FineWeb efficient-verification cost | arXiv:2505.05427 | PAPER | 2025-05 |

### A.6.5 Curation/selection methods (relative gains)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| +1.0/+1.3/+2.5pp (1B/3B/7B) across 22 tasks | REWIRE recycled-reject-pile gains | arXiv:2506.04689 (COLM 2025) | PAPER | 2025-06 |
| 82% of synthetic mix sourced from rejected low-quality docs | REWIRE | arXiv:2506.04689 | PAPER | 2025-06 |
| 504B tokens (25% cut from 672B raw); +1.71/+7.61/+7.84pp vs. DCLM/FineWeb-Edu/Ultra-FineWeb | Darwin-CC (DataEvolve output) | arXiv:2603.14420 | PAPER | 2026-03 |
| Halves FLOPs-to-target vs. static reference-model selection | MATES (410M/1B on C4) | arXiv:2406.06046 (NeurIPS 2024) | PAPER | 2024-06 |
| +3.5–9.4% relative over random, 22 tasks | Group-MATES | arXiv:2502.14709 | PAPER | 2025-02 |
| 20% of RegMix's compute for same/better result; +2% avg over 9 benchmarks | TiKMiX-D / TiKMiX-M | arXiv:2508.17677 | PAPER | 2025-08 |
| Doubles convergence speed @1.3B; +3.23% downstream; holds to 7.2B | Meta-rater | arXiv:2504.14194 (ACL 2025) | PAPER | 2025-04 |
| +5.1pp vs. Cosmopedia, +2.6pp vs. Nemotron-Synth; 7.7×/2.7× speedups | BeyondWeb (DatologyAI) | arXiv:2508.10975 | PAPER | 2025-08 |
| 4–10× lower compute to match uniform-mixture baseline; 12/13 languages improved | ÜberWeb per-language curation, 20T-token corpus, 13 languages | arXiv:2602.15210 | PAPER | 2026-02 |
| >2% avg. gain, 10 benchmarks, 0.3B–1.7B models | ProX | arXiv:2409.17115 (ICML 2025) | PAPER | 2024-09 |
| 24×24 topic/format taxonomy, NMI≈0.10; +3.0pp mixing gain; 84% of FineWeb-Edu's gain recovered via mixing alone | WebOrganizer | arXiv:2502.10341 | PAPER | 2025-02 |
| +2.0% avg vs. Llama-3.2-1B @400B tok | CLIMB / ClimbMix (1B model) | arXiv:2504.13161 | PAPER | 2025-04 |
| ~30% optimal synthetic fraction; 5–10× speedup to matched val-loss | Rephrased-synthetic + real data mixing | arXiv:2510.01631 | PAPER | 2025-10 |
| Degradation within ~5 generations even at small synthetic fraction | Iterative recursive self-training collapse | arXiv:2410.04840 "Strong Model Collapse" | PAPER | 2024-10 |
| Only 368/1000 top domains overlap between DCLM-fastText and FineWeb-Edu classifiers | Classifier disagreement/complementarity | GneissWeb-adjacent survey | HEARSAY | 2025 |

### A.6.6 Deduplication

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| Removes ~50% data, ~1% compute cost, near-zero perf loss | SemDeDup | arXiv:2303.09540 | PAPER | 2023 (still ref. method 2025-26) |
| 53.2%/14.9%/18.7–19.1% removed | Dolma's 3-stage dedup (URL/doc/paragraph) | arXiv:2402.00159 + HF card | PAPER | 2024 |
| 56-way sharded fuzzy suffix-array dedup | Olmo 3 / Dolma 3 dedup method | arXiv:2512.13961 | PAPER | 2025-11/12 |
| 21.67% of bytes duplicated | ROOTS corpus, suffix-array substring dedup | arXiv:2303.03915 | PAPER | 2023 |
| O(log N) time, O(query_length) memory | Google deduplicate-text-datasets suffix-array query cost | google-research GitHub | CODE | — |
| 16× faster fuzzy dedup vs. alternatives; up to 80% data reduction | NeMo Curator claims | NVIDIA docs | OFFICIAL | 2024-26 |

### A.6.7 Named corpora (sizes, by scale)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 24T tokens / 23.6B docs | Essential-Web v1.0 total corpus | arXiv:2506.14111 | PAPER | 2025-06 |
| 0.5B params, within 3% of Qwen2.5-32B-Instruct, ~50× faster; ~90,000 MI300X GPU-hrs to label full corpus | EAI-Distill-0.5b classifier | arXiv:2506.14111 | PAPER | 2025-06 |
| 1.3T tokens final; 460,000 pages annotated (0–5 scale, threshold=3) | FineWeb-Edu | arXiv:2406.17557 | PAPER | 2024-06 |
| 240T raw pool → 2.6T used; 64% MMLU 5-shot @7B; +6.6pp at 40% less compute vs. MAP-Neo | DCLM-Baseline | arXiv:2406.11794 | PAPER | 2024-06 |
| 20TB / 5B docs, ~100 CC snapshots, scales to 1000+ languages; beats prior multilingual sets on 11/14 languages | FineWeb2 | arXiv:2506.20920 | PAPER | 2025-06 |
| 6.3T tokens total, 2T synthetic, 5 quality tiers; 59.0 vs. 53.4 (DCLM) vs. 42.9 (FineWeb-Edu) MMLU @8B/1T tok | Nemotron-CC v1 | arXiv:2412.02595 / NVIDIA blog | PAPER | 2024-12 |
| 70.3 vs. 65.3 (+5pp) MMLU | Llama-3.1 + Nemotron-CC continued pretrain @15T long-horizon tok | NVIDIA blog | OFFICIAL | 2024-25 |
| 6,585.8B tokens total (3,359.8B English-CC core) | Nemotron-CC v2 | NVIDIA/HF dataset card | OFFICIAL | 2025 |
| 6.6T tokens collection total | Nemotron-Pretraining-Dataset-v1 | NVIDIA/HF | OFFICIAL | 2025 |
| 133B tokens (Lynx+LLM pipeline) | Nemotron-CC-Math-v1 | HF dataset card | OFFICIAL | 2025 |
| 133B tokens (title figure; 52B cited in some secondary summaries, unresolved) | Nemotron-CC-Math | arXiv:2508.15096 | **[CONFLICT — flagged "needs verification" by the sourcing note]** | 2025-08 |
| 175.1B synthetic + 747.4B permissive-license | Nemotron-Pretraining-Code | HF dataset card | OFFICIAL | 2025 |
| ClimbLab 1.2T tok (20 clusters); ClimbMix 400B tok | CLIMB dataset releases | arXiv:2504.13161 | PAPER | 2025-04 |
| ~5T unique tokens; 15T+ upsampled | TxT360 | LLM360, COLM 2025 | PAPER | 2025 |
| 15T → ~10T tokens; +2.73pp vs. FineWeb 1.1.0 @7B/350B tok | GneissWeb | arXiv:2502.14907 | PAPER | 2025-02 |
| ~9.3T tokens total (5.9T Mix + Dolmino + Longmino) | Dolma 3 | AI2 | OFFICIAL | 2025-11 |
| 50B (7B model)/100B (32B model) tok | Dolma 3 Longmino (long-doc/olmOCR PDF content) | AI2 | OFFICIAL | 2025-11 |
| 8TB, 30 sources; Comma v0.1-1T/-2T (7B) | Common Pile v0.1 | arXiv:2506.05209 | PAPER | 2025-06 |
| ~2T tokens, 517M+ docs | Common Corpus (Pleias) | arXiv:2506.01732 | PAPER | 2025-06 |
| 5T tokens; 3wks→2days build via NeMo Curator, 2× lower TCO | Zyda-2 | arXiv:2411.06068 / Zyphra/NVIDIA blog | PAPER | 2024-11 |
| ~900B tokens (4× Stack v1) | The Stack v2 | HF blog | OFFICIAL | 2024 |
| 960B raw → ~730B final tokens, 607 languages | OpenCoder RefineCode | arXiv:2411.04905 | PAPER | 2024-11 |
| 14.7B tokens | OpenWebMath | — | — | 2023 |
| 54B tokens (FineMath-4+: 5.5× prior) | FineMath | HF blog | OFFICIAL | 2024-12 |
| 371.6B tokens (279B web + 28.1B code + 64.5B synthetic) | MegaMath | arXiv:2504.02807 (COLM 2025) | PAPER | 2025-04 |
| ~1T English + ~120B Chinese tokens; +3.61/+1.98pp (en/zh) vs. FineWeb/FineWeb-Edu | Ultra-FineWeb | arXiv:2505.05427 | PAPER | 2025-05 |
| 3T tokens, 475M docs, 1,733 languages, 3.65TB | FinePDFs | HuggingFace blog | OFFICIAL | 2025-09 |
| 242B tokens, 983,004 volumes, 400M pages | Harvard Institutional Books 1.0 | arXiv:2506.08300 | PAPER | 2025-06 |
| 6T tokens, model-centric (LLM-scored, not rule-based) | Seed-Coder | arXiv:2506.03524 | PAPER | 2025-06 |
| 35T tokens (35TB): 5.2TB Chinese + 22.5TB Nemotron-CC English, 4.5B CoT templates | CCI4.0 | arXiv:2506.07463 | PAPER | 2025-06 |
| 8T tokens/193 languages (v2); ~30T tokens (v3, ~half non-English) | HPLT v2/v3 | arXiv:2503.10267 + secondary v3 sourcing | PAPER | 2025-03 |
| 50/25/17/8% (general/math/code/multilingual) | Llama 3 final data mix | arXiv:2407.21783 | PAPER | 2024-07 |
| >30 trillion tokens (Meta's own figure) vs. "up to 40 trillion tokens" (HuggingFace paraphrase) | Llama 4 family (Scout+Maverick+Behemoth) pretraining mixture | Meta AI blog vs. HF `transformers` docs | **[CONFLICT]** | 2025-04 / accessed 2026-07 |

### A.6.8 Tokenization / byte-level

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| Up to 33% fewer tokens, +4.0pp avg (30 tasks incl. +8.2pp MMLU), 27% less inference compute | SuperBPE | arXiv:2503.13423 | PAPER | 2025-03 |
| Crossover at 30B/100B/200B bytes (2-stage/1-stage/space H-Net); 58.2% vs. 55.5% avg. downstream acc. (XL scale) | H-Net vs. BPE Transformer | arXiv:2507.07955 | PAPER | 2025 |
| 8B params, 4T training bytes; ~50% inference FLOP savings vs. Llama-3 | BLT original FLOP-controlled scaling study | arXiv:2412.09871 | PAPER | 2024-12 |
| 87–92% memory-bandwidth reduction (BLT-D-16) | BLT 2026 follow-up (BLT Diffusion/-S/-DV) | Secondary coverage of Meta+Stanford work | HEARSAY | 2026-05 |
| 32K actual vs. ≥216K optimal (7× under scaling-law optimum) | Llama2-70B vocabulary under-provisioning | arXiv:2407.13623 | PAPER | 2024-07 |
| 151,669 | Qwen3 tokenizer vocab size | arXiv:2505.09388 | PAPER | 2025-05 |

---

## A.7 Model Configs (2025–2026 corpus coverage)

### A.7.1 Chinese labs

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 671B total / 37B active, 256 routed+1 shared experts top-8, 61 layers, hidden 7168, 14.8T tokens | DeepSeek-V3 | arXiv:2412.19437 | PAPER | 2024-12 |
| k=2048 top-k selected tokens; 2.1B warmup + 943.7B sparse-stage tokens | DeepSeek-V3.2-Exp (DSA retrofit) | DeepSeek-V3.2-Exp report | PAPER | 2025-09 |
| 1.6T total/49B active (V4-Pro); 284B total/13B active (V4-Flash); 33T/32T tokens | DeepSeek-V4 | arXiv:2606.19348 | PAPER | 2026-06 |
| 1,000+ Ascend 910C chips, 1,500+ iterations, zero interruptions (post-training only) | DeepSeek-V4-Pro Chinese-domestic post-training milestone | Tom's Hardware/Gigazine | HEARSAY | 2026-06 |
| 1.04T total/32.6B active, 384 experts top-8, 15.5T tokens, 61 layers, hidden 7168, 64 attn heads, MLA, vocab 160K | Kimi K2 | arXiv:2507.20534 | PAPER | 2025-07 |
| 48B total/3B active, 3:1 KDA:MLA, 5.7T tokens | Kimi Linear | arXiv:2510.26692 | PAPER | 2025-10 |
| 235B total/22B active, 94 layers, 128 experts top-8, 36T tokens (3-stage: 30T+5T+long-ctx), vocab 151,669 | Qwen3 flagship | arXiv:2505.09388 | PAPER | 2025-05 |
| 80B total/~3B active, 512 experts (10 routed+1 shared), 3:1 Gated DeltaNet:Gated Attention, 15T tokens (of 36T corpus) | Qwen3-Next-80B-A3B | Alibaba Cloud blog | OFFICIAL | 2025-09 |
| 397B-A17B | Qwen3.5 flagship MoE | Secondary coverage | HEARSAY | 2026-02 |
| 355B total/32B active, 96 attn heads, 160 experts top-8, 23T tokens | GLM-4.5 | arXiv:2508.06471 | PAPER | 2025-08 |
| 106B total/12B active | GLM-4.5-Air | Secondary/Zhipu materials | HEARSAY | 2025-07 |
| 256+1 experts, 744B/40B active, 80 layers, DeepSeek Sparse Attention | GLM-5 | arXiv:2602.15763 | PAPER | 2026 |
| 456B total/45.9B active, 1 softmax block per 7 lightning blocks, 7.5T continual-pretrain tokens | MiniMax-M1 | arXiv:2506.13585 | PAPER | 2025-06 |
| 229.9B total/9.8B active, 62 layers, full attention (reversal from M1's hybrid), 29.2T tokens, 192K native context, 48Q/8KV heads, d_model 3072, vocab 200,064 | MiniMax-M2 | arXiv:2605.26494 | PAPER | ~2026 |
| 321B total (316B LLM+5B vision)/38B active, 61 layers | Step-3 | arXiv:2507.19427 | PAPER | 2025-07 |
| 1T total/51B active, 80 layers, 256 experts top-8+1 shared, 20T+ tokens | Ant Ling-1T | arXiv:2510.22115 | PAPER | 2025-10 |
| 256 experts top-8+1 shared, ~3.5% activation, >7× efficiency leverage vs. dense | Ant Ling 2.0 config | arXiv:2510.22115 | PAPER | 2025-10 |
| 389B total/52B active, 64 layers, 7T+1.5T synthetic tokens | Tencent Hunyuan-Large | arXiv:2411.02265 | PAPER | 2024-11 |
| 560B total/56B active, 128 layers, AMF/MF hybrid Mamba-Transformer, 16T tokens | Tencent Hunyuan-TurboS | arXiv:2505.15431 | PAPER | 2025-05 |
| 424B total/47B active (VL); 300B/47B (text-only) | Baidu ERNIE 4.5 flagship | ERNIE 4.5 Technical Report | PAPER | 2025-06 |
| <3% activation; 53.7% activated/35.8% total params elastic sub-network | Baidu ERNIE 5.0 (trillion-param ultra-sparse MoE) | arXiv:2602.04705 | PAPER | 2026-02 |
| 25T tokens | Xiaomi MiMo-7B pretraining | arXiv:2505.07608 | PAPER | 2025-05 |
| 309B total/15B active, 3-layer MTP, 27T tokens, FP8 mixed precision | Xiaomi MiMo-V2-Flash | arXiv:2601.02780 | PAPER | 2026-01 |
| 36B dense, 64 layers, 512K native context, 12T tokens | ByteDance Seed-OSS-36B | ByteDance Seed | OFFICIAL | 2025-08 |
| 718B total/39B active | PanGu Ultra MoE | arXiv:2505.04519 | PAPER | 2025-05 |
| 135B dense, 13.2T tokens | PanGu Ultra (dense variant) | arXiv:2504.07866 | PAPER | 2025 (Apr–May) |
| 30,000 Kunlunxin P800 cards, 64/cabinet | Baidu Kunlun P800 cluster (separate from ERNIE 4.5, unconfirmed tie) | 智东西 (zhidx.com) | HEARSAY | 2025-04 |

### A.7.2 US / other labs

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| ~2T total/288B active, 16 experts | Llama 4 Behemoth | Meta AI blog | OFFICIAL | 2025-04 |
| 117B total/5.1B active, MXFP4 MoE weights (4.25 bits/param), ~63GB footprint (vs. ~240GB BF16) | GPT-OSS-120B | OpenAI model card + HF | OFFICIAL | 2025-08 |
| >100,000 GPU cluster | GPT-4.5 training scale | "Pre-Training GPT-4.5" video, secondary transcription | HEARSAY | 2025-04 |
| "10×" effective/pretraining compute & capability vs. GPT-4 | GPT-4.5 | Video + contemporaneous press | HEARSAY | 2025-02/04 |
| 5.0e26 FLOP (Speculative) | Grok 4, Epoch's #1-ranked training run by estimated compute | Epoch AI dataset | HEARSAY | pub. 2025-07 |
| 3.8e26 FLOP (Likely) | GPT-4.5, #2, derived from ~187M H100-hours median, 20–40% MFU | Epoch AI dataset | HEARSAY | pub. 2025-02 |
| 3.5e26 FLOP (Likely) | Grok 3, #3, based on 80,000 H100s, ~3-month run | Epoch AI dataset | HEARSAY | pub. 2025-02 |
| 6.6e25 FLOP (Speculative, CI 2e25–2e26) | GPT-5, #4 | Epoch AI dataset | HEARSAY | pub. 2025-08 |
| 5.18e25 FLOP (Likely) | Llama 4 Behemoth (preview), 2T total params | Epoch AI dataset | HEARSAY | pub. 2025-04 |
| 3.3e24 FLOP (Confident — a rare high-confidence rating at frontier-adjacent scale) | DeepSeek-V3 | Epoch AI dataset | HEARSAY | pub. 2024-12 |
| No FLOP estimate available | Gemini 3 Pro, Claude Opus 4/4.1/4.5/4.7, Claude Sonnet 4/4.5, GPT-5.1 family | Epoch AI dataset (absence noted) | HEARSAY | pulled 2026-07-12 |
| 31.6B total/3.2B active (3.6B incl. embed), 52 layers, 128 routed+2 shared experts top-6, 25T tokens | Nemotron 3 Nano 30B-A3B | arXiv:2512.20848 | PAPER | 2025-12 |
| 120.6B total/12.7B active, 88 layers, 512 total experts top-22 (LatentMoE, ℓ=1024), 25T tokens | Nemotron 3 Super 120B-A12B | arXiv:2604.12374 | PAPER | 2026-04 |
| 550B total/55B active, 108 layers, 512 experts top-22 (LatentMoE, ℓ=2048), 20T tokens | Nemotron 3 Ultra 550B-A55B | arXiv:2606.15007 | PAPER | 2026-06 |
| 15T tokens (8B) / 20T tokens FP8 (56B) | Nemotron-H | NVIDIA ADLR | OFFICIAL | 2024-25 |
| 400B total/13B active, 256+1 experts top-4, 17T tokens, 2048×B300 GPUs | Arcee Trinity Large | arXiv:2602.17004 | PAPER | 2026-02 |
| 11.2T tokens, 3-stage curriculum (8T/2T/1.1T) | SmolLM3 | HuggingFace blog | OFFICIAL | 2025-07 |
| 30T tokens, 8,000 GB200 GPUs (human-data-only) | Microsoft MAI-Base-1 | microsoft.ai report | OFFICIAL | 2026-06 |
| 35B active/~1T total params, 256K context | Microsoft MAI-Thinking-1 | microsoft.ai report | OFFICIAL | 2026-06 |
| ~15,000 H100s pre/post-training cluster | Microsoft MAI-1-preview | Microsoft AI blog | OFFICIAL | 2025-08/09 |
| 111B params | Cohere Command A | Cohere technical report | PAPER | 2025-03 |
| 480B total/17B active (10B dense + 128×3.66B experts, top-2); <$2M, <3 months | Snowflake Arctic | Snowflake Arctic Cookbook | OFFICIAL | 2024 |
| 132B total/36B active, 12T+ tokens; 3,072 H100s, ~3 months | Databricks DBRX | Databricks blog | OFFICIAL | 2024-03 |
| 3,000 H200s, trained from scratch | Mistral Large 3 | Mistral product materials | OFFICIAL | 2026 |
| 13,800 GB300 GPUs, $830M debt | Mistral new France datacenter | Secondary reporting | HEARSAY | 2026-03 |
| "Tens of trillions of tokens" target | Reflection AI frontier model | Company/press | HEARSAY | 2026 |
| Falcon-180B: ≤4,096 GPUs, ~7M GPU-hrs, 3.5T tokens, <1 epoch | TII | HuggingFace blog | OFFICIAL | 2023-09 |
| Falcon2-11B: 1,024×A100-40GB, ~2 months, 5T+ tokens | TII | arXiv:2407.14885 | PAPER | 2024-07 |
| Falcon-H1: 0.5B→2.5T tokens ... 34B→18T tokens | TII hybrid Mamba-2+attention family | arXiv:2507.22448 | PAPER | 2025-07 |
| 10B, 1T tokens | INTELLECT-1 (Prime Intellect, decentralized) | arXiv:2412.01152 | PAPER | 2024-12 |
| 106B MoE, 90.8% AIME24 | INTELLECT-3 (centralized GPUs) | Secondary press | HEARSAY | ~late 2025 |
| 8.3B total/760M active, 2,048 GPUs | ZAYA1 (Zyphra/AMD/IBM) | arXiv:2511.17127 | PAPER | 2025-12 |
| 22T tokens | IBM Granite 4.0, GB200 NVL72 on CoreWeave | IBM research blog | OFFICIAL | 2025-10 |
| ~6.437T tokens cumulative across all phases | Marin 32B | marin-community GitHub | LOGBOOK | 2025 |

### A.7.3 Multimodal / architecture specifics

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| 47% MFU, 2016 H800, RoCE, seq4096, batch15120 | ERNIE-4.5-300B-A47B (text-only flagship) | ERNIE 4.5 Technical Report | PAPER | 2025-06 |
| 630M-param ViT encoder vs. 424B total backbone | ERNIE-4.5-VL-424B-A47B | ERNIE 4.5 Technical Report | PAPER | 2025-06 |
| 65,536 vocab, 8,192 image codebook; 1,024 tokens per 512×512 image | Chameleon unified text+image tokenizer | arXiv:2405.09818 | PAPER | 2024-05 |
| 2.9T text + 1.5T image-text (1.4B pairs) + 0.4T interleaved ≈ 4.8T | Chameleon stage-1 pretraining token mix | arXiv:2405.09818 | PAPER | 2024-05 |
| 67B/~1T/~1T/100B tokens (S0–S3), 256K native max context | Qwen3-VL 4-stage pretraining | arXiv:2511.21631 | PAPER | 2025-11 |
| 2.0T+0.1T/1.4T/0.6T/0.3T tokens (after 5.2T text-only base); 400M-param MoonViT vision encoder vs. 16B total MoE backbone | Kimi-VL 4-stage joint pretrain | arXiv:2504.07491 | PAPER | 2025-06 |
| 3.7× overall FLOPs savings (2.6× text, 5.2× image), 1.4B model, 4 text+4 image experts, 1T-token budget | MoMa mixed-modal MoE | arXiv:2407.21770 | PAPER | 2024-07 |
| 4,096→256 tokens (16×) SAM-stage→CLIP-stage compression | DeepSeek-OCR | arXiv:2510.18234 | PAPER | 2025-10 |
| ~97% decode accuracy at ≤10× compression; ~60% at ~20× compression | DeepSeek-OCR | arXiv:2510.18234 | PAPER | 2025-10 |
| 5:1 local:global sliding-window ratio, RoPE base 10k(local)/1M(global) | Gemma 3 | arXiv:2503.19786 | PAPER | 2025-03 |
| ~3:1 chunked:NoPE ratio, 8192-token chunks | Llama 4 iRoPE (Scout variant) | Meta AI blog | OFFICIAL | 2025 |

---

## A.8 Costs

### A.8.1 GPU rental / spot pricing

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| $1.70/hr → $2.35/hr (+40%) | H100 1-year rental price, Oct 2025 → Mar 2026 | SemiAnalysis | HEARSAY | 2026-04 |
| $2.40/hr (neocloud, 25th %ile) vs. $3.10/hr (hyperscaler) | B200 spot pricing snapshot | SemiAnalysis | HEARSAY | 2025-08 |
| $0.30–0.40/GPU-hour | H100 operating-cost floor (capex stripped) | SemiAnalysis | HEARSAY | 2025-11 |
| $2.40/hr spot H100 vs. ~$1.40/hr build-and-amortize cost | Spot-market margin evidence of GPU scarcity | Dylan Patel, Dwarkesh interview | SPOKEN | 2026-03 |
| $2.69/GPU-hr (GB300 NVL72) → $4.18/GPU-hr (Vera Rubin NVL72) | Total-system-cost basis | SemiAnalysis | HEARSAY | 2026-05 |
| ~$1.60/TPU-hour | Anthropic's estimated GCP TPUv7 rental cost | SemiAnalysis | HEARSAY | 2025-11 |
| ~$300,000/hour | Estimated opex for a 100,000-GPU cluster (not OpenAI-stated) | Secondary analysis | HEARSAY | 2026-05 |

### A.8.2 Named training-run costs

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| $5.576M total @$2/GPU-hr rental | DeepSeek-V3 pretraining cost | arXiv:2412.19437 | PAPER | 2024-12 |
| ~$1.3B | DeepSeek server CapEx, full fleet (not V3-run-specific) | SemiAnalysis | HEARSAY | 2025-01 |
| $1.95/M tokens, $29.1M total | Llama3-405B, 15T-token BF16 pretrain, 2,304×H100 | SemiAnalysis | HEARSAY | 2025-08 |
| 72¢→54.2¢/M tokens ($218k→$162k total) | GPT-3-175B 300B-token pretrain cost, FP8, Jan→Dec 2024 | SemiAnalysis | HEARSAY | 2025-08 |
| $4.5M → $2.5M (proj.) | DeepSeek-670B 14.8T-token full pretrain cost, GB200, Jul→Dec-2025 | SemiAnalysis | HEARSAY | 2025-08 |
| 100k GB200 @15% MFU, 50T tokens: 9.41 pure-compute days, $56.5–113M | Money-table synthesis | Notes' own calculation | DERIVED | 2026-07 |
| 100k GB200 @35% MFU, 25T tokens: 2.02 pure-compute days, $12.1–24.2M | Money-table synthesis | Notes' own calculation | DERIVED | 2026-07 |
| ~$0.5B | Estimated compute cost of a 6-month Orion/GPT-4.5-class run | WSJ via Fortune | HEARSAY | 2025-02 |
| ~$1B | Estimated cloud compute cost for a "GPT-5-scale" run (excl. datacenter capex) | WSJ via Fortune | HEARSAY | 2025-02 |
| >$2B | Estimated compute cost of OpenAI "Spud" (shipped as GPT-5.5) pretraining run | Aggregated tracker sites | HEARSAY | 2026 |
| ~$5M | DeepSeek V3 re-training cost estimate (efficiency-gain, not algorithmic-edge, explanation) | Trenton Bricken, Dwarkesh interview | SPOKEN | 2025 |
| 512×H800, 3 weeks, ~$534,700 | MiniMax-M1 full RL training run | arXiv:2506.13585 | PAPER | 2025-06 |
| $10–15B/GW/year; $50–75B over 5 yrs per GW; ~$35B of ~$50B is chips; Nvidia ~75% gross margin | GW-scale buildout unit economics | Dylan Patel, Invest Like the Best | SPOKEN | 2025-09 |
| ~9.6% | Share of OpenAI's total 2024 R&D compute (~$5B) consumed by the single headline pretraining run (~$500M) — implies ablation/experimentation tax ≈10× the headline-run cost | Epoch AI analysis | HEARSAY | fetched 2026-07-13 |
| 22.6% / 12.3% | Same final-run/total-R&D-compute ratio for MiniMax (Q4'24–Q3'25, $141M base) / Z.ai (H2'24–H1'25, $216M base) | Epoch AI analysis | HEARSAY | fetched 2026-07-13 |
| MPT-7B: 440×A100-40GB, 9.5 days, ~$200K | MosaicML pre-Databricks cost baseline | Databricks blog | OFFICIAL | 2023-05 |

### A.8.3 Compute deals ($ and GW commitments)

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| Up to 1,000,000 TPUs, "well over 1 GW" in 2026, "tens of billions of dollars" | Anthropic–Google Cloud TPU deal | Anthropic official blog | OFFICIAL | 2025-10-23 |
| 1M TPUs (400K bought ~$10B + 600K rented ~$42B RPO) | Same deal, financial breakdown | SemiAnalysis | HEARSAY | 2025-11 |
| 3.5 GW | Anthropic–Google–Broadcom follow-on capacity, starting 2027 | TechCrunch | HEARSAY | 2026-04-07 |
| Up to $40B / 5GW / up to 1M Ironwood chips | Bloomberg's cumulative framing of the Anthropic-Google relationship (doesn't match any single official figure) | Bloomberg | **[CONFLICT with above rows — different aggregations of the same relationship]** | 2026-05-18 |
| $100B over 10 years, up to 5 GW | Anthropic's AWS spend commitment | Anthropic official blog | OFFICIAL | ~2025 |
| ~500,000 → >1,000,000 Trainium2/3 chips (target, training+inference), end-2026 | Anthropic Project Rainier / AWS footprint | AWS official; Anthropic/Amazon official (updated figure supersedes earlier "~500K") | OFFICIAL | 2025-10 → 2026-04-20 |
| $8B | Amazon's cumulative equity investment in Anthropic since 2023 | Yahoo Finance/Amazon | HEARSAY | 2025-10 |
| $30B run-rate revenue (up from ~$9B end-2025) | Anthropic | TechCrunch, citing Anthropic | HEARSAY | 2026-04-07 |
| $380B valuation (off $30B Series G) | Anthropic | TechCrunch | HEARSAY | 2026-04-07 |
| $18–20B hardware cost; ~$7.5B equity (incl. up to $2B Nvidia) + $12–12.5B GPU-collateralized debt (Valor SPV) | xAI Colossus 2 financing | WSJ/The Information | HEARSAY | 2025-07-22 onward |
| 6GW total, 1GW MI450 first tranche H2 2026, 160M-share warrant | OpenAI–AMD deal | OpenAI/AMD official | OFFICIAL | 2025-10 |
| 10GW total, Jalapeño tape-out in 9 months, deployment late-2026–2029 | OpenAI–Broadcom deal | OpenAI/Broadcom official | OFFICIAL | 2025-10 / 2026-06 |
| 750MW, >$20B, multi-year (Trainium3=prefill, CS-3=decode) | Cerebras–OpenAI inference deal | Cerebras/OpenAI official | OFFICIAL | 2025-12-24 (signed) |
| $20B | Nvidia's payment to license Groq IP + hire team | SemiAnalysis | HEARSAY | 2026-03 |
| $50B | OpenAI's planned 2026 compute spend, disclosed via sworn court testimony (Musk v. OpenAI) | Bloomberg, sourced to courtroom testimony | HEARSAY | 2026-05-05 |
| ~$30M → "tens of billions" | OpenAI compute cost trajectory, 2017 → 2026, per same testimony | Bloomberg | HEARSAY | 2026-05-05 |
| $2B raise / $8B valuation | Reflection AI funding | TechCrunch | HEARSAY | 2025-10 |
| ~$30B valuation, ~49% stake ($14.3B) | Meta–Scale AI transaction | SemiAnalysis | HEARSAY | 2025-07 |
| $200M–$300M/4 years | Reported Meta Superintelligence Labs researcher compensation offers (up to 100× peer comp) | SemiAnalysis | HEARSAY | 2025-07 |
| $28.7B | Microsoft record quarterly capex, Q3 FY2026 | Aggregated financial press | HEARSAY | 2026 |
| $680–720B | Combined 2026 hyperscaler capex, industry-wide | Aggregated financial press | HEARSAY | 2026 |
| $1,700–2,000/kW, 18–36mo lead time | Aeroderivative gas-turbine capex (onsite power for GPU DCs) | SemiAnalysis | HEARSAY | 2025-12 |
| $3,000–4,000/kW | Bloom Energy fuel-cell capex | SemiAnalysis | HEARSAY | 2025-12 |
| 2.3 GW | OpenAI/Oracle onsite gas order, Texas | SemiAnalysis | HEARSAY | 2025-12 (order Oct 2025) |
| 64-node cluster ≈ $20–50M over a 3-year contract | Cloud contract economics | Stas repo | HEARSAY | 2026-07 |

### A.8.4 TCO / goodput-expense frameworks

| Value | Context | Source | Provenance | Date |
|---|---|---|---|---|
| GB200 rack list price ≈$3.1–3.9M all-in; H100 server $190k/$250k (bare/all-in); TCO 1.6–1.7× H100/GPU | H100 vs. GB200 NVL72 capex | SemiAnalysis (primary PDF) | HEARSAY | 2025-08 |
| ClusterMAX goodput expense: Gold 6.14%, Hyperscaler 10.53%, Silver 20.91% | Large-LLM-pretrain scenario | SemiAnalysis | HEARSAY | 2025-08 |
| 3-yr TCO ratio, Large LLM Pretrain (5,184×GB300): 1×/1.10×/1.15× (Gold/Hyperscaler/Silver) | Goodput Calculator | SemiAnalysis | HEARSAY | 2026-04 |
| 3-yr TCO ratio, Multimodal RL Research (2,048×B200): 1×/1.61×/1.15× | Goodput Calculator | SemiAnalysis | HEARSAY | 2026-04 |
| 25,000 GPU-hr (Gold/Hyperscaler) vs. 15,000 GPU-hr (Silver) | Assumed GPU-level MTBF inputs to Goodput Calculator | SemiAnalysis | HEARSAY | 2026-04 |
| ~44% lower (internal) / ~30–41% lower (external) | Google Ironwood all-in TCO vs. GB200/GB300 | SemiAnalysis | HEARSAY | 2025-11 |
| ~62% lower cost/effective training FLOP (list); ~52% lower TCO/effective PFLOP (Anthropic rented pricing) | TPU v7 Ironwood vs. GB300 | SemiAnalysis | HEARSAY | 2025-11 |
| ~15% MFU (Google internal) / ~19% MFU (Anthropic, GCP-priced) | TCO-breakeven MFU vs. GB300 (30% MFU baseline) | SemiAnalysis | HEARSAY | 2025-11 |
| 6-7 months backplane firmware-bug resolution before v1.3 stability | GB200 NVL72 reliability drag on realized TCO | SemiAnalysis ClusterMAX 2.0 | HEARSAY | 2025-11 |
| o1 vs. GPT-4o: 6× / o1-mini vs. GPT-4o-mini: 20× per-token price | Inference-pricing gap, same-size model pairs (context for RL/inference-compute economics) | SemiAnalysis | HEARSAY | 2024-12 |
