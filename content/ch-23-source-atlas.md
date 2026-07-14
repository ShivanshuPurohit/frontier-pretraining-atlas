# Chapter 23 — The Source Atlas: Annotated Bibliography and Surprise Index

Every claim in this book traces back to one of forty-two research notes, and every one of those notes traces back to a primary artifact — a paper, a repository, a technical report, a podcast transcript, a blog post written by a researcher who happened to say more than their employer's PR team would have liked. This closing chapter is that trail made explicit: an index into the 42-file, roughly 332,000-word corpus this book draws on, an annotated bibliography organized by the topics Chapters 1–22 cover, a ranked catalogue of the corpus's most surprising individual findings, and an honest accounting of what the research fleet could not pin down. Treat it as the book's footnotes, collected in one place and made navigable, rather than as new argument.

## Reading This Atlas

The provenance tags follow Chapter 1's convention exactly: [PAPER], [OFFICIAL-BLOG], [CODE], [BENCHMARK], [TALK], [DOC], [CREDIBLE-HEARSAY], [RUMOR], with dates preserved from the underlying notes wherever the notes give one. A ★ marks the 29 items this research judges most likely to be new to a reader who has already worked through the DeepSeek V3/V4 reports, the MAI-1 paper, and the standard scaling-law canon — items buried a citation-hop deep, published in the last few months of this book's research window, or sitting in a genre (a personal wiki, a maintainer's scratch notes, an unpublished thesis) that doesn't show up in a normal literature search. The Surprise Index that follows the bibliography is deliberately terse — one sentence, one pointer — because the underlying notes already argue each point at length; the pointer is the invitation to go read the argument in full.

## The Corpus: Forty-Two Notes

The research fleet worked in two rounds. Round one (notes 01–23) covers the load-bearing frameworks, the DeepSeek and Chinese-lab reference stacks, the architecture and scaling-law literature, precision, networking, fault tolerance, benchmarks, and a first pass through Stas Bekman's `ml-engineering` book. Round two (notes 30–53) fills the gaps round one's own critique pass surfaced: a dedicated GB200-FLOPS reconciliation, multimodal integration, Baidu/PaddlePaddle, the Llama 4 Behemoth failure, Nemotron 3's LatentMoE, three further Bekman-repo deep-dives, the companion `the-art-of-debugging` book, open-lab war stories, conference-archive mining, MoE systems papers, data-processing infrastructure, theory blogs, insider journalism, podcast mining, cluster reference architectures, non-Nvidia silicon, and historical training postmortems.

| Note | Topic | Words |
|---|---|---|
| [01-jax-scaling-book](../notes/01-jax-scaling-book.md) | DeepMind's "How to Scale Your Model" + HF Ultra-Scale Playbook; roofline algebra for TP/EP/DP | 6,319 |
| [02-megatron-lm](../notes/02-megatron-lm.md) | Megatron-Core's MoE stack per arXiv:2603.07685; Parallel Folding, ECHO, Paged Stashing | 6,709 |
| [03-torchtitan](../notes/03-torchtitan.md) | TorchTitan's five MoE dispatcher backends; real production usage (Arcee, AMD/TensorWave) | 5,325 |
| [04-maxtext-tpu-stack](../notes/04-maxtext-tpu-stack.md) | MaxText/JAX/Pallas stack; TPU v2→Ironwood hardware table; elastic-recovery demo | 5,864 |
| [05-deepseek-stack](../notes/05-deepseek-stack.md) | DeepSeek V3/V3.2/V4 infra in full: FP8 recipe, DualPipe, DeepEP, 3FS, implied MFU | 7,922 |
| [06-chinese-labs](../notes/06-chinese-labs.md) | Kimi, Qwen, GLM, MiniMax, Ling, StepFun, Hunyuan, ERNIE report comparison | 8,641 |
| [07-moe-architecture](../notes/07-moe-architecture.md) | Granularity, routing, sparsity laws, attention choice, MTP; the 10T/200B strawman | 8,181 |
| [08-scaling-laws-modern](../notes/08-scaling-laws-modern.md) | Step Law, Farseer, WSD laws, MoE laws, precision-aware laws, the ladder recipe | 8,127 |
| [09-data-curation](../notes/09-data-curation.md) | Curation SOTA past Essential-Web; synthetic rephrasing, dedup, group-influence selection | 7,277 |
| [10-crawling-data-supply](../notes/10-crawling-data-supply.md) | Crawler economics, CC decline, Cloudflare blocking, licensing, book-scanning litigation | 7,345 |
| [11-dataloading-storage](../notes/11-dataloading-storage.md) | Loader requirements matrix, packing, checkpoint I/O, determinism at 100k GPUs | 8,090 |
| [12-kernels-attention](../notes/12-kernels-attention.md) | FlashAttention-4, DeepGEMM, CUTLASS/CuTe, tcgen05, small-M MoE GEMM problem | 8,363 |
| [13-precision-numerics](../notes/13-precision-numerics.md) | FP8→MXFP8→NVFP4 recipes with evidence; router numerics; hardware ground truth | 9,465 |
| [14-networking-scale](../notes/14-networking-scale.md) | MRC primary source, NCCL evolution, RoCE vs. IB, multi-DC, failure math | 8,373 |
| [15-fault-tolerance](../notes/15-fault-tolerance.md) | Failure-rate tables, SDC, checkpoint math, in-memory replication at 100k GPUs | 10,082 |
| [16-mlperf-benchmarks](../notes/16-mlperf-benchmarks.md) | MLPerf Training ground truth; dense-vs-MoE MFU gap; the 58%-DeepSeek-MFU debunk | 6,534 |
| [17-frontier-lab-crumbs](../notes/17-frontier-lab-crumbs.md) | Every load-bearing 2024-2026 disclosure from the closed labs, lab by lab | 9,966 |
| [18-optimizers-stability](../notes/18-optimizers-stability.md) | Muon at scale, distributed NS, AdamW baseline, spike anatomy, QK-Clip | 8,567 |
| [19-su-jianlin-theory](../notes/19-su-jianlin-theory.md) | Su Jianlin's kexue.fm — Moonshot's de facto public research notebook | 7,772 |
| [20-midtraining-curriculum](../notes/20-midtraining-curriculum.md) | Anneal-phase practice, mixing laws, long-context staging, RPT boundary | 9,617 |
| [21-local-semianalysis-pdfs](../notes/21-local-semianalysis-pdfs.md) | Deep reads of 13 locally-archived SemiAnalysis PDF reports | 9,639 |
| [22-local-semianalysis-htmls](../notes/22-local-semianalysis-htmls.md) | Deep reads of 23 locally-archived SemiAnalysis HTML articles | 8,999 |
| [23-stas-ml-engineering](../notes/23-stas-ml-engineering.md) | First pass through Bekman's `ml-engineering`: MAMF table, silicon lottery | 6,265 |
| [30-gb200-nvl72-peak-flops-reconciliation](../notes/30-gb200-nvl72-peak-flops-reconciliation.md) | Resolves the GB200-vs-HGX-B200 FLOPS conflation propagating through the corpus | 5,806 |
| [31-multimodal-native-pretraining-integration](../notes/31-multimodal-native-pretraining-integration.md) | Modality-isolated vs. agnostic MoE routing; DeepSeek-OCR's compression framing | 6,504 |
| [32-baidu-ernie-paddlepaddle-infra](../notes/32-baidu-ernie-paddlepaddle-infra.md) | ERNIE 4.5's 47% MFU on PaddlePaddle; the one non-Megatron-lineage frontier lab | 6,758 |
| [33-llama4-behemoth-failure-postmortem](../notes/33-llama4-behemoth-failure-postmortem.md) | The Llama 4 Behemoth failure post-mortem, recovered from a misfiled local archive | 5,028 |
| [34-nemotron-3-architecture-primary-source](../notes/34-nemotron-3-architecture-primary-source.md) | Four primary sources on Nemotron 3 and LatentMoE, read in full | 7,748 |
| [40-stas-network-comms-debug](../notes/40-stas-network-comms-debug.md) | Bekman's NCCL benchmarking/debugging chapters; the busbw-undercounting bug | 8,120 |
| [41-stas-performance-parallelism-dtype](../notes/41-stas-performance-parallelism-dtype.md) | Bekman's MFU/HFU formulas, parallelism decision tree, dtype-zoo rules | 7,935 |
| [42-stas-storage-fault-orchestration](../notes/42-stas-storage-fault-orchestration.md) | Bekman's storage/orchestration chapters; the book's own agentic-maintenance runbook | 7,968 |
| [43-art-of-debugging](../notes/43-art-of-debugging.md) | Bekman's companion debugging book; coherent-memory and py-spy forensics | 5,523 |
| [44-open-lab-war-stories](../notes/44-open-lab-war-stories.md) | Imbue, Character.AI, IBM Vela, Databricks, Cohere, Falcon, Apple, LLM360 | 8,944 |
| [45-conference-archive-mining](../notes/45-conference-archive-mining.md) | GPU MODE, Hot Chips, OCP Summit, @Scale, CS336 — conference-only disclosures | 7,564 |
| [46-moe-systems-papers](../notes/46-moe-systems-papers.md) | COMET, MegaScale-MoE, PanGu Ultra MoE, XTuner V1, DeepSpeed AutoEP | 7,792 |
| [47-data-processing-infra](../notes/47-data-processing-infra.md) | FineWeb, NeMo Curator, Daft, dedup infrastructure, tokenizer-free architectures | 10,857 |
| [48-theory-blogs-frontier-thinking](../notes/48-theory-blogs-frontier-thinking.md) | Modular Manifolds, the Newhouse thesis, microarchitecture teardown blogs | 6,668 |
| [49-insider-journalism-hearsay](../notes/49-insider-journalism-hearsay.md) | WSJ/Bloomberg/Reuters reconstructions; Epoch AI's tracker and its blind spots | 8,243 |
| [50-podcast-primary-source-mining](../notes/50-podcast-primary-source-mining.md) | Dwarkesh, Latent Space, Training Data, Lex Fridman — lab principals unscripted | 8,125 |
| [51-cluster-reference-architectures](../notes/51-cluster-reference-architectures.md) | NVIDIA/Meta/AWS/Azure/GCP/OCI reference architectures, rack to fabric | 9,092 |
| [52-non-nvidia-frontier-training](../notes/52-non-nvidia-frontier-training.md) | Trainium, TPU, Ascend, AMD — the alternative-silicon verdict, evidence-graded | 10,892 |
| [53-historical-postmortems](../notes/53-historical-postmortems.md) | OPT, BLOOM, PaLM, GPT-NeoX, Pythia, OLMo, Marin, INTELLECT — the logbook genre | 8,998 |

## Annotated Bibliography

### Epistemics, the Closed Labs, and the Provenance Discipline (Ch1–2)

- Gangidi et al., ["RDMA over Ethernet for Distributed AI Training at Meta Scale"](https://rmiao.github.io/assets/pdf/sigcomm24-final246.pdf), SIGCOMM 2024. [PAPER] The canonical Meta RoCE paper; superseded in scale (not in mechanism) by the @Scale 2025 recap below.
- ★ Meta Engineering, "@Scale: Networking 2025" recap, [engineering.fb.com](https://engineering.fb.com), Sept 2025. [OFFICIAL-BLOG] Discloses the RoCE fabric's growth to 129,000 GPUs — larger than the SIGCOMM'24 figure most readers have anchored on, and not yet reflected in most secondary coverage.
- OpenAI et al., ["Resilient AI Supercomputer Networking using MRC and SRv6"](https://arxiv.org/html/2605.04333v1), arXiv:2605.04333, May 2026. [PAPER] The actual MRC primary source — not an OCP Summit talk, a 50-author engineering-blog-grade disclosure of OpenAI's production fabric.
- Epoch AI, [notable_ai_models.csv](https://epoch.ai/data/notable_ai_models.csv), pulled July 12, 2026. [EPOCH-ESTIMATE] The field's most methodologically transparent compute estimator — and, as of this pull, blind to Gemini 3 Pro, Claude Opus 4.x, and GPT-5.1.
- Epoch AI, ["Final training runs account for a minority of R&D compute spending"](https://epoch.ai/gradient-updates/r-and-d-vs-training-compute), fetched July 13, 2026. [EPOCH-ESTIMATE/ANALYSIS] The ~9.6%–22.6% final-run-share finding that reframes every "training cost" headline in the corpus.
- Dwarkesh Podcast, ["Jeff Dean and Noam Shazeer"](https://www.dwarkesh.com/p/jeff-dean-and-noam-shazeer), Feb 12, 2025. [SPOKEN-CLAIM] Confirms synchronous multi-metro pretraining is already routine at Google, gated purely by step-time budget.
- SemiAnalysis, ["Meta Superintelligence – Leadership, Compute, Talent, and Data"](https://newsletter.semianalysis.com/p/meta-superintelligence-leadership-compute-talent-and-data), Dylan Patel et al., July 11, 2025. [OFFICIAL-BLOG/PAID-NEWSLETTER] ★ Contains the Llama 4 Behemoth failure post-mortem; the fleet's initial passes cited its title without reading the body, which turned out to be sitting in the fleet's own local archive, misfiled among untitled HTML captures.

### The Open Frontier: China's Labs and DeepSeek (Ch3–4)

- DeepSeek-AI, ["DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437), arXiv:2412.19437, Dec 2024. [PAPER] The reference stack this entire corpus benchmarks against.
- DeepSeek-AI, ["Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"](https://arxiv.org/abs/2505.09343), arXiv:2505.09343, May 2025. [PAPER] ★ A wish list to NVIDIA/AMD that several vendors substantially granted within a year — the single most-skipped DeepSeek paper by readers who stop at the V3 report.
- DeepSeek-AI, ["DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"](https://arxiv.org/abs/2606.19348), arXiv:2606.19348, June 2026. [PAPER] Hybrid CSA/HCA attention, Muon, mHC — none of what the pre-release rumor mill predicted.
- Moonshot AI, ["Kimi K2: Open Agentic Intelligence"](https://arxiv.org/abs/2507.20534), arXiv:2507.20534, July 2025. [PAPER] The most infrastructurally transparent Chinese-lab report in the corpus.
- Moonshot AI, [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine), GitHub. [CODE] ★ Broadcasts a full 1T-parameter weight update across thousands of GPUs in roughly 20 seconds — a concrete number for an RL weight-sync problem most labs mention only qualitatively.
- Ant Group/Ling Team, ["Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation"](https://arxiv.org/abs/2510.22115), arXiv:2510.22115, Oct 2025, and ["Towards Greater Leverage"](https://arxiv.org/abs/2507.17702), arXiv:2507.17702, July 2025. [PAPER] ★ The richest infra disclosure in the whole Chinese-lab beat, including a full MFU waterfall (16.9%→32.8%) with each technique isolated.
- MiniMax, ["Why did M2 end up as a full attention model?"](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model), Nov 2025. [OFFICIAL-BLOG] A rare, quantified, named postmortem on abandoning an architectural bet (linear attention) after shipping it.
- ERNIE Team, Baidu, ["ERNIE 4.5 Technical Report"](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf), June 29, 2025. [PAPER] The one frontier Chinese lab not building on the Megatron-LM lineage; 47% MFU on 2,016 H800s via PaddlePaddle.
- LMSYS, ["Squeezing 1TB Model Rollout into a Single H200"](https://lmsys.org/blog/2026-01-26-int4-qat), Jan 2026. [OFFICIAL-BLOG] Independent reproduction of Kimi K2 Thinking's INT4 QAT within roughly two months of disclosure.

### MoE Architecture, Scaling Laws, and Optimizer Theory (Ch5–7)

- Elango et al., ["LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"](https://arxiv.org/abs/2601.18089), arXiv:2601.18089, Jan 2026. [PAPER] ★ A full roofline derivation with named GB200 constants, sitting one citation-hop from the Megatron-Core MoE report and unread by any prior pass.
- NVIDIA, ["Nemotron 3 Nano/Super/Ultra"](https://arxiv.org/abs/2512.20848) technical reports, Dec 2025–June 2026. [PAPER] ★ LatentMoE in actual production, plus two undiagnosed training-divergence incidents disclosed with MaxVio/residual-norm precursor signals — a genuinely actionable monitoring recipe.
- Abnar et al. (Apple), ["Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models"](https://arxiv.org/abs/2501.12370), arXiv:2501.12370, 2025. [PAPER] Optimal sparsity trending toward 1, no observed reversal even at the lowest tested activation ratio.
- StepFun, ["Predictable Scale Part I: Step Law"](https://arxiv.org/abs/2503.04715), arXiv:2503.04715, and ["Part II: Farseer"](https://arxiv.org/abs/2506.10972), arXiv:2506.10972. [PAPER] The largest public hyperparameter-scaling sweep to date, at roughly a third of DeepSeek-V3's entire pretraining GPU-hour budget.
- Su Jianlin, [kexue.fm](https://kexue.fm), ongoing. [BLOG] ★ Moonshot's de facto public research notebook — QK-Clip's MaxLogit mechanism, a reverse-engineering of DeepSeek-V4's hash-routing table, months ahead of any English-language coverage.
- Bernstein, ["Modular Manifolds"](https://thinkingmachines.ai/blog/modular-manifolds/), Thinking Machines Connectionism, Sept 26, 2025. [OFFICIAL-BLOG] ★ The direct theoretical continuation of Muon, constraining weight matrices to the Stiefel manifold.
- Newhouse, ["Duality, Weight Decay, and Metrized Deep Learning"](https://www.lakernewhouse.com/thesis.pdf), MIT MEng thesis, May 2025. [THESIS] ★ Unpublished-outside-MIT; quantifies Muon's Newton-Schulz "noise floor" and how vacuous formal Lipschitz guarantees become at competitive model quality.
- "Fantastic Pretraining Optimizers and Where to Find Them," [arXiv:2509.02046](https://arxiv.org/abs/2509.02046), Sept 2025, and its "Hyperball" rebuttal, [arXiv:2606.16899](https://arxiv.org/abs/2606.16899), June 2026. [PAPER] An adversarial re-benchmark of nearly every optimizer-speedup claim in circulation, followed by a dated counter-rebuttal.

### Precision and Numerics (Ch8)

- NVIDIA, ["Pretraining Large Language Models with NVFP4"](https://arxiv.org/html/2509.25149v2), arXiv:2509.25149, Sept 2025. [PAPER] The reference 2-level-scaling, Random-Hadamard NVFP4 recipe, validated to 12B/10T tokens.
- Kimi K2 team, community discussion, [huggingface.co/moonshotai/Kimi-K2-Instruct/discussions/30](https://huggingface.co/moonshotai/Kimi-K2-Instruct/discussions/30). [CREDIBLE-HEARSAY] Explicit rejection of FP8 pretraining compute for a 1T-parameter MoE after a stability study — the sharpest counter-example to "FP8 pretraining is solved."
- "Four Over Six," [arXiv:2512.02010](https://arxiv.org/html/2512.02010v3), Dec 2025. [PAPER] Dual-scale NVFP4 fix closing 22% of the loss gap for under 15% overhead, validated on Nemotron-3-Nano.
- ★ mufeezamjad, ["NVFP4 Grouped GEMM"](https://mufeezamjad.com/blog/nvfp4-group-gemm), 2026. [BLOG][CREDIBLE-HEARSAY] A 10x gap between naive and hand-optimized NVFP4 grouped GEMM on B200, with the exact architectural reason (TMEM capacity forcing 1 CTA/SM) spelled out — the sharpest public treatment of the "small-M problem" in the whole corpus.

### Multimodal-Native Pretraining (Ch9)

- Baidu, ["ERNIE 5.0 Technical Report"](https://arxiv.org/abs/2602.04705), arXiv:2602.04705, Feb 2026. [PAPER] Explicitly reverses ERNIE 4.5's own modality-isolated routing within eight months, naming the prior model as the design it's abandoning.
- Zhou et al. (Meta FAIR), ["MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts"](https://arxiv.org/abs/2407.21770), arXiv:2407.21770, July 2024. [PAPER] The actual origin of modality-isolated MoE, predating ERNIE's more commonly credited version.
- DeepSeek-AI, ["DeepSeek-OCR: Contexts Optical Compression"](https://arxiv.org/abs/2510.18234), arXiv:2510.18234, Oct 2025. [PAPER] Frames vision tokens as a general context-compression primitive, not an OCR trick — directly relevant to long-context token-budget planning.

### The Data Supply Chain and Curation (Ch10–11)

- Essential AI, ["Essential-Web v1.0"](https://arxiv.org/abs/2506.14111), arXiv:2506.14111, June 2025. [PAPER] The reference curation architecture: 24T tokens, a 12-category taxonomy, an 0.5B distilled classifier.
- ★ Li et al. (Apple/Stanford/UW), ["Beyond a Single Extractor: Re-thinking HTML-to-Text Extraction for LLM Pretraining"](https://arxiv.org/abs/2602.19548), arXiv:2602.19548, Feb 2026. [PAPER] Only 39% of pages survive post-curation filtering across more than one extractor — extractors are complementary, not redundant, and running the union yields a 50–70% token-yield increase.
- ★ Ma et al. (Shanghai AI Lab), ["AICC: Parse HTML Finer, Make Models Better"](https://arxiv.org/abs/2511.16397), arXiv:2511.16397, Nov 2025. [PAPER] A 0.6B sequence-labeling extractor beating trafilatura by 18 ROUGE-F1 points.
- ★ GAIR-NLP, "Data Darwinism Part II: DataEvolve," [arXiv:2603.14420](https://arxiv.org/abs/2603.14420), March 2026. [PAPER] An LLM literally writing and grading its own data-cleaning prompts across 30 generations — the most literal instantiation of "data agents" in the corpus, with a public repo and dataset.
- ★ "Introspective X Training," [arXiv:2605.20285](https://arxiv.org/abs/2605.20285), May 2026. [PAPER] Metadata/feedback-conditioning experiments at 7.5–12B models over 18 trillion tokens seen — an order of magnitude past the technique's earlier published scale, and almost certainly unseen by anyone on a pre-2026 reading list.
- DatologyAI, ["BeyondWeb"](https://datologyai.com/blog/beyondweb), Aug 2025. [OFFICIAL-BLOG] Real numbers from a paid vendor with zero independent replication found anywhere in this research — cite the direction, not the multiplier.

### Midtraining and Curriculum (Ch12)

- Allen Institute for AI, ["2 OLMo 2 Furious"](https://arxiv.org/abs/2501.00656), arXiv:2501.00656, and ["OLMo 3"](https://arxiv.org/abs/2512.13961), arXiv:2512.13961. [PAPER] The single richest primary sources in the beat — full Dolmino-mix tables, souping ablations, the special-chat-token GSM8K landmine.
- Microsoft AI, ["MAI-Thinking-1: Building a Hill-Climbing Machine"](https://microsoft.ai), June 2026. [OFFICIAL-BLOG] Explicitly excludes AI-generated text from its 30T-token base corpus — a live, frontier-scale disagreement with the DatologyAI/Nemotron synthetic-data-scaling consensus.

### Dataloading and the Input Pipeline (Ch13)

- ★ VAST Data, ["Optimizing Checkpoint Bandwidth for LLM Training"](https://blog.vastdata.com/optimizing-checkpoint-bandwidth-for-llm-training), 2026. [OFFICIAL-BLOG, vendor] An 85,000-checkpoint, 40-production-run telemetry survey showing real global checkpoint bandwidth tops out at 50–200GB/s — a data point that should recalibrate any naive per-parameter checkpoint-storage sizing.
- Li et al., ["MegaScale-Data: Scaling Dataloader for Multisource Large Foundation Model Training"](https://arxiv.org/abs/2504.09844), EuroSys 2026. [PAPER] Documents a 3.2–6.9x per-microbatch FLOP imbalance from dataloader composition alone.

### Hardware Ground Truth: GB200 and the Alternative-Silicon Question (Ch14)

- NVIDIA "Blackwell" datasheet, ★ [document #3384703](https://www.openzeka.com/wp-content/uploads/2025/02/blackwell-datasheet.pdf), Dec 2024. [OFFICIAL-DOCS] The single authoritative side-by-side GB200 NVL72 vs. HGX B200 spec table (5,000 vs. 4,500 TFLOPS/GPU dense FP8; 186GB vs. 180GB HBM3e) that resolves a FLOPS conflation this book's own research fleet found propagating into two of its own earlier notes; NVIDIA's own CDN copy 403'd on direct fetch, so this points to the mirror the fleet actually retrieved and verified against.
- ★ Glenn K. Lockwood, ["NVIDIA B200"](https://glennklockwood.com/garden/processors/b200). [CREDIBLE-HEARSAY] An independently-compiled, footnoted, frequently-updated technical wiki that cross-validates cleanly against the official datasheet and is not indexed by generic tech-news search.
- Jouppi et al. (Google), ["Google's Training Supercomputers from TPU v2 to Ironwood"](https://arxiv.org/abs/2606.15870), arXiv:2606.15870, June 2026. [PAPER] The single most quantified cross-generation hardware disclosure any lab has given the field, including the Ironwood Hardware Replay Unit's zero-overhead SDC detection.
- ★ Zyphra/AMD/IBM, ["Training Foundation Models on a Full-Stack AMD Platform"](https://arxiv.org/abs/2511.17127), arXiv:2511.17127. [PAPER] By far the richest 2025–2026 AMD hard-numbers source, unusually candid about the platform's own limitations.
- AWS Neuron docs, [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com) and [Trainium3](https://awsdocs-neuron.readthedocs-hosted.com) architecture pages, and AWS, ["Announcing Neuron Agentic Development"](https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-agentic-development), April 2026. [OFFICIAL-BLOG] AWS's own written admission that its kernel language is still too hard for most ML engineers, three SDK generations in.

### Kernels for Blackwell (Ch15)

- Dao et al., ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, March 2026. [PAPER] The primary FA4 source; verified in the corpus's own code inspection to have a forward-only FP8 path as of July 2026.
- DeepSeek-AI, [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP), and [EPLB](https://github.com/deepseek-ai/EPLB), GitHub. [CODE] The MoE-kernel reference implementations most Blackwell-generation frameworks build against.

### Parallelism Math and Frameworks (Ch16–17)

- Tazi, Mom, Zhao et al. (HuggingFace), the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), 2025. [OFFICIAL-BLOG] Over 4,000 scaling experiments on up to 512 GPUs; the DeepSeek routing-locality constraint the scaling book's own EP section omits.
- ★ Arcee AI, ["Trinity Large Technical Report"](https://arxiv.org/abs/2602.17004), arXiv:2602.17004, Feb 2026. [PAPER] An unusually candid production post-mortem — a forced MXFP8→BF16 fallback and six concurrent stability interventions on a real ~400B-total MoE, the single best "is this actually production grade" data point in the frameworks beat.
- NVIDIA, ["Scalable Training of Mixture-of-Experts Models with Megatron Core"](https://arxiv.org/abs/2603.07685), arXiv:2603.07685, March 2026. [PAPER] The canonical Megatron-Core MoE reference, superseding scattered blog posts.

### Networking at 100k (Ch18)

- Si et al. (Meta), ["Collective Communication for 100k+ GPUs" (NCCLX)](https://arxiv.org/abs/2510.20171), arXiv:2510.20171, Jan 2026. [PAPER] The richest source on production 100K+-GPU collectives engineering found anywhere in the corpus.
- ★ arXiv, [2605.04333](https://arxiv.org/html/2605.04333v1) — see the epistemics section above; also load-bearing here as the primary source for OpenAI's actual scale-out design.
- Qian et al. (Alibaba Cloud), ["Alibaba HPN"](https://ennanzhai.github.io/pub/sigcomm24-hpn.pdf), SIGCOMM 2024. [PAPER] Quantifies multi-chip chassis switches failing 3.77x more often than single-ASIC designs — the concrete reason the industry converged on single-ASIC fabrics.

### Fault Tolerance and Checkpointing (Ch19)

- ★ "Training LLMs with Fault Tolerant HSDP on 100,000 GPUs," [arXiv:2602.00277](https://arxiv.org/abs/2602.00277), Feb 2026. [PAPER] The single best-matched primary source for this beat, giving exact 100K-GPU failure rates and recovery-stall numbers; not yet widely cited elsewhere as of this writing.
- Meta, ["The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783), arXiv:2407.21783, July 2024. [PAPER] The 419-unexpected-interruption/54-day ground truth every later fault-tolerance paper in the corpus benchmarks against.
- ★ Hu et al. (Shanghai AI Laboratory), ["Characterization of Large Language Model Development in the Datacenter"](https://arxiv.org/abs/2403.07648), NSDI 2024 (the Acme trace). [PAPER] Median job duration in a real frontier-adjacent datacenter is two minutes, not hours — 92.9% of jobs are tiny eval jobs, a scheduler-design fact easy to lose sight of while planning for the pretraining tail.
- ★ Alibaba, ["Boosting Large-Scale Parallel Training Efficiency with C4: A Communication-Driven Approach"](https://arxiv.org/abs/2406.04594), arXiv:2406.04594, Jun 2024 (venue/author list not independently confirmed in the research corpus). [PAPER] A 30x reduction in error-induced downtime (31.19%→1.16%) came almost entirely from cutting detection latency, not from any change in underlying hardware failure rate.
- ★ "Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement," [arXiv:2407.04656](https://arxiv.org/abs/2407.04656). [PAPER] 20–40-second failure-to-resume by exploiting MoE's structurally redundant, independently-placeable experts — reliability engineering as an argument for MoE distinct from its usual compute-efficiency framing.

### Operations, War Stories, and Historical Postmortems (Ch20)

- Stas Bekman, ★ [ml-engineering](https://github.com/stas00/ml-engineering) and ★ [the-art-of-debugging](https://github.com/stas00/the-art-of-debugging), GitHub, both repos actively maintained through July 2026. [CODE/PRIMARY-SOURCE] The single densest practitioner-authored primary source in the entire corpus — a MAMF achievable-efficiency table, a fully-worked GB200 NVLink-C2C duplex measurement at 38% of spec, and (per its own maintainer runbook) evidence the book is itself agent-maintained.
- Meta AI, ["OPT-175B Logbook"](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles), 2021–2022, and BigScience, ["BLOOM chronicles"](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md), 2022. [LOGBOOK] ★ The two canonical day-by-day training logbooks the entire genre is measured against — and, per note 53, a genre that has essentially vanished from frontier commercial labs since.
- Imbue, ["70B infrastructure"](https://imbue.com/blog/70b-infrastructure), June 2024. [OFFICIAL-BLOG] The most complete public "how do you actually bring up a 500-node IB fabric" gauntlet available, with exact MFU signatures per failure class.
- Character.AI, ["Squinch"](https://blog.character.ai/squinch), 2024. [OFFICIAL-BLOG] A rare admission of bandwidth-constrained (not reliability-constrained) training, and a custom 6-bit gradient compressor built specifically to make FSDP viable under that constraint.

### Benchmarks and the Money Math (Ch21)

- NVIDIA, [Megatron-Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html), 26.06 container. [OFFICIAL-DOCS] The primary source for the dense-vs-MoE MFU gap table — 32–39% dense vs. 9–18% MoE, same framework, same hardware, three GPU generations.
- MLCommons, [MLPerf Training v6.0 results](https://mlcommons.org/2026/06/mlperf-training-v6-0-results), June 16, 2026. [BENCHMARK] The first standardized, third-party-audited large-MoE training benchmark — and the round where NVIDIA, in its own words, was "the only platform to submit on every test."

## The Surprise Index

The 36 individual findings across this corpus most likely to be new to a reader who has already internalized DeepSeek V3/V4 and the MAI-1 report, ranked roughly by how far they sit from the standard reading list rather than by topic.

1. NVIDIA's own GB200 NVL72 product page reports 2:4 structured-sparse FLOPS without labeling them as such — the scaling book that calls out this exact trick elsewhere doesn't flag it for Blackwell. → [01-jax-scaling-book](../notes/01-jax-scaling-book.md)
2. ECHO (Elastic Cloning for Hot Experts) dynamically clones overloaded experts to spare capacity slots specifically to make CUDA-Graph static buffers tractable for dropless MoE — merged into Megatron-Core in February 2026 and barely discussed outside its source paper. → [02-megatron-lm](../notes/02-megatron-lm.md)
3. TorchTitan ships a dedicated `HybridEPTokenDispatcher` built for "GB200/NVLink72" in core, not an experiments folder. → [03-torchtitan](../notes/03-torchtitan.md)
4. MaxText's own GPU support is frozen at a single `"a3"` (H100) entry as of June 2026 — zero A4X/GB200/B200 config entries, despite Google Cloud selling A4X since 2025. → [04-maxtext-tpu-stack](../notes/04-maxtext-tpu-stack.md)
5. DeepSeek's V3 paper states its own comm budget could support 13 activated experts instead of the 8 it actually uses — the routing topology, not the token compute, is the binding constraint, and the lab deliberately leaves slack on the table. → [05-deepseek-stack](../notes/05-deepseek-stack.md)
6. Ling-1T's MFU optimization waterfall (16.9%→32.8%) isolates each technique's individual contribution — nearly unique in a corpus where most labs report only a single final number, if any. → [06-chinese-labs](../notes/06-chinese-labs.md)
7. DeepSeek-V4 shipped with none of what the pre-release rumor mill predicted (no "Engram," no "MODEL1") — the real innovations are a CSA/HCA hybrid-compression attention scheme and Sinkhorn-projected hyper-connections. → [07-moe-architecture](../notes/07-moe-architecture.md)
8. StepFun's Step Law hyperparameter-scaling sweep cost roughly a third of DeepSeek-V3's entire pretraining GPU-hour budget just to fit the laws — "fitting the ladder" is no longer meaningfully cheaper than training a serious mid-scale model. → [08-scaling-laws-modern](../notes/08-scaling-laws-modern.md)
9. NVIDIA's NeMo Curator dedupes 1.96 trillion tokens in 30 minutes on 32 H100s — deduplication of an entire multi-trillion-token corpus is a same-afternoon operation, not a pipeline bottleneck. → [09-data-curation](../notes/09-data-curation.md)
10. Common Crawl's own bot is now blocked almost as often as OpenAI's dedicated GPTBot (5.08% vs. 5.52% of DISALLOW rules) — "just use CC" no longer sidesteps the site-blocking fight it's supposed to avoid. → [10-crawling-data-supply](../notes/10-crawling-data-supply.md)
11. Best-Fit Packing's padding overhead measures 0.0006%–0.003% — functionally zero, closing a "live tradeoff" most secondary literature still treats as open. → [11-dataloading-storage](../notes/11-dataloading-storage.md)
12. FlashAttention-4's FP8 path is forward-only as of July 2026, verified directly in source (`NotImplementedError`) rather than inferred from documentation — anyone planning FP8 attention training on Blackwell needs a fallback. → [12-kernels-attention](../notes/12-kernels-attention.md)
13. Kimi K2's team explicitly rejected FP8 pretraining compute for a 1T-parameter MoE after a preliminary stability study — a named, citable counter-example to the "FP8 pretraining is fully solved" narrative most 2025–2026 commentary carries. → [13-precision-numerics](../notes/13-precision-numerics.md)
14. Meta runs its entire 24K+-GPU RoCE backend with DCQCN turned off, relying purely on PFC plus receiver-driven admission control in the collective library itself — directly contradicting the "DCQCN is mandatory" assumption in most vendor documentation. → [14-networking-scale](../notes/14-networking-scale.md)
15. NVLink Paladin HD connectors are rated for roughly 200 mating cycles — a hard mechanical wear-out cap on how many times a GB200 NVL72 backplane can be serviced over a multi-year deployment. → [15-fault-tolerance](../notes/15-fault-tolerance.md)
16. The widely-circulated "58% MFU for DeepSeek-V3" claim does not survive reconstruction from the paper's own stated GPU-hours and token count. → [16-mlperf-benchmarks](../notes/16-mlperf-benchmarks.md)
17. Google's April 2026 Virgo Network/TPU 8t-8i disclosure claims JAX+Pathways can orchestrate more than one million TPU chips as a single logical training cluster — a step-change in stated ambition three months old at the time of this research. → [17-frontier-lab-crumbs](../notes/17-frontier-lab-crumbs.md)
18. DeepSeek switched from AdamW (V3) to Muon (V4) between December 2024 and June 2026 — the strongest revealed-preference signal in the entire optimizer literature, from the lab with the best public track record for MoE stability. → [18-optimizers-stability](../notes/18-optimizers-stability.md)
19. Su Jianlin's blog reverse-engineers DeepSeek-V4's tid2eid hash-routing construction algorithm from a leaked checkpoint artifact — the earliest accessible technical description of that specific V4 design choice in any language. → [19-su-jianlin-theory](../notes/19-su-jianlin-theory.md)
20. Microsoft's MAI-Thinking-1 (June 2026) actively filters AI-generated text out of its 30T-token base pretraining corpus at genuinely frontier scale (8,000 GB200s) — a live, unresolved strategic disagreement with the DatologyAI/Nemotron synthetic-data-scaling consensus. → [20-midtraining-curriculum](../notes/20-midtraining-curriculum.md)
21. A SemiAnalysis contractor spent roughly $10,000 in agentic (Claude/Codex) compute fuzzing NVIDIA's closed-source `ptxas` compiler and found 40 distinct miscompiling programs in three days — one agent-discovered bug silently converts an atomic store into two non-atomic stores. → [21-local-semianalysis-pdfs](../notes/21-local-semianalysis-pdfs.md)
22. AWS states outright that NVIDIA's own GB200/InfiniBand backplane reliability saga — which took six to seven months and a firmware fix to resolve — directly motivated Trainium3's redundant-lane, hot-swap-without-drain design. → [22-local-semianalysis-htmls](../notes/22-local-semianalysis-htmls.md)
23. Stas Bekman's own `nvbandwidth` measurement on a DGX Station puts real-world NVLink-C2C duplex efficiency at 38% of the marketed 900GB/s spec, bottlenecked by Grace's LPDDR5X bandwidth — not published anywhere by NVIDIA. → [23-stas-ml-engineering](../notes/23-stas-ml-engineering.md)
24. The JAX Scaling Book contradicts itself within the same document — its main FLOPS table and its own Quiz Question 6 cite two different Blackwell SKUs under one undifferentiated "B200" label, and the resulting number gets copy-pasted, uncorrected, into a second note in this same corpus. → [30-gb200-nvl72-peak-flops-reconciliation](../notes/30-gb200-nvl72-peak-flops-reconciliation.md)
25. ERNIE 5.0 explicitly reverses ERNIE 4.5's own modality-isolated MoE routing within eight months, naming its own prior model as the design it is moving away from. → [31-multimodal-native-pretraining-integration](../notes/31-multimodal-native-pretraining-integration.md)
26. Baidu's router-orthogonalization loss bypasses the ordinary loss function entirely, patching directly into the Adam update instead — "analogous to AdamW," in the paper's own words — so a naive reimplementation that just adds the term to the loss would get it measurably wrong. → [32-baidu-ernie-paddlepaddle-infra](../notes/32-baidu-ernie-paddlepaddle-infra.md)
27. The single richest technical account of a frontier MoE pretrain failure sat unread in this research fleet's own paid-subscriber archive for an entire prior research pass, misfiled as "paywalled" when the full post-mortem was always inside the free-preview boundary. → [33-llama4-behemoth-failure-postmortem](../notes/33-llama4-behemoth-failure-postmortem.md)
28. NVIDIA's own frontier-model technical reports disclose zero MFU, step time, or training-cluster GPU count for their main pretraining runs — a striking gap given how much operational detail (RL infrastructure failure rates down to the specific NCCL bug) they disclose elsewhere. → [34-nemotron-3-architecture-primary-source](../notes/34-nemotron-3-architecture-primary-source.md)
29. The standard `all_reduce_bench.py` busbw formula systematically undercounts bandwidth on SHARP/NVLS-active fabrics — anyone running this widely-recommended script on a fresh GB200 NVL72 cluster gets a pessimistic number relative to what's actually being achieved. → [40-stas-network-comms-debug](../notes/40-stas-network-comms-debug.md)
30. H100's headline 989 TFLOPS bf16 spec appears to assume a 1830MHz clock, not the commonly-quoted 1980MHz boost clock — reverse-derived from first principles, unexplained anywhere in NVIDIA's own materials. → [41-stas-performance-parallelism-dtype](../notes/41-stas-performance-parallelism-dtype.md)
31. The `ml-engineering` book is itself agent-maintained — a maintainer-only runbook explicitly written "for a code agent" governs its own consistency checks, unit conventions, and cross-book content-diffing. → [42-stas-storage-fault-orchestration](../notes/42-stas-storage-fault-orchestration.md)
32. BLOOM-176B ran its entire 176-billion-parameter training run with `CUDA_LAUNCH_BLOCKING=1` — fully synchronous CUDA — and measured zero throughput regression, a direct counterexample to the reflexive assumption that synchronous CUDA always costs performance. → [43-art-of-debugging](../notes/43-art-of-debugging.md)
33. Character.AI trained on a cluster running at one-quarter the network bandwidth of state-of-the-art systems and built a custom 6-bit gradient compressor (Squinch) specifically to make FSDP viable — the inverse of almost every other lab's bandwidth-adequate, reliability-constrained story. → [44-open-lab-war-stories](../notes/44-open-lab-war-stories.md)
34. Meta's production RoCE fabric has already reached 129,000 GPUs, per its own @Scale 2025 disclosure — larger than the 24,576-GPU figure fixed in most readers' mental model from the well-cited SIGCOMM'24 paper, achieved without a from-scratch transport redesign. → [45-conference-archive-mining](../notes/45-conference-archive-mining.md)
35. ByteDance's COMET has been running in the company's own ten-thousand-GPU production clusters since early 2025, saving "millions of GPU-hours" — and has no Blackwell backend as of its last public commit, a real gap between production deployment and hardware currency. → [46-moe-systems-papers](../notes/46-moe-systems-papers.md)
36. A classifier-cascade labeling pass for a 100T-token corpus costs only 0.2–0.5% of the training FLOPs it feeds, by three independently converging estimation methods — "can we afford to label everything" should be permanently retired as a planning question. → [47-data-processing-infra](../notes/47-data-processing-infra.md)

## What We Could Not Verify

The honesty this book owes its reader runs in both directions: every load-bearing claim is tagged with its evidence tier, and every genuine gap the research fleet hit is worth stating as plainly as the findings themselves.

**The current frontier is genuinely dark.** No source anywhere in this corpus — not Epoch AI, not the insider-journalism layer, not any lab's own disclosure — offers a FLOP estimate, GPU count, or duration for Gemini 3 Pro, any Claude Opus 4.x model, Claude Sonnet 4/4.5, or GPT-5.1 and its successors. Several of SemiAnalysis's most consequential pieces for this book's purposes — the Colossus 2 gigawatt-datacenter analysis, the fuller Meta Superintelligence progress update — remain title-confirmed but paywall-blocked in this pass.

**DeepSeek-V4's actual training footprint is undisclosed.** No source confirms V4's training hardware (H800, growing Ascend capacity, or some mix), exact parallelism degrees, cluster size, or GPU-hour cost — a sharper silence than V3's unusually transparent report, and one the corpus can only infer around from the absence of Ascend-specific kernel code in DeepSeek's public repositories.

**Whether MoE genuinely wants a different pretraining data mixture than a compute-matched dense model remains an open, and consequential, question.** No DeepSeek, Kimi, Qwen, Ling, or Hunyuan technical report claims a MoE-specific mixture, and no controlled ablation isolating this variable was found anywhere in the corpus — this may reflect genuine mixture-invariance, or it may simply reflect nobody having published the experiment.

**No joint EP+TP+PP+FSDP optimal-split closed form exists for GB200's hierarchical topology** in either the JAX Scaling Book or the HuggingFace Ultra-Scale Playbook — both handle individual parallelism axes well, but neither derives a true joint optimum across all four simultaneously for a two-tier NVLink-domain/scale-out-fabric hierarchy. This is a genuine open engineering problem, not a reading gap, and it recurs as an explicitly-flagged hole across three independent notes in this corpus.

**Achieved GEMM efficiency at real MoE training shapes, on GB200, in MXFP8 or NVFP4, at scale, does not appear anywhere in the public record this research could reach.** Every measured-efficiency number found is either an isolated microbenchmark tile shape or an H800/Hopper-generation figure — a load-bearing gap for anyone sizing a 10T/200B-active precision strategy.

**Whether SHARP in-network reduction is actually load-bearing in any confirmed 100K+-GPU frontier pretraining run, or is capability marketing that isn't exercised in practice at that scale, is unconfirmed.** NCCL 2.27's SHARP support is real and shipped; a documented NVSwitch-ALU silent-corruption incident implies someone is exercising in-network ALUs enough to have found the failure mode — but attribution and scale remain unconfirmed.

**No public source gives a validated hot-spare ratio for a 100K-GPU-class mega-cluster.** The only concrete number found — a 2% vendor-floor recommendation for NVL72 — is not derived against any specific mean-time-between-interruption curve at that scale.

**DatologyAI's headline multipliers (7.7x–43x "compute-equivalent") remain entirely vendor-self-reported**, with no independent replication located anywhere in this research, across a paid-vendor incentive structure this book flags every time the numbers recur.

**The training-logbook genre has essentially vanished from frontier commercial labs.** No lab training at genuine 2025–2026 frontier scale publishes anything resembling OPT's or BLOOM's day-by-day diary — DeepSeek's entire public disclosure on this axis is a one-sentence "zero loss spikes" claim. A reader wanting this book's Chapter 20 texture for a 10T-scale run has, as of this writing, no frontier-scale precedent to read; only extrapolation from smaller, still-instructive, non-frontier runs.

**No aggregated, fleet-wide stack-dump tooling for hang forensics exists past roughly 40 nodes anywhere in the public record**, and SLURM's FairShare scheduling algorithm has no drop-in Kubernetes-native equivalent — both confirmed gaps, not search failures, and both belong on the build list this book's Chapter 22 costs out.

## Implications for the 10T/200B Run

Read as a single object, this atlas argues for a specific discipline rather than a specific number: weight every claim in this book — and every claim a vendor, a competitor's paper, or a conference talk hands you during the actual run — by the evidentiary route it traveled, not by how confidently it's stated. The corpus's own worst failure modes recur in a pattern worth internalizing before the run starts. A number gets copy-pasted from a vendor spec sheet without checking whether it's dense or 2:4-sparse, and it silently halves or doubles every downstream roofline calculation built on it — this happened inside this book's own research fleet, caught only by a dedicated reconciliation pass (Note 30) built specifically because the error had already propagated into two other notes. A single unsourced blog post fans out into apparent multi-outlet consensus within months (Chapter 1's Gemini-3-on-TPU case is the cleanest example, but it is not the only one in this corpus). A vendor's own most candid admissions — AWS on Trainium's kernel-language gap, Meta on running without DCQCN, NVIDIA's silence on its own frontier-model MFU — surface only in primary documents, never in secondary summaries of them. And the field's best-evidenced numbers are reliably its most stale ones: DeepSeek's technical reports are the gold standard for verifiability in this entire corpus precisely because DeepSeek is not the lab currently defining the capability frontier.

For a team actually building the 10T/200B run, the operational lesson is to build the equivalent of this atlas as a living document from day one, not a retrospective. Tag every internal number by its evidentiary route the way this book tags its external ones — a vendor benchmark, a paper's stated methodology, an internal ablation, a rumor from a competitor's ex-employee — and revisit the tags as new evidence arrives, exactly as Note 30 forced a correction two other notes had already made. Budget explicitly for the gaps this atlas surfaces rather than assuming they'll resolve themselves before they matter: the joint EP+TP+PP+FSDP optimum, the achieved-MXFP8-at-MoE-shape number, the FairShare-equivalent for a Kubernetes fleet, and the fleet-scale hang-forensics tooling are all genuine, checkable absences in the public record as of this writing, and a program at this scale will need to derive, benchmark, or build each one internally rather than finding it published. The frontier lets very little slip, and what it does let slip has to be triangulated, dated, and dated again before it holds enough weight to bet a nine-figure training run on.
