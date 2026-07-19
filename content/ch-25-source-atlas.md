# Chapter 25 — The Source Atlas: Annotated Bibliography and Surprise Index

Every load-bearing claim in Chapters 1–24 traces back to a primary artifact — a paper, a repository, a technical report, a podcast transcript, a blog post written by a researcher who happened to say more than their employer's PR team would have liked. This closing chapter is that trail made explicit: an annotated bibliography organized by the topics the chapters cover, a ranked index of the most surprising individual findings, and an honest accounting of what could not be pinned down. Treat it as the footnotes, collected in one place and made navigable, rather than as new argument.

## Reading This Atlas

A ★ marks the 29 items most likely to be new to a reader who has already worked through the DeepSeek V3/V4 reports, the MAI-1 paper, and the standard scaling-law canon — items buried a citation-hop deep, published in the last few months before the July 13, 2026 evidence snapshot, or sitting in a genre (a personal wiki, a maintainer's scratch notes, an unpublished thesis) that doesn't show up in a normal literature search. The Surprise Index that follows the bibliography is deliberately terse — one sentence per finding — because the chapters already argue each point at length; the chapter pointer is the invitation to go read the argument in full.

## Annotated Bibliography

### Epistemics and the Closed Labs (Ch1–2)

- Gangidi et al., ["RDMA over Ethernet for Distributed AI Training at Meta Scale"](https://rmiao.github.io/assets/pdf/sigcomm24-final246.pdf), SIGCOMM 2024. The canonical Meta RoCE paper; superseded in scale (not in mechanism) by the @Scale 2025 recap below.
- ★ Meta Engineering, "@Scale: Networking 2025" recap, [engineering.fb.com](https://engineering.fb.com), Sept 2025. Discloses the RoCE fabric's growth to 129,000 GPUs — larger than the SIGCOMM'24 figure most readers have anchored on, and not yet reflected in most secondary coverage.
- OpenAI et al., ["Resilient AI Supercomputer Networking using MRC and SRv6"](https://arxiv.org/html/2605.04333v1), arXiv:2605.04333, May 2026. The actual MRC primary source — not an OCP Summit talk, a 50-author engineering-blog-grade disclosure of OpenAI's production fabric.
- Epoch AI, [notable_ai_models.csv](https://epoch.ai/data/notable_ai_models.csv), pulled July 12, 2026. The field's most methodologically transparent compute estimator — and, as of this pull, blind to Gemini 3 Pro, Claude Opus 4.x, and GPT-5.1.
- Epoch AI, ["Final training runs account for a minority of R&D compute spending"](https://epoch.ai/gradient-updates/r-and-d-vs-training-compute), fetched July 13, 2026. The ~9.6%–22.6% final-run-share finding that reframes every "training cost" headline.
- Dwarkesh Podcast, ["Jeff Dean and Noam Shazeer"](https://www.dwarkesh.com/p/jeff-dean-and-noam-shazeer), Feb 12, 2025. Confirms, on the record, that synchronous multi-metro pretraining is already routine at Google, gated purely by step-time budget.
- SemiAnalysis, ["Meta Superintelligence – Leadership, Compute, Talent, and Data"](https://newsletter.semianalysis.com/p/meta-superintelligence-leadership-compute-talent-and-data), Dylan Patel et al., July 11, 2025 (paywalled). ★ Contains the Llama 4 Behemoth failure post-mortem — which, despite the paywall, sits inside the free-preview boundary.

### The Open Frontier: China's Labs and DeepSeek (Ch3–4)

- DeepSeek-AI, ["DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437), arXiv:2412.19437, Dec 2024. The reference stack everything here benchmarks against.
- DeepSeek-AI, ["Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures"](https://arxiv.org/abs/2505.09343), arXiv:2505.09343, May 2025. ★ A wish list to NVIDIA/AMD that several vendors substantially granted within a year — the single most-skipped DeepSeek paper by readers who stop at the V3 report.
- DeepSeek-AI, ["DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"](https://arxiv.org/abs/2606.19348), arXiv:2606.19348, June 2026. Hybrid CSA/HCA attention, Muon, mHC — none of what the pre-release rumor mill predicted.
- Moonshot AI, ["Kimi K2: Open Agentic Intelligence"](https://arxiv.org/abs/2507.20534), arXiv:2507.20534, July 2025. The most infrastructurally transparent Chinese-lab report available.
- Moonshot AI, [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine), GitHub. ★ Broadcasts a full 1T-parameter weight update across thousands of GPUs in roughly 20 seconds — a concrete number for an RL weight-sync problem most labs mention only qualitatively.
- Ant Group/Ling Team, ["Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation"](https://arxiv.org/abs/2510.22115), arXiv:2510.22115, Oct 2025, and ["Towards Greater Leverage"](https://arxiv.org/abs/2507.17702), arXiv:2507.17702, July 2025. ★ The richest infra disclosure of any Chinese lab, including a full MFU waterfall (16.9%→32.8%) with each technique isolated.
- MiniMax, ["Why did M2 end up as a full attention model?"](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model), Nov 2025. A rare, quantified, named postmortem on abandoning an architectural bet (linear attention) after shipping it.
- ERNIE Team, Baidu, ["ERNIE 4.5 Technical Report"](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf), June 29, 2025. The one frontier Chinese lab not building on the Megatron-LM lineage; 47% MFU on 2,016 H800s via PaddlePaddle.
- LMSYS, ["Squeezing 1TB Model Rollout into a Single H200"](https://lmsys.org/blog/2026-01-26-int4-qat), Jan 2026. Independent reproduction of Kimi K2 Thinking's INT4 QAT within roughly two months of disclosure.
- Moonshot AI, [Kimi K2.5 model card](https://huggingface.co/moonshotai/Kimi-K2.5) and [K2.5 report](https://arxiv.org/abs/2602.02276), Jan/Feb 2026. The largest disclosed instance of retrofitting native multimodality onto a text flagship — a full ~15T-token continual-pretraining pass atop K2-Base, roughly doubling its cumulative budget; Ch11's staging anchor at 1T scale.
- Yang Zhilin, "How We Scaled Kimi K2.5," GTC 2026 main-stage keynote, March 2026, [via biggo's coverage](https://finance.biggo.com/news/WZHQBJ0ByH9TLH69H_BN) (secondhand — no official transcript). The three-pillars framing — Adam→MuonClip, softmax attention→3:1 KDA hybrid, residuals→Attention Residuals, each open-sourced — the clearest statement of Moonshot's open-components-as-differentiation posture, and the roadmap K3 then shipped.
- MiniMax, ["MiniMax Sparse Attention"](https://arxiv.org/abs/2606.13392), arXiv:2606.13392, June 2026. The M3 re-entry that completes the field's only closed-loop attention trajectory (linear → full → block-sparse softmax) — from the same lab that published the only quantified retreat.
- VentureBeat/MarkTechPost, [Kimi K3 launch coverage](https://www.marktechpost.com/2026/07/16/moonshot-ai-releases-kimi-k3-a-2-8-trillion-parameter-open-moe-model-with-kimi-delta-attention-and-1m-context/), July 16, 2026. The 2.8T/16-of-896 release, three days past the July 13 snapshot — every claim launch-coverage grade pending the technical report; Ch3's dated addendum owns the analysis.

### MoE Architecture, Scaling Laws, and Optimizer Theory (Ch5–9)

- Elango et al., ["LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"](https://arxiv.org/abs/2601.18089), arXiv:2601.18089, Jan 2026. ★ A full roofline derivation with named GB200 constants, sitting one citation-hop from the Megatron-Core MoE report and read by almost nobody.
- NVIDIA, ["Nemotron 3 Nano/Super/Ultra"](https://arxiv.org/abs/2512.20848) technical reports, Dec 2025–June 2026. ★ LatentMoE in actual production, plus two undiagnosed training-divergence incidents disclosed with MaxVio/residual-norm precursor signals — a genuinely actionable monitoring recipe.
- Abnar et al. (Apple), ["Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models"](https://arxiv.org/abs/2501.12370), arXiv:2501.12370, 2025. Optimal sparsity trending toward 1, no observed reversal even at the lowest tested activation ratio.
- StepFun, ["Predictable Scale Part I: Step Law"](https://arxiv.org/abs/2503.04715), arXiv:2503.04715, and ["Part II: Farseer"](https://arxiv.org/abs/2506.10972), arXiv:2506.10972. The largest public hyperparameter-scaling sweep to date, at roughly a third of DeepSeek-V3's entire pretraining GPU-hour budget.
- Su Jianlin, [kexue.fm](https://kexue.fm), ongoing. ★ Moonshot's de facto public research notebook — QK-Clip's MaxLogit mechanism, a reverse-engineering of DeepSeek-V4's hash-routing table, months ahead of any English-language coverage.
- Bernstein, ["Modular Manifolds"](https://thinkingmachines.ai/blog/modular-manifolds/), Thinking Machines Connectionism, Sept 26, 2025. ★ The direct theoretical continuation of Muon, constraining weight matrices to the Stiefel manifold.
- Newhouse, ["Duality, Weight Decay, and Metrized Deep Learning"](https://www.lakernewhouse.com/thesis.pdf), MIT MEng thesis, May 2025. ★ Unpublished-outside-MIT; quantifies Muon's Newton-Schulz "noise floor" and how vacuous formal Lipschitz guarantees become at competitive model quality.
- "Fantastic Pretraining Optimizers and Where to Find Them," [arXiv:2509.02046](https://arxiv.org/abs/2509.02046), Sept 2025, and its "Hyperball" rebuttal, [arXiv:2606.16899](https://arxiv.org/abs/2606.16899), June 2026. An adversarial re-benchmark of nearly every optimizer-speedup claim in circulation, followed by a dated counter-rebuttal.

### Precision and Numerics (Ch10)

- NVIDIA, ["Pretraining Large Language Models with NVFP4"](https://arxiv.org/html/2509.25149v2), arXiv:2509.25149, Sept 2025. The reference 2-level-scaling, Random-Hadamard NVFP4 recipe, validated to 12B/10T tokens.
- Kimi K2 team, community discussion, [huggingface.co/moonshotai/Kimi-K2-Instruct/discussions/30](https://huggingface.co/moonshotai/Kimi-K2-Instruct/discussions/30). Explicit rejection of FP8 pretraining compute for a 1T-parameter MoE after a stability study — the sharpest counter-example to "FP8 pretraining is solved."
- "Four Over Six," [arXiv:2512.02010](https://arxiv.org/html/2512.02010v3), Dec 2025. Dual-scale NVFP4 fix closing 22% of the loss gap for under 15% overhead, validated on Nemotron-3-Nano.
- ★ mufeezamjad, ["NVFP4 Grouped GEMM"](https://mufeezamjad.com/blog/nvfp4-group-gemm), 2026 (a personal engineering blog). A 10x gap between naive and hand-optimized NVFP4 grouped GEMM on B200, with the exact architectural reason (TMEM capacity forcing 1 CTA/SM) spelled out — the sharpest public treatment of the "small-M problem" anywhere.

### Multimodal-Native Pretraining (Ch11)

- Baidu, ["ERNIE 5.0 Technical Report"](https://arxiv.org/abs/2602.04705), arXiv:2602.04705, Feb 2026. Explicitly reverses ERNIE 4.5's own modality-isolated routing within eight months, naming the prior model as the design it's abandoning.
- Zhou et al. (Meta FAIR), ["MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts"](https://arxiv.org/abs/2407.21770), arXiv:2407.21770, July 2024. The actual origin of modality-isolated MoE, predating ERNIE's more commonly credited version.
- DeepSeek-AI, ["DeepSeek-OCR: Contexts Optical Compression"](https://arxiv.org/abs/2510.18234), arXiv:2510.18234, Oct 2025. Frames vision tokens as a general context-compression primitive, not an OCR trick — directly relevant to long-context token-budget planning.

### The Data Supply Chain and Curation (Ch12–13)

- Essential AI, ["Essential-Web v1.0"](https://arxiv.org/abs/2506.14111), arXiv:2506.14111, June 2025. The reference curation architecture: 24T tokens, a 12-category taxonomy, an 0.5B distilled classifier.
- ★ Li et al. (Apple/Stanford/UW), ["Beyond a Single Extractor: Re-thinking HTML-to-Text Extraction for LLM Pretraining"](https://arxiv.org/abs/2602.19548), arXiv:2602.19548, Feb 2026. Only 39% of pages survive post-curation filtering across more than one extractor — extractors are complementary, not redundant, and running the union yields a 50–70% token-yield increase.
- ★ Ma et al. (Shanghai AI Lab), ["AICC: Parse HTML Finer, Make Models Better"](https://arxiv.org/abs/2511.16397), arXiv:2511.16397, Nov 2025. A 0.6B sequence-labeling extractor beating trafilatura by 18 ROUGE-F1 points.
- ★ GAIR-NLP, "Data Darwinism Part II: DataEvolve," [arXiv:2603.14420](https://arxiv.org/abs/2603.14420), March 2026. An LLM literally writing and grading its own data-cleaning prompts across 30 generations — the most literal instantiation of "data agents" published so far, with a public repo and dataset.
- ★ "Introspective X Training," [arXiv:2605.20285](https://arxiv.org/abs/2605.20285), May 2026. Metadata/feedback-conditioning experiments at 7.5–12B models over 18 trillion tokens seen — an order of magnitude past the technique's earlier published scale, and almost certainly unseen by anyone on a pre-2026 reading list.
- DatologyAI, ["BeyondWeb"](https://datologyai.com/blog/beyondweb), Aug 2025. Real numbers from a paid vendor with zero independent replication anywhere we could find — cite the direction, not the multiplier.

### Midtraining and Curriculum (Ch14)

- Allen Institute for AI, ["2 OLMo 2 Furious"](https://arxiv.org/abs/2501.00656), arXiv:2501.00656, and ["OLMo 3"](https://arxiv.org/abs/2512.13961), arXiv:2512.13961. The single richest primary sources on midtraining — full Dolmino-mix tables, souping ablations, the special-chat-token GSM8K landmine.
- Microsoft AI, ["MAI-Thinking-1: Building a Hill-Climbing Machine"](https://microsoft.ai), June 2026. Explicitly excludes AI-generated text from its 30T-token base corpus — a live, frontier-scale disagreement with the DatologyAI/Nemotron synthetic-data-scaling consensus.

### Dataloading and the Input Pipeline (Ch15)

- ★ VAST Data, ["Optimizing Checkpoint Bandwidth for LLM Training"](https://blog.vastdata.com/optimizing-checkpoint-bandwidth-for-llm-training), 2026 (a vendor blog). An 85,000-checkpoint, 40-production-run telemetry survey showing real global checkpoint bandwidth tops out at 50–200GB/s — a data point that should recalibrate any naive per-parameter checkpoint-storage sizing.
- Li et al., ["MegaScale-Data: Scaling Dataloader for Multisource Large Foundation Model Training"](https://arxiv.org/abs/2504.09844), EuroSys 2026. Documents a 3.2–6.9x per-microbatch FLOP imbalance from dataloader composition alone.

### Hardware Ground Truth: GB200 and the Alternative-Silicon Question (Ch16)

- NVIDIA "Blackwell" datasheet, ★ [document #3384703](https://www.openzeka.com/wp-content/uploads/2025/02/blackwell-datasheet.pdf), Dec 2024. The single authoritative side-by-side GB200 NVL72 vs. HGX B200 spec table (5,000 vs. 4,500 TFLOPS/GPU dense FP8; 186GB vs. 180GB HBM3e) that resolves the FLOPS conflation Ch16 documents. NVIDIA's own CDN copy 403'd on direct fetch, so the link points to the mirror actually retrieved and verified.
- ★ Glenn K. Lockwood, ["NVIDIA B200"](https://glennklockwood.com/garden/processors/b200). An independently-compiled, footnoted, frequently-updated technical wiki that cross-validates cleanly against the official datasheet and is not indexed by generic tech-news search.
- Jouppi et al. (Google), ["Google's Training Supercomputers from TPU v2 to Ironwood"](https://arxiv.org/abs/2606.15870), arXiv:2606.15870, June 2026. The single most quantified cross-generation hardware disclosure any lab has given the field, including the Ironwood Hardware Replay Unit's zero-overhead SDC detection.
- ★ Zyphra/AMD/IBM, ["Training Foundation Models on a Full-Stack AMD Platform"](https://arxiv.org/abs/2511.17127), arXiv:2511.17127. By far the richest 2025–2026 AMD hard-numbers source, unusually candid about the platform's own limitations.
- AWS Neuron docs, [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com) and [Trainium3](https://awsdocs-neuron.readthedocs-hosted.com) architecture pages, and AWS, ["Announcing Neuron Agentic Development"](https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-agentic-development), April 2026. AWS's own written admission that its kernel language is still too hard for most ML engineers, three SDK generations in.

### Kernels for Blackwell (Ch17)

- Dao et al., ["FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"](https://arxiv.org/abs/2603.05451), arXiv:2603.05451, March 2026. The primary FA4 source; verified by direct code inspection (Ch17) to have a forward-only FP8 path as of July 2026.
- DeepSeek-AI, [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [DeepEP](https://github.com/deepseek-ai/DeepEP), and [EPLB](https://github.com/deepseek-ai/EPLB), GitHub. The MoE-kernel reference implementations most Blackwell-generation frameworks build against.

### Parallelism Math and Frameworks (Ch18–19)

- Tazi, Mom, Zhao et al. (HuggingFace), the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), 2025. Over 4,000 scaling experiments on up to 512 GPUs; carries the DeepSeek routing-locality constraint the Scaling Book's own EP section omits.
- ★ Arcee AI, ["Trinity Large Technical Report"](https://arxiv.org/abs/2602.17004), arXiv:2602.17004, Feb 2026. An unusually candid production post-mortem — a forced MXFP8→BF16 fallback and six concurrent stability interventions on a real ~400B-total MoE, the single best "is this actually production grade" data point for open frameworks.
- NVIDIA, ["Scalable Training of Mixture-of-Experts Models with Megatron Core"](https://arxiv.org/abs/2603.07685), arXiv:2603.07685, March 2026. The canonical Megatron-Core MoE reference, superseding scattered blog posts.

### Networking at 100k (Ch20)

- Si et al. (Meta), ["Collective Communication for 100k+ GPUs" (NCCLX)](https://arxiv.org/abs/2510.20171), arXiv:2510.20171, Oct 2025. The richest source on production 100K+-GPU collectives engineering anywhere public.
- ★ arXiv, [2605.04333](https://arxiv.org/html/2605.04333v1) — see the epistemics section above; also load-bearing here as the primary source for OpenAI's actual scale-out design.
- Qian et al. (Alibaba Cloud), ["Alibaba HPN"](https://ennanzhai.github.io/pub/sigcomm24-hpn.pdf), SIGCOMM 2024. Quantifies multi-chip chassis switches failing 3.77x more often than single-ASIC designs — the concrete reason the industry converged on single-ASIC fabrics.

### Fault Tolerance and Checkpointing (Ch21)

- ★ "Training LLMs with Fault Tolerant HSDP on 100,000 GPUs," [arXiv:2602.00277](https://arxiv.org/abs/2602.00277), Feb 2026. The single best-matched primary source for 100K-GPU reliability, giving exact failure rates and recovery-stall numbers; not yet widely cited elsewhere as of mid-2026.
- Meta, ["The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783), arXiv:2407.21783, July 2024. The 419-unexpected-interruption/54-day ground truth every later fault-tolerance paper benchmarks against.
- ★ Hu et al. (Shanghai AI Laboratory), ["Characterization of Large Language Model Development in the Datacenter"](https://arxiv.org/abs/2403.07648), NSDI 2024 (the Acme trace). Median job duration in a real frontier-adjacent datacenter is two minutes, not hours — 92.9% of jobs are tiny eval jobs, a scheduler-design fact easy to lose sight of while planning for the pretraining tail.
- ★ Alibaba, ["Boosting Large-Scale Parallel Training Efficiency with C4: A Communication-Driven Approach"](https://arxiv.org/abs/2406.04594), arXiv:2406.04594, Jun 2024 (venue/author list not independently confirmed). A 30x reduction in error-induced downtime (31.19%→1.16%) came almost entirely from cutting detection latency, not from any change in underlying hardware failure rate.
- ★ "Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement," [arXiv:2407.04656](https://arxiv.org/abs/2407.04656). 20–40-second failure-to-resume by exploiting MoE's structurally redundant, independently-placeable experts — reliability engineering as an argument for MoE distinct from its usual compute-efficiency framing.

### Operations, War Stories, and Historical Postmortems (Ch22)

- Stas Bekman, ★ [ml-engineering](https://github.com/stas00/ml-engineering) and ★ [the-art-of-debugging](https://github.com/stas00/the-art-of-debugging), GitHub, both repos actively maintained through July 2026. The single densest practitioner-authored primary source in this entire bibliography — a MAMF achievable-efficiency table, a fully-worked GB200 NVLink-C2C duplex measurement at 38% of spec, and (per its own maintainer runbook) evidence the book is itself agent-maintained.
- Meta AI, ["OPT-175B Logbook"](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles), 2021–2022, and BigScience, ["BLOOM chronicles"](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md), 2022. ★ The two canonical day-by-day training logbooks the entire genre is measured against — a genre that has since essentially vanished from frontier commercial labs (Ch22).
- Imbue, ["70B infrastructure"](https://imbue.com/blog/70b-infrastructure), June 2024. The most complete public "how do you actually bring up a 500-node IB fabric" gauntlet available, with exact MFU signatures per failure class.
- Character.AI, ["Squinch"](https://blog.character.ai/squinch), 2024. A rare admission of bandwidth-constrained (not reliability-constrained) training, and a custom 6-bit gradient compressor built specifically to make FSDP viable under that constraint.

### Benchmarks and the Money Math (Ch23)

- NVIDIA, [Megatron-Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html), 26.06 container. The primary source for the dense-vs-MoE MFU gap table — 32–39% dense vs. 9–18% MoE in its earlier snapshot, same framework, same hardware, three GPU generations.
- MLCommons, [MLPerf Training v6.0 results](https://mlcommons.org/2026/06/mlperf-training-v6-0-results), June 16, 2026. The first standardized, third-party-audited large-MoE training benchmark — and the round where NVIDIA, in its own words, was "the only platform to submit on every test."

## The Surprise Index

The 36 findings most likely to be new to a reader who has already internalized DeepSeek V3/V4 and the MAI-1 report, ranked roughly by how far they sit from the standard reading list rather than by topic. Each is one sentence; where a chapter argues the point in full, the pointer follows.

1. NVIDIA's own GB200 NVL72 product page reports 2:4 structured-sparse FLOPS without labeling them as such — the scaling book that calls out this exact trick elsewhere doesn't flag it for Blackwell. → Ch16
2. ECHO (Elastic Cloning for Hot Experts) dynamically clones overloaded experts to spare capacity slots specifically to make CUDA-Graph static buffers tractable for dropless MoE — merged into Megatron-Core in February 2026 and barely discussed outside its source paper. → Ch19
3. TorchTitan ships a dedicated `HybridEPTokenDispatcher` built for "GB200/NVLink72" in core, not an experiments folder. → Ch19
4. MaxText's own GPU support is frozen at a single `"a3"` (H100) entry as of June 2026 — zero A4X/GB200/B200 config entries, despite Google Cloud selling A4X since 2025. → Ch19
5. DeepSeek's V3 paper states its own comm budget could support 13 activated experts instead of the 8 it actually uses — the routing topology, not the token compute, is the binding constraint, and the lab deliberately leaves slack on the table. → Ch4
6. Ling-1T's MFU optimization waterfall (16.9%→32.8%) isolates each technique's individual contribution — nearly unique in a field where most labs report only a single final number, if any. → Ch3
7. DeepSeek-V4 shipped with none of what the pre-release rumor mill predicted (no "Engram," no "MODEL1") — the real innovations are a CSA/HCA hybrid-compression attention scheme and Sinkhorn-projected hyper-connections. → Ch4
8. StepFun's Step Law hyperparameter-scaling sweep cost roughly a third of DeepSeek-V3's entire pretraining GPU-hour budget just to fit the laws — "fitting the ladder" is no longer meaningfully cheaper than training a serious mid-scale model. → Ch8
9. NVIDIA's NeMo Curator dedupes 1.96 trillion tokens in 30 minutes on 32 H100s — deduplication of an entire multi-trillion-token corpus is a same-afternoon operation, not a pipeline bottleneck. → Ch13
10. Common Crawl's own bot is now blocked almost as often as OpenAI's dedicated GPTBot (5.08% vs. 5.52% of DISALLOW rules) — "just use CC" no longer sidesteps the site-blocking fight it's supposed to avoid. → Ch12
11. Best-Fit Packing's padding overhead measures 0.0006%–0.003% — functionally zero, closing a "live tradeoff" most secondary literature still treats as open. → Ch15
12. FlashAttention-4's FP8 path is forward-only as of July 2026, verified directly in source (`NotImplementedError`) rather than inferred from documentation — anyone planning FP8 attention training on Blackwell needs a fallback. → Ch17
13. Kimi K2's team explicitly rejected FP8 pretraining compute for a 1T-parameter MoE after a preliminary stability study — a named, citable counter-example to the "FP8 pretraining is fully solved" narrative most 2025–2026 commentary carries. → Ch10
14. Meta runs its entire 24K+-GPU RoCE backend with DCQCN turned off, relying purely on PFC plus receiver-driven admission control in the collective library itself — directly contradicting the "DCQCN is mandatory" assumption in most vendor documentation. → Ch20
15. NVLink Paladin HD connectors are rated for roughly 200 mating cycles — a hard mechanical wear-out cap on how many times a GB200 NVL72 backplane can be serviced over a multi-year deployment. → Ch16
16. The widely-circulated "58% MFU for DeepSeek-V3" claim does not survive reconstruction from the paper's own stated GPU-hours and token count. → Ch23
17. Google's April 2026 Virgo Network/TPU 8t-8i disclosure claims JAX+Pathways can orchestrate more than one million TPU chips as a single logical training cluster — a step-change in stated ambition three months old as of the snapshot. → Ch2
18. DeepSeek switched from AdamW (V3) to Muon (V4) between December 2024 and June 2026 — the strongest revealed-preference signal in the entire optimizer literature, from the lab with the best public track record for MoE stability. → Ch9
19. Su Jianlin's blog reverse-engineers DeepSeek-V4's tid2eid hash-routing construction algorithm from a leaked checkpoint artifact — the earliest accessible technical description of that specific V4 design choice in any language. → Ch4
20. Microsoft's MAI-Thinking-1 (June 2026) actively filters AI-generated text out of its 30T-token base pretraining corpus at genuinely frontier scale (8,000 GB200s) — a live, unresolved strategic disagreement with the DatologyAI/Nemotron synthetic-data-scaling consensus. → Ch13
21. A SemiAnalysis contractor spent roughly $10,000 in agentic (Claude/Codex) compute fuzzing NVIDIA's closed-source `ptxas` compiler and found 40 distinct miscompiling programs in three days — one agent-discovered bug silently converts an atomic store into two non-atomic stores.
22. AWS states outright that NVIDIA's own GB200/InfiniBand backplane reliability saga — which took six to seven months and a firmware fix to resolve — directly motivated Trainium3's redundant-lane, hot-swap-without-drain design. → Ch16
23. Stas Bekman's own `nvbandwidth` measurement on a DGX Station puts real-world NVLink-C2C duplex efficiency at 38% of the marketed 900GB/s spec, bottlenecked by Grace's LPDDR5X bandwidth — not published anywhere by NVIDIA.
24. The JAX Scaling Book contradicts itself within the same document — its main FLOPS table and its own Quiz Question 6 cite two different Blackwell SKUs under one undifferentiated "B200" label, and the resulting number gets copy-pasted, uncorrected, into downstream derivations. → Ch16
25. ERNIE 5.0 explicitly reverses ERNIE 4.5's own modality-isolated MoE routing within eight months, naming its own prior model as the design it is moving away from. → Ch11
26. Baidu's router-orthogonalization loss bypasses the ordinary loss function entirely, patching directly into the Adam update instead — "analogous to AdamW," in the paper's own words — so a naive reimplementation that just adds the term to the loss would get it measurably wrong. → Ch11
27. The single richest technical account of a frontier MoE pretrain failure — Llama 4 Behemoth's — sits inside the free-preview boundary of a SemiAnalysis piece most readers assume is fully paywalled. → Ch2
28. NVIDIA's own frontier-model technical reports disclose zero MFU, step time, or training-cluster GPU count for their main pretraining runs — a striking gap given how much operational detail (RL infrastructure failure rates down to the specific NCCL bug) they disclose elsewhere.
29. The standard `all_reduce_bench.py` busbw formula systematically undercounts bandwidth on SHARP/NVLS-active fabrics — anyone running this widely-recommended script on a fresh GB200 NVL72 cluster gets a pessimistic number relative to what's actually being achieved. → Ch20
30. H100's headline 989 TFLOPS bf16 spec appears to assume a 1830MHz clock, not the commonly-quoted 1980MHz boost clock — reverse-derived from first principles, unexplained anywhere in NVIDIA's own materials.
31. The `ml-engineering` book is itself agent-maintained — a maintainer-only runbook explicitly written "for a code agent" governs its own consistency checks, unit conventions, and cross-book content-diffing.
32. BLOOM-176B ran its entire 176-billion-parameter training run with `CUDA_LAUNCH_BLOCKING=1` — fully synchronous CUDA — and measured zero throughput regression, a direct counterexample to the reflexive assumption that synchronous CUDA always costs performance. → Ch22
33. Character.AI trained on a cluster running at one-quarter the network bandwidth of state-of-the-art systems and built a custom 6-bit gradient compressor (Squinch) specifically to make FSDP viable — the inverse of almost every other lab's bandwidth-adequate, reliability-constrained story. → Ch22
34. Meta's production RoCE fabric has already reached 129,000 GPUs, per its own @Scale 2025 disclosure — larger than the 24,576-GPU figure fixed in most readers' mental model from the well-cited SIGCOMM'24 paper, achieved without a from-scratch transport redesign. → Ch20
35. ByteDance's COMET has been running in the company's own ten-thousand-GPU production clusters since early 2025, saving "millions of GPU-hours" — and has no Blackwell backend as of its last public commit, a real gap between production deployment and hardware currency. → Ch19
36. A classifier-cascade labeling pass for a 100T-token corpus costs only 0.2–0.5% of the training FLOPs it feeds, by three independently converging estimation methods — "can we afford to label everything" should be permanently retired as a planning question. → Ch13

## What We Could Not Verify

Honesty runs in both directions: the chapters say how hard each claim can be leaned on, and the genuine gaps deserve stating as plainly as the findings themselves.

**The current frontier is genuinely dark.** No public source — not Epoch AI, not the insider-journalism layer, not any lab's own disclosure — offers a FLOP estimate, GPU count, or duration for Gemini 3 Pro, any Claude Opus 4.x model, Claude Sonnet 4/4.5, or GPT-5.1 and its successors. Several of SemiAnalysis's most consequential pieces — the Colossus 2 gigawatt-datacenter analysis, the fuller Meta Superintelligence progress update — remain title-confirmed but unread behind the paywall.

**Kimi K3's launch-week claims are entirely secondary-sourced as of mid-July 2026.** The July 16, 2026 release post-dates the July 13 snapshot by three days; its active-parameter count, long-context evaluations, and the entire named-but-undocumented training stack (Per-Head Muon, SiTU, Quantile Balancing in production) await the technical report — the single most important pending document for the architecture chapters (Ch3's addendum tracks it).

**DeepSeek-V4's actual training footprint is undisclosed.** No source confirms V4's training hardware (H800, growing Ascend capacity, or some mix), exact parallelism degrees, cluster size, or GPU-hour cost — a sharper silence than V3's unusually transparent report, and one that can only be inferred around from the absence of Ascend-specific kernel code in DeepSeek's public repositories.

**Whether MoE genuinely wants a different pretraining data mixture than a compute-matched dense model remains an open, and consequential, question.** No DeepSeek, Kimi, Qwen, Ling, or Hunyuan technical report claims a MoE-specific mixture, and no controlled ablation isolating this variable has been published anywhere — this may reflect genuine mixture-invariance, or it may simply reflect nobody having published the experiment.

**No joint EP+TP+PP+FSDP optimal-split closed form exists for GB200's hierarchical topology** in either the JAX Scaling Book or the HuggingFace Ultra-Scale Playbook — both handle individual parallelism axes well, but neither derives a true joint optimum across all four simultaneously for a two-tier NVLink-domain/scale-out-fabric hierarchy, and no third source fills the gap. This is a genuine open engineering problem, not a reading gap.

**Achieved GEMM efficiency at real MoE training shapes, on GB200, in MXFP8 or NVFP4, at scale, does not appear anywhere in the public record we could reach.** Every measured-efficiency number found is either an isolated microbenchmark tile shape or an H800/Hopper-generation figure — a load-bearing gap for anyone sizing a 10T/200B-active precision strategy.

**Whether SHARP in-network reduction is actually load-bearing in any confirmed 100K+-GPU frontier pretraining run, or is capability marketing that isn't exercised in practice at that scale, is unconfirmed.** NCCL 2.27's SHARP support is real and shipped; a documented NVSwitch-ALU silent-corruption incident implies someone is exercising in-network ALUs enough to have found the failure mode — but attribution and scale remain unconfirmed.

**No public source gives a validated hot-spare ratio for a 100K-GPU-class mega-cluster.** The only concrete number found — a 2% vendor-floor recommendation for NVL72 — is not derived against any specific mean-time-between-interruption curve at that scale.

**DatologyAI's headline multipliers (7.7x–43x "compute-equivalent") remain entirely vendor-self-reported**, with no independent replication located anywhere — a paid-vendor incentive structure worth remembering every time the numbers recur.

**The training-logbook genre has essentially vanished from frontier commercial labs.** No lab training at genuine 2025–2026 frontier scale publishes anything resembling OPT's or BLOOM's day-by-day diary — DeepSeek's entire public disclosure on this axis is a one-sentence "zero loss spikes" claim. Anyone wanting Chapter 22's texture for a 10T-scale run has, as of mid-2026, no frontier-scale precedent to read; only extrapolation from smaller, still-instructive, non-frontier runs.

**No aggregated, fleet-wide stack-dump tooling for hang forensics exists past roughly 40 nodes anywhere in the public record**, and SLURM's FairShare scheduling algorithm has no drop-in Kubernetes-native equivalent — both confirmed gaps, not search failures, and both belong on Chapter 24's build list.

## Implications for the 10T/200B Run

Read as a single object, this atlas argues for a specific discipline rather than a specific number: weight every claim — from these chapters, from a vendor, from a competitor's paper, from a conference talk mid-run — by the evidentiary route it traveled, not by how confidently it's stated. The failure modes recur in a pattern worth internalizing before the run starts. A number gets copy-pasted from a vendor spec sheet without checking whether it's dense or 2:4-sparse, and it silently halves or doubles every downstream roofline calculation built on it — the JAX Scaling Book's B200 tables show even careful authors doing exactly this, and Ch16 exists because the error propagates. A single unsourced blog post fans out into apparent multi-outlet consensus within months (Ch1's Gemini-3-on-TPU case is the cleanest example, but not the only one). A vendor's own most candid admissions — AWS on Trainium's kernel-language gap, Meta on running without DCQCN, NVIDIA's silence on its own frontier-model MFU — surface only in primary documents, never in secondary summaries of them. And the field's best-evidenced numbers are reliably its most stale ones: DeepSeek's technical reports are the gold standard for verifiability precisely because DeepSeek is not the lab currently defining the capability frontier.

For a team actually building the 10T/200B run, the operational lesson is to keep an atlas like this as a living document from day one, not a retrospective. Tag every internal number by its evidentiary route — a vendor benchmark, a paper's stated methodology, an internal ablation, a rumor from a competitor's ex-employee — and revisit the tags as new evidence arrives; a periodic reconciliation pass that hunts for already-propagated errors pays for itself the first time it catches a wrong denominator, exactly as the GB200 SKU correction did here. Budget explicitly for the gaps this atlas surfaces rather than assuming they'll resolve themselves before they matter: the joint EP+TP+PP+FSDP optimum, the achieved-MXFP8-at-MoE-shape number, the FairShare-equivalent for a Kubernetes fleet, and the fleet-scale hang-forensics tooling are all genuine, checkable absences in the public record as of mid-July 2026, and a program at this scale will need to derive, benchmark, or build each one internally rather than finding it published. The frontier lets very little slip, and what it does let slip has to be triangulated, dated, and dated again before it holds enough weight to bet a nine-figure training run on.
