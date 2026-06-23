# The Frontier Pretraining Atlas

**A systems-and-data field manual for the 10-trillion-parameter MoE.** What it
actually takes to pretrain a 10T-total / 200B-active Mixture-of-Experts model on
100,000 GB200 GPUs — hardware, networking, parallelism, MoE systems, kernels,
numerics, optimizers, scaling laws, data, dataloading, fault tolerance, and
midtraining — grounded in primary sources and a live MaxText checkout.

### 📖 Read it

Open `index.html` locally (it needs no server), or grab the single-file
[`frontier-pretraining-atlas.html`](frontier-pretraining-atlas.html) to read
offline / send around — it inlines everything (CSS + JS) and makes **zero
external requests**, no fonts, no analytics, no math CDN.

## What's inside

20 chapters + 4 appendices, ~110,000 words, MoE-only (dense is not relitigated).
Every figure is live, theme-aware SVG — there are **no image assets** — with
scroll-reveal, self-drawing charts, interactive toggles, and a few bespoke
animations (MRC packet-spray, rail topology, MoE load-imbalance). Color encodes
meaning: every concept family (compute · comms · moe · mem · prec · data · optim
· attn) owns one color, used in both prose accents and figures.

**Part I — the machine:** the frontier landscape · the 100k-GPU machine · networking & collectives.
**Part II — mapping the model:** parallelism · MoE systems · framework shoot-out · the Scaling Book applied.
**Part III — the numerical core:** attention & kernels · low-precision · optimizers · stability.
**Part IV — data, scale & the long run:** scaling laws · data curation · dataloading · fault tolerance · midtraining.
**Part V — the frontier & the payoff:** the Chinese frontier · the theory corner · public throughput signals · the punch-list.
**Appendices:** the scaling cheat-sheet · the operational & economic reality · Stas Bekman's ML-Engineering distilled · a ~200-resource bibliography.

## Build

The site is plain static HTML + one CSS file + one JS file (`.nojekyll`, so
GitHub Pages serves it verbatim). Chapter bodies live in `parts/<id>.main.html`;
`tools/build.py` wraps them in the page template and produces the standalone.

```bash
pyenv activate sn120          # pandoc must be on PATH
python tools/build.py all     # wrap every page + rebuild the standalone
```

- `tools/build.py baseline` — re-derive clean HTML bodies from the source markdown via pandoc (safety net).
- `tools/CONTRACT.md` — the authoring contract: component vocabulary + the figure-spec schema.

## Credits & license

Design system adapted from the **MiniMax-M3 Inference Atlas**. Prose and figures:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code (`book.js`,
`book.css`, build script): MIT.

Built from primary sources — arXiv papers, official technical reports, framework
source code, the JAX Scaling Book, a SemiAnalysis archive, Stas Bekman's
ML-Engineering, and a live MaxText checkout (HEAD 2026-06-22) — with every
load-bearing number put through an adversarial verification pass. Where a number
is a marketing or sparse figure it is flagged; where the open literature does not
know, the book says so.
