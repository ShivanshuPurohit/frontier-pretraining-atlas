# Authoring contract — The Frontier Pretraining Atlas

You are converting ONE chapter of a book-length technical report into a richly
visual, web-ready chapter for a hand-built static site (lineage: the MiniMax-M3
Inference Atlas). You write **only the inner content of `<main>`** — no
`<!doctype>`, `<head>`, top `<nav>`, or `<script>` tags; the build wraps those.

The output is a single HTML fragment written to a file you are told. It begins
with the chapter header block and ends with the Sources section. Do **not**
include the prev/next footer nav (the build adds it).

## Absolute rules

1. **Preserve every claim and number from the source markdown.** You are
   *enriching and re-rendering*, never summarizing. The chapter should retain
   ≥95% of the source's substance. Re-flow prose into the components below, but
   do not drop paragraphs, caveats, or citations.
2. **Keep every inline citation link.** Source uses `[text](url)` → render as
   `<a href="url">text</a>`. The Sources list at the end stays.
3. **Color = meaning.** Every concept family has ONE color. Use the family
   classes consistently (see taxonomy). A networking term is always `comms`
   (red); an MoE term always `moe` (amber); etc.
4. **Real code, not pseudocode.** Add 2–4 genuine code blocks per chapter where
   they illuminate (JAX/Flax, PyTorch, Triton/Pallas, CUDA-ish, or YAML config).
   Prefer real API shapes (e.g. `jax.lax.ragged_dot`, `shard_map`,
   `optax.contrib.muon`, MaxText flag names). Annotate the important ones.
5. **3–7 figures per chapter**, chosen from the figure library below. Figures
   must be data-driven JSON specs — never hand-write `<svg>`. Numbers in figures
   must match the chapter's numbers.
6. **No math engine.** Math is Unicode (× · ≈ ≤ ≥ → √ ² ∝) and `<sub>`/`<sup>`.
   For display equations use the `.formula` block.
7. Output valid, well-formed HTML. Every `<figure>`'s `<script class="spec">`
   must contain valid JSON (no trailing commas, double quotes only).

## Family taxonomy (class suffix → concept → color)

| family    | concept                                              | color  |
|-----------|------------------------------------------------------|--------|
| `compute` | GEMM, tensor cores, FLOPs, raw compute, frameworks   | purple |
| `comms`   | networking, all-reduce, all-to-all, MRC, collectives | red    |
| `moe`     | experts, routing, MoE systems, capacity              | amber  |
| `mem`     | HBM, KV cache, checkpoints, storage, I/O, dataloading| green  |
| `prec`    | FP8/FP4/MXFP, numerics, quantization                 | teal   |
| `data`    | data curation, scaling laws, dataloading logic       | blue   |
| `optim`   | optimizer, muP, schedules, stability                 | magenta|
| `attn`    | attention, kernels (FA4, MLA, sparse)                | indigo |
| `norm`    | glue, overview, misc, neutral                        | slate  |

## Header block (start every chapter with this)

```html
<p class="eyebrow">Chapter NN · <SHORT KICKER></p>
<h1><CHAPTER TITLE></h1>
<p class="lede"><1–3 sentence hook that states the stakes for the 10T/200B/100k-GB200 run.></p>
<p class="byline"><span>~NN min read</span><span class="dot">N figures</span><span class="dot"><grounding, e.g. grounded in MaxText moe.py></span></p>
<hr />
```

## Section headings

```html
<h2 id="slug"><span class="sec-num">01</span>Section title</h2>
<h3 id="slug2">Sub-section</h3>
```
Number the `sec-num` 01, 02, … per chapter. Always give headings an `id`.

## Components (copy these patterns)

**Callouts** — kinds: `note` (blue, context), `warn` (red, danger/trap),
`key` (amber, the one thing to remember), `do` (green, actionable), `miss` (slate, caveat/gap):
```html
<div class="callout key"><div class="ct">The one number to remember</div>
  <p>…</p></div>
```

**Stat strip** — for headline numbers (add `data-fam` to color the value):
```html
<div class="stats">
  <div class="stat" data-fam="moe"><div class="v">256<span class="u">exp</span></div><div class="l">top-8 routing</div></div>
  <div class="stat" data-fam="comms"><div class="v">18<span class="u">×</span></div><div class="l">faster all-to-all on NVL72</div></div>
</div>
```

**Fact cards** (`data-fam` colors the left rule + mono line):
```html
<div class="tcards">
  <div class="tcard" data-fam="prec"><div class="tn">expert weights</div><div class="ts">FP4 (NVFP4)</div><div class="tm">2× compute on the dominant GEMM class.</div></div>
</div>
```

**Display equation** (Unicode):
```html
<div class="formula"><span class="lbl">comms</span>
  <span class="eq">AllToAll bytes/layer = 2 · tokens · top-k · d<sub>model</sub> · bytes</span>
  <span class="note">Doubled for dispatch + combine. The term that gates MoE MFU.</span></div>
```

**Inline accents**: `<span class="f-comms">all-to-all</span>` (colored bold word),
`<span class="k k-moe">ragged_dot</span>` (mono chip for an API/kernel/flag),
`<span class="conf conf-leaked">leaked</span>` / `conf-confirmed` / `conf-rumored`
(epistemic tier pill — use them; the book labels confidence).

**Plain code block**:
```html
<div class="code-cap">jax — fused MoE dispatch (illustrative)</div>
<pre class="code"><code># real, runnable-looking code here
group_sizes = jnp.bincount(expert_idx, length=num_experts)
out = jax.lax.ragged_dot(x_sorted, w_experts, group_sizes)
</code></pre>
```

**Annotated code** (margin notes; use for the 1–2 most important snippets):
```html
<div class="codecard">
  <div class="src-meta"><span class="file">src/maxtext/layers/moe.py</span><span class="tag">RoutedMoE.sparse_matmul</span><span class="tag">trimmed</span></div>
  <div class="annot-grid">
<pre class="code"><code>def sparse_matmul(self, x, w, group_sizes):   # [1]
    return jax.lax.ragged_dot(x, w, group_sizes)  # [2]
</code></pre>
    <div class="annot-list">
      <div class="annot"><p><span class="badge">1</span><span class="anh">Grouped, not batched.</span> Each expert sees a variable row count …</p></div>
      <div class="annot"><p><span class="badge">2</span><span class="anh">One ragged kernel.</span> Replaces a Python loop over experts …</p></div>
    </div>
  </div>
</div>
```

**Pull quote** (one per chapter, for the thesis):
```html
<div class="pull">At 100k GB200 you are communication-bound on the MoE all-to-all — keep expert parallelism inside the NVLink domain.</div>
```

**Table** (style every table this way; add `class="num"` to numeric cells):
```html
<div class="tablewrap"><table class="data">
  <thead><tr><th>Framework</th><th class="num">MoE MFU</th><th>Notes</th></tr></thead>
  <tbody><tr><td>Megatron-Core</td><td class="num">~23%</td><td>best published Blackwell MoE</td></tr></tbody>
  <caption>Dense-FP8 MFU; quote against dense peak.</caption>
</table></div>
```

**Sources** (end every chapter):
```html
<div class="sources"><h3>Sources</h3><ul>
  <li><a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report</a></li>
</ul></div>
```

## Figure library (data-driven — emit `<figure>` + JSON spec)

Skeleton — `data-fig` selects the type; the `<script class="spec">` carries the data; `book.js` renders the SVG:
```html
<figure class="fig" data-fig="TYPE">
  <figcaption class="figcap"><b>Figure N — Title.</b> One-sentence read of the figure.
    <span class="meta"><span>Regime · <b>…</b></span><span>Evidence · <b>…</b></span></span></figcaption>
  <script type="application/json" class="spec">{ ...SPEC... }</script>
</figure>
```
`fam` in any spec is one of the family suffixes above. Use `.narrow` on `<figure class="fig narrow">` for text-width figures.

### Types & exact JSON schemas (with a usable example each)

- **`barshare`** — 100%-stacked share bar (time/cost/comms/memory breakdown).
  `{ "title":"one decode step", "unit":"% of step", "items":[{"label":"MoE GEMM","value":33,"fam":"compute"},{"label":"all-to-all","value":18,"fam":"comms"}], "note":"optional" }`

- **`bars`** — vertical bar chart (compare a metric across categories).
  `{ "ylabel":"MFU %","bars":[{"label":"Megatron","value":23,"fam":"compute"},{"label":"DeepSeek-V3\nfloor","value":21,"fam":"moe"}],"ref":{"y":50,"label":"sparse-peak myth"} }`

- **`line`** — line chart; `logx`/`logy` optional booleans.
  `{ "xlabel":"tokens","ylabel":"loss","logx":true,"series":[{"name":"6.4×data","fam":"data","pts":[[1e11,2.9],[1e12,2.5],[1e13,2.2]]}],"annotations":[{"x":1e13,"y":2.2,"text":"data wall"}] }`

- **`roofline`** — log-log roofline; `peak` FLOP/s, `bw` byte/s.
  `{ "peak":4.5e15,"bw":8e12,"points":[{"name":"MoE GEMM","ai":120,"perf":1.0e15,"fam":"compute"},{"name":"all-to-all","ai":3,"perf":2.4e13,"fam":"comms"}] }`

- **`membar`** — memory breakdown with weight/grad dtype toggle. `perParam` is a
  number (fixed bytes/param), or `"w"` (scales w/ dtype), `"opt"` (uses `optBytes`),
  `"act"` (uses `activGB`).
  `{ "paramsB":200,"activGB":200,"optBytes":8,"dtypes":[{"id":"bf16","w":2,"label":"BF16"},{"id":"fp8","w":1,"label":"FP8"},{"id":"fp4","w":0.5,"label":"FP4"}],"parts":[{"label":"Weights","perParam":"w","fam":"compute"},{"label":"Adam m+v","perParam":"opt","fam":"optim"},{"label":"Grads","perParam":"grad","fam":"comms"},{"label":"Activations","perParam":"act","fam":"mem"}] }`

- **`meshlayout`** — device-mesh factorization (EP×TP×PP×DP…).
  `{ "deviceTotal":100000,"axes":[{"id":"EP","name":"expert","size":8,"fam":"moe"},{"id":"TP","name":"tensor","size":8,"fam":"compute"},{"id":"PP","name":"pipeline","size":16,"fam":"data"},{"id":"DP","name":"data","size":98,"fam":"mem"}],"note":"EP×TP inside the NVL72 domain" }`

- **`pipeline`** — pipeline-bubble gantt with schedule toggle.
  `{ "stages":8,"microbatches":12,"schedules":["1f1b","dualpipe"] }`

- **`flow`** — node/edge dataflow on a grid (`col`,`row` 0-indexed). Set
  `"anim":true` for animated tokens along edges (great for all-to-all / pipelines / data pipeline).
  `{ "cols":4,"rows":1,"anim":true,"nodes":[{"id":"a","label":"router","fam":"moe","col":0,"row":0},{"id":"b","label":"dispatch\n(all-to-all)","fam":"comms","col":1,"row":0},{"id":"c","label":"expert GEMM","fam":"compute","col":2,"row":0},{"id":"d","label":"combine\n(all-to-all)","fam":"comms","col":3,"row":0}],"edges":[{"from":"a","to":"b"},{"from":"b","to":"c"},{"from":"c","to":"d"}] }`

- **`timeline`** — proportional horizontal timeline (training stages).
  `{ "segments":[{"label":"pretrain","frac":80,"fam":"data","sub":"~24T tok"},{"label":"midtrain","frac":15,"fam":"optim","sub":"anneal HQ"},{"label":"long-ctx","frac":5,"fam":"attn"}] }`

- **`compare`** — two-column comparison (+ optional rows table).
  `{ "left":{"title":"Megatron-Core","fam":"compute","items":["best published MoE MFU","NVIDIA-tuned"]},"right":{"title":"MaxText/JAX","fam":"data","items":["TPU-native","XLA:GPU risk on GB200"]},"rows":[{"k":"EP maturity","left":"high","right":"medium"}] }`

## Density target per chapter

- Header block + pull quote + Sources (required).
- 3–7 figures (mix types; at least one interactive — `membar`/`pipeline` — or
  animated `flow`).
- 4–8 callouts, 1 stat strip, 2–4 code blocks (≥1 annotated where apt),
  family chips on key terms, confidence pills on leaked/rumored claims.
- Keep all source prose and links.

## Output

Write the fragment to the exact file path given. It must start with the
`<p class="eyebrow">` header and end with the `</div>` of `.sources`. No page
chrome. Reply with the file path and an approximate word count.
