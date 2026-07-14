# The Frontier Pretraining Atlas

**The State of Frontier Pretraining, July 2026** — a technical planning
reference for a 10-trillion-total / 200-billion-active MoE pretraining run on
roughly 100,000 GB200 GPUs. 23 chapters, ~137k words, 250+ sources, every
claim carrying a provenance tag.

Read it at <https://shivanshupurohit.github.io/frontier-pretraining-atlas/>.

## How this site is built

Plain HTML and one hand-written stylesheet. **No JavaScript, no webfonts, no
external requests of any kind.** Dark mode follows the OS.

- `content/` — the book, one Markdown file per chapter (front matter,
  chapters 1–23, appendix)
- `generator/` — a small Rust static-site generator (pulldown-cmark)
- `assets/style.css` — the theme; copied to `style.css` at build time

Rebuild after editing content:

```sh
cargo run --release --manifest-path generator/Cargo.toml
```

Generated pages live at the repo root so GitHub Pages serves `main` directly.

## The theme

A lab-monograph design, intended to be reused for future posts:

- **Two voices.** Prose is set in a book serif (Iowan Old Style → Palatino →
  Georgia). Everything *around* the prose — chapter eyebrows, section
  markers, table captions, citations, nav, metadata — is the system
  monospace. The mono is the "apparatus" voice; the serif is the text.
- **Numbered everything.** Sections get `§ 16.3` markers and tables get
  `TABLE 16.2` captions automatically (CSS counters + the generator), so any
  part of the book can be cited precisely.
- **Provenance as typography.** The book's bracketed source tags
  (`[PAPER, 02/2024]`, `[CREDIBLE-HEARSAY, paraphrase]`) render as small
  green mono citations; `[RUMOR]`/`[CONFLICT]` render terracotta.
- **One accent.** Pine green (`#245f4f` light / `#85bda9` dark), terracotta
  reserved for warnings. Ink on bone in light mode, sage on charcoal in dark.
- **No-JS interactivity.** Collapsible per-chapter contents use native
  `<details>`. Future posts needing math get build-time KaTeX (HTML+CSS
  output, still zero client JS); CSS/SVG handles animation where necessary.

Each chapter page carries prev/next navigation; `book.html` is the entire
book on one page.
