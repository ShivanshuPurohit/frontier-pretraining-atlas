#!/usr/bin/env python3
"""Build the Frontier Pretraining Atlas static site.

Subcommands:
  baseline   pandoc every chapter .md -> parts/<id>.baseline.html (clean body
             fragment: headings promoted, tables styled). A safety net / reference.
  wrap       wrap every parts/<id>.main.html (inner-<main> content authored by a
             chapter agent) into the canonical page template -> <id>.html
  standalone inline css+js and concat all pages into one offline file
  all        wrap + standalone

Pure stdlib + the `pandoc` binary. No external Python deps.
"""
import os, re, sys, json, html, glob, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
SECT = os.path.normpath(os.path.join(SITE, "..", "sections"))
PARTS = os.path.join(SITE, "parts")
os.makedirs(PARTS, exist_ok=True)

SITE_URL = "https://shivanshupurohit.github.io/frontier-pretraining-atlas"
BOOK_TITLE = "The Frontier Pretraining Atlas"
BOOK_SUB = "a field manual for the 10-trillion-parameter MoE"

# id, source-md, output-html, short title, concept family, est. minutes
PAGES = [
    ("ch01", "01-landscape.md",            "ch01.html", "The frontier landscape",          "norm",    18),
    ("ch02", "02-hardware.md",             "ch02.html", "The 100k-GPU machine",            "compute", 18),
    ("ch03", "03-networking.md",           "ch03.html", "Networking & collectives",        "comms",   22),
    ("ch04", "04-parallelism.md",          "ch04.html", "Parallelism strategy",            "data",    24),
    ("ch05", "05-moe-systems.md",          "ch05.html", "MoE systems engineering",         "moe",     24),
    ("ch06", "06-frameworks.md",           "ch06.html", "Framework shoot-out",             "compute", 24),
    ("ch07", "07-scaling-book.md",         "ch07.html", "The Scaling Book, applied",       "data",    22),
    ("ch08", "08-attention-kernels.md",    "ch08.html", "Attention & kernels",             "attn",    20),
    ("ch09", "09-low-precision.md",        "ch09.html", "Low-precision training",          "prec",    22),
    ("ch10", "10-optimizers.md",           "ch10.html", "Optimizers & HP transfer",        "optim",   20),
    ("ch11", "11-stability.md",            "ch11.html", "Training stability",              "comms",   18),
    ("ch12", "12-scaling-laws.md",         "ch12.html", "Modern scaling laws",             "data",    22),
    ("ch13", "13-data-curation.md",        "ch13.html", "Data curation",                   "data",    24),
    ("ch14", "14-dataloading.md",          "ch14.html", "Dataloading & I/O",               "mem",     20),
    ("ch15", "15-fault-tolerance.md",      "ch15.html", "Fault tolerance",                 "mem",     20),
    ("ch16", "16-midtraining.md",          "ch16.html", "Midtraining & long-context",      "optim",   16),
    ("ch17", "17-chinese-frontier.md",     "ch17.html", "The Chinese frontier",            "moe",     24),
    ("ch18", "18-theory.md",               "ch18.html", "The theory corner",               "attn",    16),
    ("ch19", "19-mlperf.md",               "ch19.html", "Public throughput signals",       "compute", 13),
    ("ch20", "20-punchlist.md",            "ch20.html", "The punch-list",                  "comms",   24),
    ("apxA", "AX-scaling-cheatsheet.md",   "apxA.html", "Appendix A · Cheat-sheet",        "data",    14),
    ("apxB", "AX-B-semianalysis.md",       "apxB.html", "Appendix B · Operational reality","norm",    24),
    ("apxC", "AX-C-stas-mlengineering.md", "apxC.html", "Appendix C · Ops playbook",       "mem",     22),
    ("apxD", "AX-D-bibliography.md",       "apxD.html", "Appendix D · Bibliography",       "norm",    10),
]
BYID = {p[0]: p for p in PAGES}


def promote_headings(h):
    # h2->h1, h3->h2, h4->h3, h5->h4  (chapters use ## as the title level)
    for a, b in [("h4", "hX3"), ("h3", "hX2"), ("h2", "hX1"), ("h5", "hX4")]:
        h = re.sub(rf"<{a}(\b[^>]*)>", rf"<{b}\1>", h)
        h = h.replace(f"</{a}>", f"</hX{ {'h4':'3','h3':'2','h2':'1','h5':'4'}[a] }>")
    return h.replace("hX1", "h1").replace("hX2", "h2").replace("hX3", "h3").replace("hX4", "h4")


def style_tables(h):
    h = h.replace("<table>", '<table class="data">')
    # wrap each table in .tablewrap
    h = re.sub(r"(<table class=\"data\">.*?</table>)", r'<div class="tablewrap">\1</div>', h, flags=re.S)
    return h


def pandoc(md_path):
    out = subprocess.run(
        ["pandoc", "--from", "gfm", "--to", "html5", "--no-highlight", md_path],
        capture_output=True, text=True, check=True).stdout
    return style_tables(promote_headings(out))


def cmd_baseline():
    for pid, src, *_ in PAGES:
        p = os.path.join(SECT, src)
        if not os.path.exists(p):
            print("  MISSING", src); continue
        body = pandoc(p)
        open(os.path.join(PARTS, f"{pid}.baseline.html"), "w").write(body)
        print(f"  baseline {pid}: {len(body.split())} words of html")


def chrail():
    out = ['<a href="index.html">Contents</a>']
    for pid, *_ , in [(p[0],) for p in PAGES]:
        label = pid[2:] if pid.startswith("ch") else "·" + pid[3]
        out.append(f'<a href="{BYID[pid][2]}" data-rail="{pid}">{label}</a>')
    return '<div class="chrail">' + "".join(out) + "</div>"


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="robots" content="noindex, nofollow" />
<title>{title}</title>
<meta name="description" content="{desc}" />
<meta name="color-scheme" content="light dark" />
<meta name="theme-color" content="#f4f2ec" />
<link rel="icon" href="{favicon}" />
<link rel="stylesheet" href="book.css" />
</head>
<body>
<nav class="nav">
  <a class="home" href="index.html">{book} <small>· {sub}</small></a>
  <span class="spacer"></span>
  {chrail}
  <div class="progress-wrap"><div class="progress"></div></div>
  <button class="theme-btn" aria-label="Toggle dark mode">◑ theme</button>
</nav>
<main class="wrap page">
{body}
{chnav}
</main>
<script src="book.js"></script>
</body>
</html>
"""

FAVICON = ("data:image/svg+xml,"
           "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Crect width='32' height='32' rx='7' fill='%232f6fba'/%3E"
           "%3Ctext x='16' y='22' font-family='monospace' font-size='15' fill='white' "
           "text-anchor='middle' font-weight='bold'%3EFP%3C/text%3E%3C/svg%3E")


def chnav_for(i):
    prev = PAGES[i-1] if i > 0 else None
    nxt = PAGES[i+1] if i < len(PAGES)-1 else None
    pv = (f'<a class="prev" href="{prev[2]}"><span class="dir">← Previous</span>'
          f'<span class="t">{html.escape(prev[3])}</span></a>') if prev else \
         '<a class="prev disabled"><span class="dir">← Previous</span><span class="t">—</span></a>'
    nx = (f'<a class="next" href="{nxt[2]}"><span class="dir">Next →</span>'
          f'<span class="t">{html.escape(nxt[3])}</span></a>') if nxt else \
         '<a class="next disabled"><span class="dir">Next →</span><span class="t">—</span></a>'
    return f'<nav class="chnav">{pv}{nx}</nav>'


def cmd_wrap():
    rail = chrail()
    for i, (pid, src, out, title, fam, mins) in enumerate(PAGES):
        mp = os.path.join(PARTS, f"{pid}.main.html")
        if not os.path.exists(mp):
            print("  no main yet:", pid); continue
        body = open(mp).read()
        page = TEMPLATE.format(
            title=html.escape(f"{title} · {BOOK_TITLE}"),
            desc=html.escape(f"{title} — {BOOK_SUB}."),
            favicon=FAVICON, book=html.escape(BOOK_TITLE), sub=html.escape(BOOK_SUB),
            chrail=rail, body=body, chnav=chnav_for(i))
        # mark current in rail
        page = page.replace(f'data-rail="{pid}">', f'data-rail="{pid}" class="cur">')
        open(os.path.join(SITE, out), "w").write(page)
        print(f"  wrote {out} ({len(body.split())} words)")


def cmd_standalone():
    css = open(os.path.join(SITE, "book.css")).read()
    js = open(os.path.join(SITE, "book.js")).read()
    secs = []
    # index first
    idx = os.path.join(SITE, "index.html")
    pagelist = []
    if os.path.exists(idx):
        m = re.search(r"<main\b[^>]*>(.*)</main>", open(idx).read(), re.S)
        if m: pagelist.append(("top", "Contents", m.group(1)))
    for pid, src, out, title, fam, mins in PAGES:
        f = os.path.join(SITE, out)
        if not os.path.exists(f): continue
        m = re.search(r"<main\b[^>]*>(.*?)(?:<nav class=\"chnav\">)", open(f).read(), re.S) or \
            re.search(r"<main\b[^>]*>(.*)</main>", open(f).read(), re.S)
        if m: pagelist.append((pid, title, m.group(1)))
    linkmap = {"index.html": "#top"}
    for pid, *_ , in [(p[0],) for p in PAGES]: linkmap[BYID[pid][2]] = "#" + pid
    for pid, title, inner in pagelist:
        for a, b in linkmap.items(): inner = inner.replace(f'href="{a}"', f'href="{b}"')
        secs.append(f'<section id="{pid}" class="wrap page chapter-section">\n{inner}\n</section>')
    rail = '<div class="chrail">' + '<a href="#top">Contents</a>' + \
        "".join(f'<a href="#{p[0]}">{p[0][2:] if p[0].startswith("ch") else "·"+p[0][3]}</a>' for p in PAGES) + "</div>"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(BOOK_TITLE)} — {html.escape(BOOK_SUB)}</title>
<meta name="description" content="{html.escape(BOOK_TITLE)}: {html.escape(BOOK_SUB)}."/>
<meta name="color-scheme" content="light dark"/><link rel="icon" href="{FAVICON}"/>
<style>{css}</style></head><body>
<nav class="nav"><a class="home" href="#top">{html.escape(BOOK_TITLE)} <small>· {html.escape(BOOK_SUB)}</small></a>
<span class="spacer"></span>{rail}<div class="progress-wrap"><div class="progress"></div></div>
<button class="theme-btn" aria-label="Toggle dark mode">◑ theme</button></nav>
<main>
{chr(10).join(secs)}
</main><script>{js}</script></body></html>"""
    outp = os.path.join(SITE, "frontier-pretraining-atlas.html")
    open(outp, "w").write(doc)
    print(f"  standalone -> {outp} ({len(doc)//1024} KB, {len(pagelist)} sections)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if cmd == "baseline": cmd_baseline()
    elif cmd == "wrap": cmd_wrap()
    elif cmd == "standalone": cmd_standalone()
    elif cmd == "all": cmd_wrap(); cmd_standalone()
    else: print("usage: build.py [baseline|wrap|standalone|all]")
