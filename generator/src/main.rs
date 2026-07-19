// atlasgen — static site generator for The Frontier Pretraining Atlas.
// Reads content/*.md (front matter, 25 chapters, appendix), emits plain
// HTML + one stylesheet at the repo root. No JavaScript anywhere.

use pulldown_cmark::{html, Options, Parser};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Content hash of style.css, used to cache-bust the stylesheet URL.
static CSS_V: OnceLock<String> = OnceLock::new();

const SITE_NAME: &str = "The Frontier Pretraining Atlas";
const REPO_URL: &str = "https://github.com/ShivanshuPurohit/frontier-pretraining-atlas";
const SNAPSHOT: &str = "July 13, 2026 snapshot";

const PARTS: &[(u32, u32, &str, &str)] = &[
    (1, 4, "Part I", "The Evidence Base"),
    (5, 11, "Part II", "The Model"),
    (12, 15, "Part III", "The Data"),
    (16, 20, "Part IV", "The Machine"),
    (21, 23, "Part V", "The Run"),
    (24, 25, "Part VI", "The Ledger"),
];

const PROVENANCE_TAGS: &[&str] = &[
    "PAPER",
    "OFFICIAL-BLOG",
    "OFFICIAL-STATEMENT",
    "OFFICIAL X POST",
    "OFFICIAL-X-POST",
    "OFFICIAL-VIDEO",
    "OFFICIAL-INTERVIEW",
    "OFFICIAL-DOCS",
    "SPOKEN-CLAIM",
    "CREDIBLE-HEARSAY",
    "EPOCH-ESTIMATE",
    "ANALYSIS",
    "BENCHMARK",
    "TALK",
    "DOC",
    "DOCS",
    "CODE",
    "RUMOR",
    "FORUM",
    "LIVE-DOC",
    "CONFLICT",
    "CHECK",
    "DERIVED",
];

const FAVICON: &str = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23245f4f'/%3E%3Cg fill='%23f6f4ec'%3E%3Ccircle cx='16' cy='16' r='5' opacity='.32'/%3E%3Ccircle cx='32' cy='16' r='5'/%3E%3Ccircle cx='48' cy='16' r='5' opacity='.32'/%3E%3Ccircle cx='16' cy='32' r='5' opacity='.32'/%3E%3Ccircle cx='32' cy='32' r='5' opacity='.32'/%3E%3Ccircle cx='48' cy='32' r='5' opacity='.32'/%3E%3Ccircle cx='16' cy='48' r='5'/%3E%3Ccircle cx='32' cy='48' r='5' opacity='.32'/%3E%3Ccircle cx='48' cy='48' r='5' opacity='.32'/%3E%3C/g%3E%3C/svg%3E";

struct Doc {
    file: String,
    eyebrow: String,
    title: String,
    subtitle: Option<String>,
    body_md: String,
    blurb: String,
    pos: String,
    anchor: String, // id within book.html
    chp: String,    // section/table numbering prefix: "16", "FM", "A"
}

fn main() {
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let content = root.join("content");

    let css = fs::read(root.join("assets/style.css")).expect("stylesheet");
    let mut h: u64 = 5381;
    for b in &css {
        h = h.wrapping_mul(33) ^ (*b as u64);
    }
    CSS_V.set(format!("{:x}", h & 0xffff_ffff)).unwrap();

    // ---- front matter: title, subtitle, body, per-chapter blurbs ----
    let fm_raw = fs::read_to_string(content.join("00-front-matter.md")).expect("front matter");
    let mut fm_lines = fm_raw.lines();
    let book_title = fm_lines
        .next()
        .and_then(|l| l.strip_prefix("# "))
        .expect("front matter H1")
        .trim()
        .to_string();
    let after_h1 = &fm_raw[fm_raw.find('\n').unwrap() + 1..];
    let sub_line = after_h1
        .lines()
        .find(|l| l.starts_with("### "))
        .expect("front matter subtitle");
    let book_subtitle = sub_line.strip_prefix("### ").unwrap().trim().to_string();
    let sub_pos = after_h1.find(sub_line).unwrap();
    let fm_body_full = &after_h1[sub_pos + sub_line.len()..];

    let toc_pos = fm_body_full.find("## Table of Contents").expect("ToC section");
    let toc_section = &fm_body_full[toc_pos..];
    let fm_body = trim_rules(&fm_body_full[..toc_pos]);

    let mut blurbs: Vec<(u32, String)> = Vec::new();
    for line in toc_section.lines() {
        if let Some(rest) = line.strip_prefix("**Chapter ") {
            if let Some(sp) = rest.find(' ') {
                if let Ok(num) = rest[..sp].parse::<u32>() {
                    if let Some(dot) = rest.find(".**") {
                        blurbs.push((num, rest[dot + 3..].trim().to_string()));
                    }
                }
            }
        }
    }
    let blurb_for = |n: u32| -> String {
        blurbs
            .iter()
            .find(|(k, _)| *k == n)
            .map(|(_, b)| b.clone())
            .unwrap_or_default()
    };

    // Pull the opening paragraph of "What This Book Is" for the cover lede.
    let lede_md = fm_body
        .split("## What This Book Is")
        .nth(1)
        .and_then(|s| s.split("\n\n").find(|p| !p.trim().is_empty()))
        .unwrap_or("")
        .trim()
        .to_string();

    // ---- assemble the reading order ----
    let mut docs: Vec<Doc> = Vec::new();

    docs.push(Doc {
        file: "front-matter.html".into(),
        eyebrow: "Front matter".into(),
        title: book_title.clone(),
        subtitle: Some(book_subtitle.clone()),
        body_md: fm_body.clone(),
        blurb: "The question the book answers, how to read it, three reading paths, and the \
                executive summary."
            .into(),
        pos: "Front matter".into(),
        anchor: "front-matter".into(),
        chp: "FM".into(),
    });

    let mut chapter_files: Vec<String> = fs::read_dir(&content)
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.starts_with("ch-").then_some(name)
        })
        .collect();
    chapter_files.sort();

    for name in &chapter_files {
        let raw = fs::read_to_string(content.join(name)).unwrap();
        let h1 = raw.lines().next().and_then(|l| l.strip_prefix("# ")).unwrap_or("").trim();
        let (label, title) = h1.split_once(" — ").expect("chapter H1 format");
        let num: u32 = label.strip_prefix("Chapter ").unwrap().parse().unwrap();
        let body = trim_rules(&raw[raw.find('\n').unwrap() + 1..]);
        let part = PARTS
            .iter()
            .find(|&&(lo, hi, _, _)| num >= lo && num <= hi)
            .map(|&(_, _, pn, pt)| format!(" · {} — {}", pn, pt))
            .unwrap_or_default();
        docs.push(Doc {
            file: format!("{}.html", name.strip_suffix(".md").unwrap()),
            eyebrow: format!("Chapter {}{}", num, part),
            title: title.to_string(),
            subtitle: None,
            body_md: body,
            blurb: blurb_for(num),
            pos: format!("{} / 25", num),
            anchor: format!("ch-{}", num),
            chp: num.to_string(),
        });
    }

    let apx_raw = fs::read_to_string(content.join("99-appendix-hard-numbers.md")).unwrap();
    let apx_h1 = apx_raw.lines().next().and_then(|l| l.strip_prefix("# ")).unwrap().trim();
    let apx_title = apx_h1.split_once(": ").map(|(_, t)| t.to_string()).unwrap_or(apx_h1.into());
    docs.push(Doc {
        file: "appendix.html".into(),
        eyebrow: "Appendix A".into(),
        title: apx_title,
        subtitle: None,
        body_md: trim_rules(&apx_raw[apx_raw.find('\n').unwrap() + 1..]),
        blurb: "Every load-bearing number in the book — silicon, networking, MFU, cost, \
                reliability, data — in one master table, each row carrying its provenance \
                grade and source."
            .into(),
        pos: "Appendix A".into(),
        anchor: "appendix".into(),
        chp: "A".into(),
    });

    // ---- word count for the cover meta line ----
    let words: usize = docs.iter().map(|d| d.body_md.split_whitespace().count()).sum();
    let words_disp = format!("≈{}k words", (words + 500) / 1000);

    // ---- emit chapter pages ----
    for i in 0..docs.len() {
        let d = &docs[i];
        let (body_html, sections) = md_to_html(&d.body_md, "", &d.chp);
        let prev = if i > 0 { Some(&docs[i - 1]) } else { None };
        let next = docs.get(i + 1);

        let mut article = String::new();
        article.push_str(&format!(
            "<article style=\"--chp:'{}'\">\n<header class=\"chapter-head\">\n",
            d.chp
        ));
        article.push_str(&format!("<p class=\"eyebrow\">{}</p>\n", esc(&d.eyebrow)));
        article.push_str(&format!("<h1 class=\"chapter-title\">{}</h1>\n", esc(&d.title)));
        if let Some(s) = &d.subtitle {
            article.push_str(&format!("<p class=\"chapter-sub\">{}</p>\n", esc(s)));
        }
        article.push_str(&format!("<p class=\"vitals\">{}</p>\n", vitals(&sections, &body_html, &d.body_md)));
        article.push_str("</header>\n");

        if sections.len() >= 3 {
            article.push_str(
                "<details class=\"contents\"><summary>Contents</summary><ol>\n",
            );
            for (id, text) in &sections {
                // `text` is extracted from rendered HTML — already escaped
                article.push_str(&format!("<li><a href=\"#{}\">{}</a></li>\n", id, text));
            }
            article.push_str("</ol></details>\n");
        }

        article.push_str(&body_html);
        article.push_str("\n</article>\n");

        // pager
        article.push_str("<nav class=\"pager\">\n");
        match prev {
            Some(p) => article.push_str(&format!(
                "<div class=\"prev\"><span class=\"label\">Previous</span><a href=\"{}\">{}</a></div>\n",
                p.file,
                esc(&short_label(p))
            )),
            None => article.push_str("<div class=\"prev\"></div>\n"),
        }
        article.push_str("<div class=\"up\"><a href=\"index.html\">Contents</a></div>\n");
        match next {
            Some(n) => article.push_str(&format!(
                "<div class=\"next\"><span class=\"label\">Next</span><a href=\"{}\">{}</a></div>\n",
                n.file,
                esc(&short_label(n))
            )),
            None => article.push_str("<div class=\"next\"></div>\n"),
        }
        article.push_str("</nav>\n");

        let page_title = if d.file == "front-matter.html" {
            format!("Front Matter · {}", SITE_NAME)
        } else {
            format!("{} · {}", d.title, SITE_NAME)
        };
        let html_page = shell(
            &page_title,
            &d.blurb,
            &article,
            Some(&d.pos),
            &rail(&docs, Some(&d.chp)),
        );
        fs::write(root.join(&d.file), html_page).unwrap();
    }

    // ---- index ----
    let (lede_html, _) = md_to_html(&lede_md, "", "");
    let mut idx = String::new();
    idx.push_str("<div class=\"cover\">\n");
    let title_2tone = esc(&book_title).replacen(
        "Frontier Pretraining",
        "<span class=\"tt\">Frontier Pretraining</span>",
        1,
    );
    idx.push_str(&format!("<h1 class=\"book-title\">{}</h1>\n", title_2tone));
    idx.push_str(&format!("<p class=\"book-sub\">{}</p>\n", esc(&book_subtitle)));
    idx.push_str(
        "<div class=\"byline\">\n\
         <div><span class=\"bl\">Author</span>Shivanshu Purohit</div>\n\
         <div><span class=\"bl\">Research &amp; drafting</span>Claude</div>\n\
         <div><span class=\"bl\">Snapshot</span>July 13, 2026</div>\n\
         </div>\n",
    );
    idx.push_str(&format!(
        "<table class=\"spec\">\n\
         <tr><th>Model frame</th><td>10T total · 200B active · MoE</td></tr>\n\
         <tr><th>Fleet frame</th><td>100,000 GB200 NVL72</td></tr>\n\
         <tr><th>Scope</th><td>25 chapters · front matter · appendix</td></tr>\n\
         <tr><th>Length</th><td>{}</td></tr>\n\
         <tr><th>Sources</th><td>250+ · every claim carries its source inline</td></tr>\n\
         <tr><th>Edition</th><td>v2.3 — expanded to 25 chapters, prose rewritten (July 19, 2026)</td></tr>\n\
         </table>\n",
        words_disp
    ));
    idx.push_str(&format!("<div class=\"lede\">{}</div>\n", lede_html));
    idx.push_str("<p class=\"begin\"><a href=\"front-matter.html\">Begin with the front matter&thinsp;&rarr;</a></p>\n");
    idx.push_str("</div>\n");

    idx.push_str("<div class=\"toc\">\n");
    idx.push_str(&toc_item("front-matter.html", "FM", &docs[0].title, &docs[0].blurb));
    for &(lo, hi, pn, pname) in PARTS {
        idx.push_str(&format!(
            "<p class=\"part-label\"><span class=\"pn\">{}</span><span class=\"pt\">{}</span></p>\n",
            pn, pname
        ));
        for d in &docs {
            if let Ok(n) = d.chp.parse::<u32>() {
                if n >= lo && n <= hi {
                    idx.push_str(&toc_item(&d.file, &n.to_string(), &d.title, &d.blurb));
                }
            }
        }
    }
    let apx = docs.last().unwrap();
    idx.push_str("<p class=\"part-label\"><span class=\"pn\">Appendix</span></p>\n");
    idx.push_str(&toc_item(&apx.file, "A", &apx.title, &apx.blurb));
    idx.push_str("</div>\n");

    idx.push_str(
        "<div class=\"citebox\">\n<p class=\"label\">Cite this work</p>\n\
         <pre>@book{purohit2026frontierpretraining,\n  \
         title  = {The State of Frontier Pretraining, July 2026},\n  \
         author = {Purohit, Shivanshu and Claude},\n  \
         year   = {2026},\n  \
         note   = {https://shivanshupurohit.github.io/frontier-pretraining-atlas/}\n}</pre>\n</div>\n",
    );

    let index_page = shell(
        &format!("{} — {}", SITE_NAME, book_subtitle),
        &book_subtitle,
        &idx,
        None,
        &rail(&docs, None),
    );
    fs::write(root.join("index.html"), index_page).unwrap();

    // ---- single-page edition ----
    let mut book = String::new();
    book.push_str("<div class=\"cover onepage\">\n");
    book.push_str(&format!("<h1 class=\"book-title\">{}</h1>\n", esc(&book_title)));
    book.push_str(&format!("<p class=\"book-sub\">{}</p>\n", esc(&book_subtitle)));
    book.push_str(&format!(
        "<p class=\"book-meta\">Single-page edition<span>·</span>{}<span>·</span>{}</p>\n",
        words_disp, SNAPSHOT
    ));
    book.push_str("<nav class=\"onepage-toc\"><ol>\n");
    for d in &docs {
        book.push_str(&format!("<li><a href=\"#{}\">{}</a></li>\n", d.anchor, esc(&short_label(d))));
    }
    book.push_str("</ol></nav>\n</div>\n");

    for d in &docs {
        if let Ok(n) = d.chp.parse::<u32>() {
            if let Some(&(_, _, pn, pname)) = PARTS.iter().find(|&&(lo, _, _, _)| lo == n) {
                book.push_str(&format!(
                    "<div class=\"part-divider\"><span class=\"pn\">{}</span><span class=\"pt\">{}</span></div>\n",
                    pn, pname
                ));
            }
        }
        let (body_html, _) = md_to_html(&d.body_md, &format!("{}-", d.anchor), &d.chp);
        book.push_str(&format!(
            "<article id=\"{}\" style=\"--chp:'{}'\">\n<header class=\"chapter-head\">\n",
            d.anchor, d.chp
        ));
        book.push_str(&format!("<p class=\"eyebrow\">{}</p>\n", esc(&d.eyebrow)));
        // the cover directly above already carries the book title + subtitle
        if d.anchor != "front-matter" {
            book.push_str(&format!("<h1 class=\"chapter-title\">{}</h1>\n", esc(&d.title)));
        }
        book.push_str("</header>\n");
        book.push_str(&body_html);
        book.push_str("\n</article>\n");
    }
    let book_page = shell(
        &format!("Single-page edition · {}", SITE_NAME),
        "The complete book on one page.",
        &book,
        Some("Single page"),
        &rail(&docs, None),
    );
    fs::write(root.join("book.html"), book_page).unwrap();

    // ---- static files ----
    fs::copy(root.join("assets/style.css"), root.join("style.css")).unwrap();
    fs::write(root.join(".nojekyll"), "").unwrap();

    println!(
        "built {} pages + index + single-page edition ({} words)",
        docs.len(),
        words
    );
}

// ---------- rendering helpers ----------

fn shell(title: &str, desc: &str, main: &str, pos: Option<&str>, rail: &str) -> String {
    let topbar = match pos {
        Some(p) => format!(
            "<header class=\"topbar\"><a class=\"home\" href=\"index.html\">{}</a><span class=\"pos\">{}</span></header>\n",
            SITE_NAME, esc(p)
        ),
        None => format!(
            "<header class=\"topbar\"><span class=\"home\">{}</span></header>\n",
            SITE_NAME
        ),
    };
    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#101312">
<title>{}</title>
<meta name="description" content="{}">
<link rel="icon" href="{}">
<link rel="stylesheet" href="style.css?v={}">
</head>
<body>
<div class="frame">
{}<div class="pagecol">
{}<main>
{}</main>
<footer class="colophon">
<span>{} · {}</span>
<span><a href="book.html">Single-page edition</a></span>
<span><a href="{}">Source</a></span>
<span>No JavaScript was harmed or used in the making of this site.</span>
</footer>
</div>
</div>
</body>
</html>
"##,
        esc(title),
        esc(desc),
        FAVICON,
        CSS_V.get().map(|s| s.as_str()).unwrap_or("0"),
        rail,
        topbar,
        main,
        SITE_NAME,
        SNAPSHOT,
        REPO_URL
    )
}

/// The sticky left rail: wordmark, a 5×5 grid of chapter cells (FM, 1–23, A),
/// and secondary links. `current` is the chp key of the open page, if any.
fn rail(docs: &[Doc], current: Option<&str>) -> String {
    let mut cells = String::new();
    for d in docs {
        let on = current == Some(d.chp.as_str());
        cells.push_str(&format!(
            "<a class=\"cell{}\" href=\"{}\" title=\"{}\">{}</a>\n",
            if on { " on" } else { "" },
            d.file,
            esc(&short_label(d)),
            d.chp
        ));
    }
    format!(
        r#"<aside class="rail">
<a class="wordmark" href="index.html">
<svg viewBox="0 0 64 64" aria-hidden="true"><g fill="currentColor"><circle cx="16" cy="16" r="6" opacity=".3"/><circle cx="32" cy="16" r="6"/><circle cx="48" cy="16" r="6" opacity=".3"/><circle cx="16" cy="32" r="6" opacity=".3"/><circle cx="32" cy="32" r="6" opacity=".3"/><circle cx="48" cy="32" r="6" opacity=".3"/><circle cx="16" cy="48" r="6"/><circle cx="32" cy="48" r="6" opacity=".3"/><circle cx="48" cy="48" r="6" opacity=".3"/></g></svg>
<span>Frontier<br>Pretraining<br>Atlas</span>
</a>
<nav class="cellgrid">
{}</nav>
<nav class="rail-links">
<a href="index.html">Cover</a>
<a href="book.html">Single page</a>
<a href="{}">Source</a>
</nav>
<p class="rail-note">Snapshot<br>2026-07-13</p>
</aside>
"#,
        cells, REPO_URL
    )
}

/// Mono metadata line under a chapter title: "11 sections · 3 tables · ≈4.4k words".
fn vitals(sections: &[(String, String)], body_html: &str, body_md: &str) -> String {
    let mut parts = Vec::new();
    if sections.len() > 1 {
        parts.push(format!("{} sections", sections.len()));
    }
    let tables = body_html.matches("<figure class=\"table\"").count();
    if tables == 1 {
        parts.push("1 table".into());
    } else if tables > 1 {
        parts.push(format!("{} tables", tables));
    }
    let words = body_md.split_whitespace().count();
    parts.push(format!("≈{:.1}k words", words as f64 / 1000.0));
    parts.join(" · ")
}

fn toc_item(href: &str, num: &str, title: &str, blurb: &str) -> String {
    format!(
        "<div class=\"toc-item\"><span class=\"toc-num\">{}</span><a class=\"toc-title\" href=\"{}\">{}</a><p class=\"toc-blurb\">{}</p></div>\n",
        num,
        href,
        esc(title),
        esc(blurb)
    )
}

fn short_label(d: &Doc) -> String {
    if d.chp.parse::<u32>().is_ok() {
        let t = match d.title.split_once(':') {
            Some((head, _)) => head,
            None => &d.title,
        };
        format!("{}. {}", d.chp, t)
    } else if d.file == "front-matter.html" {
        "Front matter".into()
    } else {
        format!("{} — {}", d.eyebrow, d.title)
    }
}

fn md_to_html(md: &str, id_prefix: &str, tprefix: &str) -> (String, Vec<(String, String)>) {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_SMART_PUNCTUATION);
    let parser = Parser::new_ext(md, opts);
    let mut out = String::with_capacity(md.len() * 3 / 2);
    html::push_html(&mut out, parser);

    let (out, sections) = add_heading_ids(&out, id_prefix);
    let out = wrap_tables(&out, tprefix);
    let out = chipify(&out);
    (out, sections)
}

/// Wrap every table in a numbered figure with a horizontal-scroll container.
fn wrap_tables(html: &str, tprefix: &str) -> String {
    let mut out = String::with_capacity(html.len() + 2048);
    let mut n = 0;
    let mut rest = html;
    while let Some(pos) = rest.find("<table>") {
        n += 1;
        out.push_str(&rest[..pos]);
        if tprefix.is_empty() {
            out.push_str("<figure class=\"table\"><div class=\"scroll\"><table>");
        } else {
            out.push_str(&format!(
                "<figure class=\"table\" id=\"table-{}-{}\"><figcaption class=\"table-no\">Table {}.{}</figcaption><div class=\"scroll\"><table>",
                tprefix.to_lowercase(), n, tprefix, n
            ));
        }
        rest = &rest[pos + "<table>".len()..];
    }
    out.push_str(rest);
    out.replace("</table>", "</table></div></figure>")
}

fn add_heading_ids(html: &str, prefix: &str) -> (String, Vec<(String, String)>) {
    let mut out = String::with_capacity(html.len() + 1024);
    let mut sections = Vec::new();
    let mut used: HashSet<String> = HashSet::new();
    let mut rest = html;
    loop {
        let p2 = rest.find("<h2>");
        let p3 = rest.find("<h3>");
        let (pos, lvl) = match (p2, p3) {
            (None, None) => break,
            (Some(a), None) => (a, 2),
            (None, Some(b)) => (b, 3),
            (Some(a), Some(b)) => {
                if a < b {
                    (a, 2)
                } else {
                    (b, 3)
                }
            }
        };
        out.push_str(&rest[..pos]);
        rest = &rest[pos..];
        let close = if lvl == 2 { "</h2>" } else { "</h3>" };
        let Some(end) = rest.find(close) else {
            break;
        };
        let inner = &rest[4..end];
        let text = strip_tags(inner);
        let base = format!("{}{}", prefix, slugify(&text));
        let mut id = base.clone();
        let mut n = 1;
        while !used.insert(id.clone()) {
            n += 1;
            id = format!("{}-{}", base, n);
        }
        out.push_str(&format!("<h{lvl} id=\"{id}\">{inner}</h{lvl}>"));
        if lvl == 2 {
            sections.push((id, text));
        }
        rest = &rest[end + close.len()..];
    }
    out.push_str(rest);
    (out, sections)
}

/// Convert bracketed provenance tags ([PAPER], [CREDIBLE-HEARSAY], …) into
/// styled chips, skipping <pre> and <code> content.
fn chipify(html: &str) -> String {
    let bytes = html.as_bytes();
    let mut out = String::with_capacity(html.len() + 2048);
    let mut i = 0;
    let mut in_pre = false;
    let mut in_code = false;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'<' {
            let end = html[i..].find('>').map(|e| i + e + 1).unwrap_or(bytes.len());
            let tag = &html[i..end];
            if tag.starts_with("<pre") {
                in_pre = true;
            } else if tag.starts_with("</pre") {
                in_pre = false;
            } else if tag.starts_with("<code") {
                in_code = true;
            } else if tag.starts_with("</code") {
                in_code = false;
            }
            out.push_str(tag);
            i = end;
            continue;
        }
        if b == b'[' && !in_pre && !in_code {
            if let Some(close) = html[i + 1..].find(']') {
                if close <= 200 {
                    let inner = &html[i + 1..i + 1 + close];
                    if !inner.contains('<') && !inner.contains('[') && !inner.contains('\n') {
                        if let Some(rendered) = render_cite(inner) {
                            out.push_str(&rendered);
                            i = i + close + 2;
                            continue;
                        }
                    }
                }
            }
        }
        // copy one UTF-8 scalar
        let ch_len = utf8_len(b);
        out.push_str(&html[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// Render the inside of a bracketed provenance citation, or None to leave it
/// as literal text. Handles `[PAPER]`, `[PAPER, Oct 2025]`,
/// `[REPORTED-BY:The Information]`; qualifiers longer than ~56 chars stay
/// literal.
fn render_cite(inner: &str) -> Option<String> {
    fn loud(t: &str) -> bool {
        matches!(t, "RUMOR" | "FORUM" | "CONFLICT" | "CHECK")
    }

    if PROVENANCE_TAGS.contains(&inner) {
        return Some(format!(
            "<span class=\"cite{}\">[{}]</span>",
            if loud(inner) { " loud" } else { "" },
            inner
        ));
    }
    if let Some(rest) = inner.strip_prefix("REPORTED-BY:") {
        let rest = rest.trim();
        if !rest.is_empty() && rest.len() <= 48 {
            return Some(format!("<span class=\"cite\">[{}]</span>", inner));
        }
        return None;
    }
    // longest known tag followed by a comma/semicolon qualifier
    let mut best: Option<&str> = None;
    for t in PROVENANCE_TAGS {
        if inner.starts_with(t)
            && matches!(inner.as_bytes().get(t.len()), Some(b',') | Some(b';'))
            && best.map_or(true, |b| t.len() > b.len())
        {
            best = Some(t);
        }
    }
    let t = best?;
    let suffix = inner[t.len() + 1..].trim();
    if suffix.is_empty() || suffix.len() > 56 {
        return None;
    }
    Some(format!(
        "<span class=\"cite{}\">[{}]</span>",
        if loud(t) { " loud" } else { "" },
        inner
    ))
}

fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ => {
                if !in_tag {
                    out.push(c)
                }
            }
        }
    }
    out
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Trim leading/trailing horizontal rules and whitespace from a markdown body.
fn trim_rules(s: &str) -> String {
    let mut t = s.trim();
    if let Some(rest) = t.strip_prefix("---") {
        t = rest.trim_start();
    }
    if let Some(rest) = t.strip_suffix("---") {
        t = rest.trim_end();
    }
    t.to_string()
}
