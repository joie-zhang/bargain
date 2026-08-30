#!/usr/bin/env python3
"""
=============================================================================
Offline Graphics Triage Page Builder
=============================================================================

Builds ONE self-contained HTML file for triaging the image files in
overleaf/icml_aiwild_template/graphics/ that the compiled ICML AIWILD paper
does not reference.

Why this exists: the Streamlit viewer needs a live server plus an SSH tunnel
to the exact login node the server runs on. On Della, `della.princeton.edu`
round-robins to a different node than the one your shell is on, so the tunnel
silently points at a machine with nothing listening. This page has no server,
no port, and no tunnel -- scp it to your laptop and open it.

Images are embedded as downscaled JPEG data URIs, so the file works offline
and after the originals move. Decisions live in the browser's localStorage and
survive reloads. Export writes a decisions CSV and a staging shell script.

It reads exactly the file list in
docs/reproducibility/unreferenced_graphics_manifest.json. Files the paper
references are never included and can never be staged.

Usage:
    python scripts/build_graphics_triage_html.py
    python scripts/build_graphics_triage_html.py --max-px 1400 --quality 88

What it creates:
    docs/reproducibility/graphics_triage.html   (~11 MB, self-contained)

Then, from your laptop:
    scp <user>@della-vis2.princeton.edu:/scratch/gpfs/DANQIC/jz4391/bargain/\
docs/reproducibility/graphics_triage.html .
    open graphics_triage.html

Options:
    --max-px    Longest thumbnail edge in pixels (default 1200).
    --quality   JPEG quality 1-95 (default 82).
    --out       Output path.

Dependencies:
    Pillow; a manifest from scripts/build_unreferenced_graphics_manifest.py

=============================================================================
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "overleaf" / "icml_aiwild_template"
MANIFEST = PROJECT_ROOT / "docs" / "reproducibility" / "unreferenced_graphics_manifest.json"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "reproducibility" / "graphics_triage.html"
STAGE_DIR = "experiments/results/TO_DELETE_20260809/unreferenced_graphics"


def thumbnail_data_uri(path: Path, max_px: int, quality: int) -> str | None:
    """Downscaled JPEG data URI, or None if the file cannot be rasterized."""
    try:
        im = Image.open(path)
    except Exception:
        return None
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        im = flat
    elif im.mode != "RGB":
        im = im.convert("RGB")
    im.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def similar_referenced(path: str, referenced: list[str], n: int = 4) -> list[str]:
    stem = Path(path).stem
    same_dir = [r for r in referenced if Path(r).parent == Path(path).parent]
    pool = same_dir or referenced
    ranked = sorted(
        pool,
        key=lambda r: difflib.SequenceMatcher(None, stem, Path(r).stem).ratio(),
        reverse=True,
    )
    return [Path(r).name for r in ranked[:n]]


def build_items(max_px: int, quality: int) -> tuple[list[dict], dict]:
    data = json.loads(MANIFEST.read_text())
    unref, ref = data["unreferenced"], data["referenced"]
    items = []
    for i, e in enumerate(unref, 1):
        p = TEMPLATE_DIR / e["path"]
        uri = thumbnail_data_uri(p, max_px, quality) if p.suffix.lower() != ".pdf" else None
        items.append(
            {
                "path": e["path"],
                "name": Path(e["path"]).name,
                "dir": e["dir"],
                "kb": round(e["bytes"] / 1024, 1),
                "img": uri,
                "siblings": similar_referenced(e["path"], ref),
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(unref)} thumbnails")
    return items, data["counts"]


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unreferenced graphics triage</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e3e3e0;--card:#fff;
--keep:#1a7f4b;--del:#b3261e;--accent:#2b5cd9}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e8e6;--mut:#9a9a97;
--line:#2e3033;--card:#1f2124;--keep:#4ac585;--del:#f27a72;--accent:#7aa2f7}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:640}
.bar{flex:1;min-width:160px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);width:0}
.pill{font-size:12px;color:var(--mut);white-space:nowrap}
main{max-width:1180px;margin:0 auto;padding:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.imgwrap{display:flex;align-items:center;justify-content:center;min-height:380px;
background:#fff;border-radius:8px;overflow:auto}
@media(prefers-color-scheme:dark){.imgwrap{background:#f4f4f2}}
.imgwrap img{max-width:100%;height:auto;display:block}
.meta{display:flex;gap:16px;flex-wrap:wrap;align-items:baseline;margin-bottom:12px}
code{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);
padding:2px 6px;border-radius:4px;border:1px solid var(--line);word-break:break-all}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
button{font:inherit;font-size:14px;padding:9px 16px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}
button.keep{background:var(--keep);border-color:var(--keep);color:#fff;font-weight:600}
button.del{background:var(--del);border-color:var(--del);color:#fff;font-weight:600}
button:disabled{opacity:.4;cursor:not-allowed}
.side{margin-top:14px;font-size:13px;color:var(--mut)}
.side b{color:var(--fg);font-weight:600}
.side ul{margin:6px 0 0;padding-left:18px}
.state{font-weight:700;font-size:13px;letter-spacing:.03em}
.state.k{color:var(--keep)} .state.d{color:var(--del)} .state.u{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
.tile{border:2px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;background:#fff}
.tile.k{border-color:var(--keep)} .tile.d{border-color:var(--del);opacity:.55}
.tile img{width:100%;height:135px;object-fit:contain;display:block;background:#fff}
.tile p{margin:0;padding:6px 8px;font-size:11px;color:var(--mut);background:var(--card);
word-break:break-all}
.hide{display:none}
.note{font-size:12.5px;color:var(--mut);margin-top:10px}
.pdf{color:var(--mut);text-align:center;padding:40px 20px;font-size:14px}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;
border-radius:4px;padding:1px 5px;background:var(--bg)}
</style></head><body>
<header>
  <h1>Unreferenced graphics triage</h1>
  <div class="bar"><i id="bar"></i></div>
  <span class="pill" id="stat"></span>
  <button id="mode">Grid</button>
  <button id="exp">Export</button>
</header>
<main>
  <div id="single">
    <div class="card">
      <div class="meta">
        <span class="pill" id="pos"></span>
        <code id="path"></code>
        <span class="pill" id="size"></span>
        <span class="state" id="state"></span>
      </div>
      <div class="imgwrap" id="imgwrap"></div>
      <div class="row">
        <button class="keep" id="bk">Keep &nbsp;<kbd>K</kbd></button>
        <button class="del" id="bd">Delete &nbsp;<kbd>D</kbd></button>
        <button id="bu">Clear</button>
        <span style="flex:1"></span>
        <button id="bp">&larr; Prev</button>
        <button id="bn">Next &rarr;</button>
      </div>
      <div class="side">
        <b>Closest figures the paper DOES use</b>
        <ul id="sib"></ul>
      </div>
      <p class="note"><kbd>K</kbd> keep &middot; <kbd>D</kbd> delete &middot;
      <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &middot; decisions save automatically in this browser.</p>
    </div>
  </div>
  <div id="gridv" class="hide"><div class="grid" id="grid"></div></div>
</main>
<script>
const ITEMS = __ITEMS__;
const STAGE_DIR = "__STAGE__";
const KEY = "icml_graphics_triage_v1";
let D = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0, grid = false;
const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(D));

function stats(){
  const k = Object.values(D).filter(v=>v==="keep").length;
  const d = Object.values(D).filter(v=>v==="delete").length;
  const mb = ITEMS.filter(t=>D[t.path]==="delete").reduce((a,b)=>a+b.kb,0)/1024;
  $("bar").style.width = (100*(k+d)/ITEMS.length)+"%";
  $("stat").textContent = `${k+d}/${ITEMS.length} decided · ${k} keep · ${d} delete · ${mb.toFixed(1)} MB staged`;
}
function firstUndecided(){
  const n = ITEMS.findIndex(t=>!D[t.path]);
  return n === -1 ? 0 : n;
}
function render(){
  const t = ITEMS[i];
  $("pos").textContent = `${i+1} / ${ITEMS.length}`;
  $("path").textContent = t.path;
  $("size").textContent = t.kb >= 1024 ? (t.kb/1024).toFixed(2)+" MB" : t.kb+" KB";
  const s = D[t.path] || "undecided";
  $("state").textContent = s.toUpperCase();
  $("state").className = "state " + (s==="keep"?"k":s==="delete"?"d":"u");
  $("imgwrap").innerHTML = t.img
    ? `<img src="${t.img}" alt="${t.name}">`
    : `<div class="pdf"><b>${t.name}</b><br>PDF — no inline preview.<br>Judge from the name and siblings below.</div>`;
  $("sib").innerHTML = t.siblings.map(x=>`<li><code>${x}</code></li>`).join("") || "<li>none</li>";
  $("bp").disabled = i===0; $("bn").disabled = i===ITEMS.length-1;
  stats();
}
function decide(v){
  const t = ITEMS[i];
  if(v) D[t.path]=v; else delete D[t.path];
  save();
  if(v && i < ITEMS.length-1) i++;
  grid ? renderGrid() : render();
}
function renderGrid(){
  $("grid").innerHTML = ITEMS.map((t,n)=>{
    const s = D[t.path]||"";
    const cls = s==="keep"?"k":s==="delete"?"d":"";
    const img = t.img ? `<img src="${t.img}">` : `<div style="height:135px;display:flex;align-items:center;justify-content:center;color:#888;font-size:12px">PDF</div>`;
    return `<div class="tile ${cls}" data-n="${n}">${img}<p>${t.name}</p></div>`;
  }).join("");
  [...document.querySelectorAll(".tile")].forEach(el=>{
    el.onclick = e => {
      const n = +el.dataset.n, p = ITEMS[n].path;
      // click cycles: undecided -> delete -> keep -> undecided
      D[p] = !D[p] ? "delete" : D[p]==="delete" ? "keep" : undefined;
      if(!D[p]) delete D[p];
      save(); renderGrid();
    };
  });
  stats();
}
function dl(name, text){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type:"text/plain"}));
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}
$("exp").onclick = () => {
  const del = ITEMS.filter(t=>D[t.path]==="delete").map(t=>t.path);
  const when = new Date().toISOString().slice(0,19);
  let csv = "path,decision,decided_via\\n";
  ITEMS.forEach(t => { if(D[t.path]) csv += `${t.path},${D[t.path]},offline_html\\n`; });
  dl("graphics_triage_decisions.csv", csv);
  let sh = ["#!/usr/bin/env bash",
    "# Stage triaged unreferenced ICML AIWILD graphics.",
    `# Generated by the offline triage page on ${when}.`,
    "# MOVES files into a staging directory. It deletes nothing.",
    `# Files staged: ${del.length}`,
    "set -euo pipefail",
    'cd "$(git rev-parse --show-toplevel)"',
    `S="${STAGE_DIR}"`, ""];
  del.forEach(p => {
    const d = p.split("/").slice(0,-1).join("/");
    sh.push(`mkdir -p "$S/${d}"`, `mv "overleaf/icml_aiwild_template/${p}" "$S/${d}/"`);
  });
  sh.push("", `echo "Staged ${del.length} files into $S"`);
  dl("stage_unreferenced_graphics.sh", sh.join("\\n")+"\\n");
};
$("mode").onclick = () => {
  grid = !grid;
  $("mode").textContent = grid ? "Single" : "Grid";
  $("single").classList.toggle("hide", grid);
  $("gridv").classList.toggle("hide", !grid);
  grid ? renderGrid() : render();
};
$("bk").onclick=()=>decide("keep"); $("bd").onclick=()=>decide("delete");
$("bu").onclick=()=>decide(null);
$("bp").onclick=()=>{if(i>0){i--;render();}};
$("bn").onclick=()=>{if(i<ITEMS.length-1){i++;render();}};
document.onkeydown = e => {
  if(grid || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  if(k==="k") decide("keep");
  else if(k==="d") decide("delete");
  else if(e.key==="ArrowLeft" && i>0){i--;render();}
  else if(e.key==="ArrowRight" && i<ITEMS.length-1){i++;render();}
  else return;
  e.preventDefault();
};
i = firstUndecided();
render();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-px", type=int, default=1200)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"Manifest missing: {MANIFEST}")
        print("Run: python scripts/build_unreferenced_graphics_manifest.py")
        return 1

    print(f"Embedding thumbnails (max {args.max_px}px, q{args.quality})...")
    items, counts = build_items(args.max_px, args.quality)

    html = (
        HTML.replace("__ITEMS__", json.dumps(items, separators=(",", ":")))
        .replace("__STAGE__", STAGE_DIR)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    n_pdf = sum(1 for it in items if it["img"] is None)
    print(
        f"\nwrote {args.out.relative_to(PROJECT_ROOT)}  "
        f"({args.out.stat().st_size/1024/1024:.1f} MB)"
    )
    print(f"  {len(items)} files ({len(items)-n_pdf} with preview, {n_pdf} PDF placeholders)")
    print(f"  referenced files excluded: {counts['referenced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
