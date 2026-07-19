#!/usr/bin/env python3
"""Build a self-contained browser UI for reviewing strategic tags."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAG_DIR = PROJECT_ROOT / "analysis/strategic_qualitative_tags_20260628"
CODEBOOK_CSV = TAG_DIR / "new_strategy_tag_codebook.csv"
EVIDENCE_CSV = TAG_DIR / "new_strategy_tag_evidence.csv"
OUT_HTML = TAG_DIR / "tag_hot_or_not_review_game.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_data() -> list[dict]:
    codebook = read_csv(CODEBOOK_CSV)
    evidence = read_csv(EVIDENCE_CSV)
    by_tag: dict[str, list[dict]] = {}
    for row in evidence:
        by_tag.setdefault(row["tag_code"], []).append(
            {
                "config_id": row.get("config_id", ""),
                "experiment_family": row.get("experiment_family", ""),
                "game_label": row.get("game_label", ""),
                "n_agents": row.get("n_agents", ""),
                "result_path": row.get("result_path", ""),
                "quote": row.get("quote", ""),
            }
        )

    tags = []
    for row in codebook:
        tags.append(
            {
                "tag_code": row["tag_code"],
                "tag_title": row["tag_title"],
                "category": row["category"],
                "description": row["description"],
                "paper_value": row["paper_value"],
                "games": row.get("games") or "",
                "min_agents": row.get("min_agents") or "",
                "structural": str(row.get("structural", "")).lower() == "true",
                "patterns": row.get("patterns", ""),
                "count": int(float(row.get("count") or 0)),
                "share": float(row.get("share") or 0),
                "examples": by_tag.get(row["tag_code"], []),
            }
        )
    return tags


def build_html(tags: list[dict]) -> str:
    data_json = json.dumps(tags, ensure_ascii=False)
    escaped_data = data_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Strategic Tag Hot-or-Not Review</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5c6672;
      --line: #d7dce2;
      --blue: #2457c5;
      --green: #247a4b;
      --red: #b03a3a;
      --gold: #8a6500;
      --teal: #176b73;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      --radius: 8px;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}

    button, input, textarea, select {{
      font: inherit;
    }}

    button {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      padding: 8px 11px;
      cursor: pointer;
      min-height: 38px;
    }}

    button:hover {{ border-color: #9aa8b6; }}
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {{
      outline: 3px solid rgba(36, 87, 197, 0.25);
      outline-offset: 2px;
    }}

    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
    }}

    .sidebar {{
      border-right: 1px solid var(--line);
      background: #eef1f5;
      padding: 16px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }}

    .brand {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 14px;
    }}

    .brand h1 {{
      font-size: 18px;
      margin: 0;
      line-height: 1.2;
    }}

    .progressBox {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 12px;
      box-shadow: var(--shadow);
      margin-bottom: 12px;
    }}

    .meter {{
      height: 10px;
      background: #dce2ea;
      border-radius: 999px;
      overflow: hidden;
      margin: 9px 0 8px;
    }}

    .meter > div {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--blue), var(--teal));
      transition: width 140ms ease;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      font-size: 12px;
    }}

    .stat {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px;
      text-align: center;
    }}

    .stat strong {{ display: block; font-size: 16px; }}

    .controlBlock {{
      display: grid;
      gap: 8px;
      margin-bottom: 12px;
    }}

    .search, .select {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 7px;
      padding: 9px 10px;
      min-height: 38px;
    }}

    .tagList {{
      display: grid;
      gap: 5px;
    }}

    .tagRow {{
      width: 100%;
      display: grid;
      grid-template-columns: 24px 1fr auto;
      align-items: center;
      gap: 7px;
      padding: 8px;
      border: 1px solid transparent;
      background: transparent;
      text-align: left;
    }}

    .tagRow.active {{
      background: var(--panel);
      border-color: #9fb1ca;
      box-shadow: var(--shadow);
    }}

    .tagRow .idx {{
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }}

    .tagRow .name {{
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      min-width: 0;
      font-size: 13px;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--muted);
      white-space: nowrap;
    }}

    .pill.hot {{ color: var(--green); border-color: rgba(36, 122, 75, 0.35); background: #edf8f1; }}
    .pill.not {{ color: var(--red); border-color: rgba(176, 58, 58, 0.35); background: #fff1f1; }}
    .pill.maybe {{ color: var(--gold); border-color: rgba(138, 101, 0, 0.35); background: #fff8df; }}

    .main {{
      padding: 18px;
      min-width: 0;
    }}

    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}

    .topbarLeft {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .topbarRight {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .primary {{
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }}

    .danger {{
      border-color: rgba(176, 58, 58, 0.35);
      color: var(--red);
      background: #fff7f7;
    }}

    .quiet {{
      color: var(--muted);
    }}

    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}

    .tagCard {{
      padding: 18px;
    }}

    .kicker {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}

    .tagTitle {{
      font-size: 28px;
      margin: 0 0 4px;
      line-height: 1.15;
    }}

    .code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #eef3f8;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 6px;
    }}

    .definitionGrid {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr);
      gap: 14px;
      margin-top: 14px;
    }}

    .field {{
      border-top: 1px solid var(--line);
      padding-top: 11px;
      margin-top: 11px;
    }}

    .field h2 {{
      margin: 0 0 5px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
    }}

    .field p {{
      margin: 0;
    }}

    .decisionPanel {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 12px;
      align-self: start;
    }}

    .decisionButtons {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 10px;
    }}

    .decisionButtons button {{
      min-height: 50px;
      font-weight: 700;
    }}

    .decisionButtons button.selected.hot {{ background: #e5f6eb; border-color: var(--green); color: var(--green); }}
    .decisionButtons button.selected.not {{ background: #ffe8e8; border-color: var(--red); color: var(--red); }}
    .decisionButtons button.selected.maybe {{ background: #fff2bd; border-color: var(--gold); color: var(--gold); }}

    textarea {{
      width: 100%;
      min-height: 100px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      background: white;
    }}

    .examples {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}

    .example {{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfcfe;
      padding: 12px;
    }}

    .exampleHead {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 8px;
    }}

    .exampleMeta {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      min-width: 0;
    }}

    .quote {{
      margin: 0;
      color: #26313d;
    }}

    .pathLine {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }}

    .pathText {{
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      color: var(--muted);
      font-size: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}

    .review {{
      display: none;
    }}

    .review.active {{
      display: block;
    }}

    .reviewTable {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }}

    .reviewTable th, .reviewTable td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}

    .reviewTable th {{
      background: #eef3f8;
      color: #2b3642;
    }}

    .reviewTable tr:last-child td {{ border-bottom: 0; }}

    .hidden {{ display: none !important; }}

    .fileInput {{
      display: none;
    }}

    .smallNote {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}

    @media (max-width: 900px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .tagList {{
        max-height: 220px;
        overflow: auto;
      }}
      .definitionGrid {{
        grid-template-columns: 1fr;
      }}
      .topbar {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .topbarRight {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>Tag Hot-or-Not</h1>
        <span class="pill" id="totalPill">50 tags</span>
      </div>

      <div class="progressBox">
        <div><strong id="progressLabel">0 / 50 rated</strong></div>
        <div class="meter"><div id="progressMeter"></div></div>
        <div class="stats">
          <div class="stat"><strong id="hotCount">0</strong>Hot</div>
          <div class="stat"><strong id="maybeCount">0</strong>Maybe</div>
          <div class="stat"><strong id="notCount">0</strong>Not</div>
        </div>
        <div class="smallNote" id="saveStatus">Autosave ready.</div>
      </div>

      <div class="controlBlock">
        <input class="search" id="searchInput" placeholder="Search tags..." />
        <select class="select" id="categoryFilter"></select>
        <select class="select" id="decisionFilter">
          <option value="all">All decisions</option>
          <option value="unrated">Unrated only</option>
          <option value="hot">Hot only</option>
          <option value="maybe">Maybe only</option>
          <option value="not">Not only</option>
        </select>
      </div>

      <div class="tagList" id="tagList"></div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div class="topbarLeft">
          <button id="prevBtn">Back</button>
          <button id="nextBtn">Next</button>
          <button id="reviewBtn">Review</button>
        </div>
        <div class="topbarRight">
          <button id="saveBtn">Save Progress JSON</button>
          <button id="importBtn">Import Progress</button>
          <input id="importInput" class="fileInput" type="file" accept="application/json,.json" />
          <button id="downloadBtn" class="primary">Download Final JSON</button>
          <button id="resetBtn" class="danger">Reset</button>
        </div>
      </div>

      <section id="gameView" class="card tagCard"></section>
      <section id="reviewView" class="review"></section>
    </main>
  </div>

  <script id="tag-data" type="application/json">{escaped_data}</script>
  <script>
    const TAGS = JSON.parse(document.getElementById('tag-data').textContent);
    const STORAGE_KEY = 'strategic-tag-hot-or-not-v1';
    const RUN_META = {{
      app: 'strategic-tag-hot-or-not',
      version: 1,
      source: 'analysis/strategic_qualitative_tags_20260628',
      generatedAt: {json.dumps(__import__("datetime").datetime.now().isoformat(timespec="seconds"))}
    }};

    let state = loadState();
    let index = clamp(state.currentIndex || 0, 0, TAGS.length - 1);
    let reviewMode = false;

    const els = {{
      tagList: document.getElementById('tagList'),
      gameView: document.getElementById('gameView'),
      reviewView: document.getElementById('reviewView'),
      progressLabel: document.getElementById('progressLabel'),
      progressMeter: document.getElementById('progressMeter'),
      hotCount: document.getElementById('hotCount'),
      maybeCount: document.getElementById('maybeCount'),
      notCount: document.getElementById('notCount'),
      saveStatus: document.getElementById('saveStatus'),
      searchInput: document.getElementById('searchInput'),
      categoryFilter: document.getElementById('categoryFilter'),
      decisionFilter: document.getElementById('decisionFilter'),
      prevBtn: document.getElementById('prevBtn'),
      nextBtn: document.getElementById('nextBtn'),
      reviewBtn: document.getElementById('reviewBtn'),
      saveBtn: document.getElementById('saveBtn'),
      importBtn: document.getElementById('importBtn'),
      importInput: document.getElementById('importInput'),
      downloadBtn: document.getElementById('downloadBtn'),
      resetBtn: document.getElementById('resetBtn')
    }};

    init();

    function init() {{
      buildCategoryFilter();
      bindEvents();
      render();
    }}

    function loadState() {{
      const blank = {{
        currentIndex: 0,
        responses: Object.fromEntries(TAGS.map(tag => [tag.tag_code, {{ decision: '', notes: '' }}]))
      }};
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return blank;
        const parsed = JSON.parse(raw);
        return mergeState(blank, parsed);
      }} catch {{
        return blank;
      }}
    }}

    function mergeState(base, incoming) {{
      const next = structuredClone(base);
      if (incoming && Number.isFinite(Number(incoming.currentIndex))) {{
        next.currentIndex = Number(incoming.currentIndex);
      }}
      const responses = incoming.responses || incoming.decisions || {{}};
      for (const tag of TAGS) {{
        const row = responses[tag.tag_code] || {{}};
        const decision = ['hot', 'maybe', 'not'].includes(row.decision) ? row.decision : '';
        next.responses[tag.tag_code] = {{
          decision,
          notes: typeof row.notes === 'string' ? row.notes : ''
        }};
      }}
      return next;
    }}

    function persist(message = 'Autosaved.') {{
      state.currentIndex = index;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      els.saveStatus.textContent = message;
    }}

    function buildCategoryFilter() {{
      const categories = [...new Set(TAGS.map(tag => tag.category))].sort();
      els.categoryFilter.innerHTML = '<option value="all">All categories</option>' +
        categories.map(cat => `<option value="${{escapeAttr(cat)}}">${{escapeHtml(cat)}}</option>`).join('');
    }}

    function bindEvents() {{
      els.prevBtn.addEventListener('click', () => move(-1));
      els.nextBtn.addEventListener('click', () => move(1));
      els.reviewBtn.addEventListener('click', () => {{
        reviewMode = !reviewMode;
        render();
      }});
      els.searchInput.addEventListener('input', renderSidebar);
      els.categoryFilter.addEventListener('change', renderSidebar);
      els.decisionFilter.addEventListener('change', renderSidebar);
      els.saveBtn.addEventListener('click', () => downloadJson('strategic_tag_review_progress.json'));
      els.downloadBtn.addEventListener('click', () => downloadJson('strategic_tag_review_final.json', true));
      els.importBtn.addEventListener('click', () => els.importInput.click());
      els.importInput.addEventListener('change', importProgress);
      els.resetBtn.addEventListener('click', resetProgress);
      window.addEventListener('keydown', handleKeydown);
    }}

    function handleKeydown(event) {{
      if (event.target.matches('textarea, input, select')) return;
      if (event.key === 'ArrowLeft') move(-1);
      if (event.key === 'ArrowRight') move(1);
      if (event.key === '1') setDecision('hot');
      if (event.key === '2') setDecision('maybe');
      if (event.key === '3') setDecision('not');
    }}

    function render() {{
      renderStats();
      renderSidebar();
      if (reviewMode) {{
        els.gameView.classList.add('hidden');
        els.reviewView.classList.add('active');
        els.reviewBtn.textContent = 'Return to Game';
        renderReview();
      }} else {{
        els.gameView.classList.remove('hidden');
        els.reviewView.classList.remove('active');
        els.reviewBtn.textContent = 'Review';
        renderCard();
      }}
    }}

    function renderStats() {{
      const counts = {{ hot: 0, maybe: 0, not: 0, rated: 0 }};
      for (const tag of TAGS) {{
        const d = state.responses[tag.tag_code]?.decision || '';
        if (d) {{
          counts[d] += 1;
          counts.rated += 1;
        }}
      }}
      els.progressLabel.textContent = `${{counts.rated}} / ${{TAGS.length}} rated`;
      els.progressMeter.style.width = `${{Math.round((counts.rated / TAGS.length) * 100)}}%`;
      els.hotCount.textContent = counts.hot;
      els.maybeCount.textContent = counts.maybe;
      els.notCount.textContent = counts.not;
    }}

    function renderSidebar() {{
      const rows = filteredTags();
      els.tagList.innerHTML = rows.map((tag) => {{
        const actualIndex = TAGS.findIndex(item => item.tag_code === tag.tag_code);
        const decision = state.responses[tag.tag_code]?.decision || '';
        const decisionLabel = decision ? decision : 'open';
        const active = actualIndex === index && !reviewMode ? ' active' : '';
        return `
          <button class="tagRow${{active}}" data-index="${{actualIndex}}">
            <span class="idx">${{actualIndex + 1}}</span>
            <span class="name" title="${{escapeAttr(tag.tag_title)}}">${{escapeHtml(tag.tag_title)}}</span>
            <span class="pill ${{decision}}">${{escapeHtml(decisionLabel)}}</span>
          </button>
        `;
      }}).join('');
      for (const btn of els.tagList.querySelectorAll('.tagRow')) {{
        btn.addEventListener('click', () => {{
          index = Number(btn.dataset.index);
          reviewMode = false;
          persist('Autosaved.');
          render();
        }});
      }}
    }}

    function filteredTags() {{
      const q = els.searchInput.value.trim().toLowerCase();
      const cat = els.categoryFilter.value;
      const decision = els.decisionFilter.value;
      return TAGS.filter(tag => {{
        const response = state.responses[tag.tag_code]?.decision || '';
        if (cat !== 'all' && tag.category !== cat) return false;
        if (decision === 'unrated' && response) return false;
        if (['hot', 'maybe', 'not'].includes(decision) && response !== decision) return false;
        if (!q) return true;
        const haystack = [tag.tag_code, tag.tag_title, tag.category, tag.description, tag.paper_value].join(' ').toLowerCase();
        return haystack.includes(q);
      }});
    }}

    function renderCard() {{
      const tag = TAGS[index];
      const response = state.responses[tag.tag_code] || {{ decision: '', notes: '' }};
      const examples = tag.examples || [];
      els.gameView.innerHTML = `
        <div class="kicker">
          <span class="pill">${{index + 1}} / ${{TAGS.length}}</span>
          <span class="pill">${{escapeHtml(tag.category)}}</span>
          <span class="pill">${{tag.count}} / 2730 (${{formatPct(tag.share)}})</span>
          ${{tag.structural ? '<span class="pill maybe">structural</span>' : ''}}
        </div>
        <h1 class="tagTitle">${{escapeHtml(tag.tag_title)}}</h1>
        <div class="code">${{escapeHtml(tag.tag_code)}}</div>

        <div class="definitionGrid">
          <div>
            <div class="field">
              <h2>Definition</h2>
              <p>${{escapeHtml(tag.description)}}</p>
            </div>
            <div class="field">
              <h2>Why It Matters</h2>
              <p>${{escapeHtml(tag.paper_value)}}</p>
            </div>
            <div class="field">
              <h2>Scope / Patterns</h2>
              <p>
                ${{tag.games ? `Games: <span class="code">${{escapeHtml(tag.games)}}</span>` : 'Games: all'}}
                ${{tag.min_agents ? ` &nbsp; Min agents: <span class="code">${{escapeHtml(String(tag.min_agents))}}</span>` : ''}}
              </p>
              <p class="smallNote">${{escapeHtml(tag.patterns || 'structural check')}}</p>
            </div>
          </div>

          <div class="decisionPanel">
            <div class="decisionButtons">
              <button class="${{response.decision === 'hot' ? 'selected hot' : ''}}" data-decision="hot">Hot</button>
              <button class="${{response.decision === 'maybe' ? 'selected maybe' : ''}}" data-decision="maybe">Maybe</button>
              <button class="${{response.decision === 'not' ? 'selected not' : ''}}" data-decision="not">Not</button>
            </div>
            <textarea id="notesBox" placeholder="Notes for yourself...">${{escapeHtml(response.notes || '')}}</textarea>
            <div class="smallNote">Keyboard: 1 = Hot, 2 = Maybe, 3 = Not, arrows = previous/next.</div>
          </div>
        </div>

        <div class="field">
          <h2>Examples</h2>
          <div class="examples">
            ${{examples.length ? examples.map(renderExample).join('') : '<div class="example">No snippets available for this tag.</div>'}}
          </div>
        </div>
      `;
      for (const btn of els.gameView.querySelectorAll('[data-decision]')) {{
        btn.addEventListener('click', () => setDecision(btn.dataset.decision));
      }}
      const notes = document.getElementById('notesBox');
      notes.addEventListener('input', () => {{
        state.responses[tag.tag_code].notes = notes.value;
        persist('Autosaved notes.');
        renderStats();
        renderSidebar();
      }});
    }}

    function renderExample(ex, i) {{
      const path = ex.result_path || '';
      return `
        <article class="example">
          <div class="exampleHead">
            <div class="exampleMeta">
              <span class="pill">config ${{escapeHtml(String(ex.config_id || ''))}}</span>
              <span class="pill">${{escapeHtml(ex.experiment_family || '')}}</span>
              <span class="pill">${{escapeHtml(ex.game_label || '')}}</span>
              <span class="pill">n=${{escapeHtml(String(ex.n_agents || ''))}}</span>
            </div>
            <button data-copy="${{escapeAttr(path)}}">Copy path</button>
          </div>
          <p class="quote">${{escapeHtml(ex.quote || '')}}</p>
          <div class="pathLine">
            <div class="pathText" title="${{escapeAttr(path)}}">${{escapeHtml(path)}}</div>
            <button data-copy="${{escapeAttr(ex.quote || '')}}">Copy quote</button>
          </div>
        </article>
      `;
    }}

    function renderReview() {{
      const rows = TAGS.map((tag, i) => {{
        const response = state.responses[tag.tag_code] || {{ decision: '', notes: '' }};
        const decision = response.decision || 'unrated';
        return `
          <tr>
            <td>${{i + 1}}</td>
            <td><button data-jump="${{i}}">${{escapeHtml(tag.tag_title)}}</button><div class="code">${{escapeHtml(tag.tag_code)}}</div></td>
            <td><span class="pill ${{response.decision}}">${{escapeHtml(decision)}}</span></td>
            <td>${{escapeHtml(tag.category)}}</td>
            <td>${{escapeHtml(response.notes || '')}}</td>
          </tr>
        `;
      }}).join('');
      els.reviewView.innerHTML = `
        <div class="card tagCard">
          <div class="topbar">
            <div>
              <h1 class="tagTitle">Review Responses</h1>
              <p class="quiet">Click a tag name to jump back and edit it. Download final JSON when ready.</p>
            </div>
            <button class="primary" id="reviewDownload">Download Final JSON</button>
          </div>
          <table class="reviewTable">
            <thead><tr><th>#</th><th>Tag</th><th>Decision</th><th>Category</th><th>Notes</th></tr></thead>
            <tbody>${{rows}}</tbody>
          </table>
        </div>
      `;
      document.getElementById('reviewDownload').addEventListener('click', () => downloadJson('strategic_tag_review_final.json', true));
      for (const btn of els.reviewView.querySelectorAll('[data-jump]')) {{
        btn.addEventListener('click', () => {{
          index = Number(btn.dataset.jump);
          reviewMode = false;
          persist('Autosaved.');
          render();
        }});
      }}
    }}

    function setDecision(decision) {{
      const tag = TAGS[index];
      state.responses[tag.tag_code].decision = decision;
      persist(`Marked ${{tag.tag_title}} as ${{decision}}.`);
      render();
    }}

    function move(delta) {{
      if (reviewMode) {{
        reviewMode = false;
      }}
      index = clamp(index + delta, 0, TAGS.length - 1);
      persist('Autosaved.');
      render();
    }}

    function exportPayload(finalExport = false) {{
      const responses = TAGS.map((tag, i) => {{
        const response = state.responses[tag.tag_code] || {{ decision: '', notes: '' }};
        return {{
          index: i + 1,
          tag_code: tag.tag_code,
          tag_title: tag.tag_title,
          category: tag.category,
          decision: response.decision || '',
          notes: response.notes || '',
          source_count: tag.count,
          source_share: tag.share
        }};
      }});
      const summary = responses.reduce((acc, row) => {{
        const key = row.decision || 'unrated';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }}, {{}});
      return {{
        ...RUN_META,
        exportedAt: new Date().toISOString(),
        finalExport,
        currentIndex: index,
        summary,
        responsesByCode: Object.fromEntries(responses.map(row => [row.tag_code, {{ decision: row.decision, notes: row.notes }}])),
        responses
      }};
    }}

    function downloadJson(filename, finalExport = false) {{
      const payload = exportPayload(finalExport);
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      persist(finalExport ? 'Final JSON downloaded.' : 'Progress JSON downloaded.');
    }}

    function importProgress(event) {{
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {{
        try {{
          const incoming = JSON.parse(String(reader.result || '{{}}'));
          const blank = {{ currentIndex: 0, responses: Object.fromEntries(TAGS.map(tag => [tag.tag_code, {{ decision: '', notes: '' }}])) }};
          const source = incoming.responsesByCode ? {{ currentIndex: incoming.currentIndex, responses: incoming.responsesByCode }} : incoming;
          state = mergeState(blank, source);
          index = clamp(state.currentIndex || 0, 0, TAGS.length - 1);
          persist('Imported progress and autosaved.');
          reviewMode = false;
          render();
        }} catch (err) {{
          alert('Could not import that JSON file.');
        }}
      }};
      reader.readAsText(file);
      event.target.value = '';
    }}

    function resetProgress() {{
      if (!confirm('Reset all decisions and notes?')) return;
      localStorage.removeItem(STORAGE_KEY);
      state = loadState();
      index = 0;
      reviewMode = false;
      persist('Reset complete.');
      render();
    }}

    document.body.addEventListener('click', async (event) => {{
      const btn = event.target.closest('[data-copy]');
      if (!btn) return;
      try {{
        await navigator.clipboard.writeText(btn.dataset.copy || '');
        els.saveStatus.textContent = 'Copied to clipboard.';
      }} catch {{
        els.saveStatus.textContent = 'Clipboard copy failed.';
      }}
    }});

    function formatPct(value) {{
      return `${{(Number(value || 0) * 100).toFixed(1)}}%`;
    }}

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function escapeAttr(value) {{
      return escapeHtml(value).replace(/`/g, '&#96;');
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    tags = build_data()
    OUT_HTML.write_text(build_html(tags), encoding="utf-8")
    print(f"wrote {OUT_HTML} with {len(tags)} tags")


if __name__ == "__main__":
    main()
