"use strict";

const main = document.getElementById("main");
let review;
let renderSerial = 0;
let toastTimer;
const num = (value) => Number(value).toLocaleString("en-US");
const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
const byId = (items, id) => items.find((item) => item.id === id);
const reportLink = (id, text = "Read the detailed review") =>
  '<a class="button small" href="#report/' +
  encodeURIComponent(id) +
  '">' +
  escapeHtml(text) +
  ' <span aria-hidden="true">↗</span></a>';
const sourceLinks = (ids) =>
  '<div class="source-links">' +
  ids
    .map((id) => {
      const s = byId(review.sources, id);
      return '<a href="#source/' + id + '">' + escapeHtml(s.label) + " ↗</a>";
    })
    .join("") +
  "</div>";
const list = (items) =>
  "<ul>" +
  items.map((item) => "<li>" + escapeHtml(item) + "</li>").join("") +
  "</ul>";
const keys = (values) =>
  '<div class="key-list">' +
  values.map((value) => "<code>" + escapeHtml(value) + "</code>").join("") +
  "</div>";
const heading = (eyebrow, title, subtitle) =>
  '<div class="page-heading"><div class="eyebrow">' +
  escapeHtml(eyebrow) +
  "</div><h1>" +
  escapeHtml(title) +
  '</h1><p class="lead">' +
  escapeHtml(subtitle) +
  "</p></div>";

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error("The review server returned HTTP " + response.status + ".");
  return response.json();
}

function toast(message) {
  const box = document.getElementById("toast");
  box.textContent = message;
  box.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("visible"), 2600);
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    toast("Copied to clipboard");
  } catch (error) {
    toast("Clipboard is unavailable. Select and copy the text directly.");
  }
}

function overview() {
  main.innerHTML =
    '<section class="hero"><div><div class="eyebrow">A proposal for external-user setup</div><h1>From research scripts<br>to repeatable commands.</h1><p class="lead">Keep the game engine. Add one clear launch interface, with explicit settings and checked results.</p><div class="actions"><a class="button primary" href="#experiments">Explore the experiments <span aria-hidden="true">→</span></a><a class="button" href="#patch">See the patch plan</a></div></div><div class="hero-note"><strong>The main conclusion</strong>The experiment logic is largely available. The shared launch, failure, and output paths need fixes before a new user can rely on them.</div></section>' +
    '<div class="metrics"><div class="metric"><span class="metric-value">7</span><span class="metric-label">Experiment families</span></div><div class="metric"><span class="metric-value">7,160</span><span class="metric-label">Runs in the paper breakdown</span></div><div class="metric"><span class="metric-value">13</span><span class="metric-label">Independent agent reviews</span></div><div class="metric"><span class="metric-value">41</span><span class="metric-label">Source files with personal-path matches</span></div></div>' +
    '<div class="section-title"><h2>The proposed path</h2><small>One shared execution contract</small></div><div class="pipeline">' +
    [
      [
        "01",
        "Choose a preset",
        "Select the experiment, model, game, and seed.",
      ],
      [
        "02",
        "Resolve the plan",
        "Save the exact roster, settings, routes, and inputs.",
      ],
      [
        "03",
        "Use the existing engine",
        "Run each negotiation in an isolated worker.",
      ],
      [
        "04",
        "Check the result",
        "Keep exact attempts, outputs, and failure states.",
      ],
    ]
      .map(
        (s) =>
          '<div class="pipeline-step"><div class="step">' +
          s[0] +
          "</div><h3>" +
          s[1] +
          "</h3><p>" +
          s[2] +
          "</p></div>",
      )
      .join("") +
    "</div>" +
    '<div class="section-title"><h2>Two ways to use the code</h2><a class="text-link" href="#experiments">Explore all seven →</a></div><div class="two-col"><section class="panel"><span class="tag green">Small new experiment</span><h3>Try one setting</h3><p class="muted">One game instance and one declared seed. TTC is the four-effort exception.</p><p><code>bargain run homogeneous --model gpt-5-nano --agents 4</code></p><small class="muted">Proposed command. One negotiation can make many paid requests.</small></section><section class="panel"><span class="tag amber">Fixed paper sweep</span><h3>Preserve the selected design</h3><p class="muted">Use a versioned manifest with exact cells, rosters, seeds, and matched inputs.</p><p><code>bargain sweep --preset paper-homogeneous-v1 --plan-only</code></p><small class="muted">The current generator creates 325 runs; the paper retains 300.</small></section></div>' +
    '<div class="section-title"><h2>What needs attention first?</h2><a class="text-link" href="#findings">All findings →</a></div><div class="two-col"><section class="panel"><div class="mini-findings">' +
    review.findings
      .slice(0, 4)
      .map(
        (f) =>
          '<a class="mini-finding" href="#finding/' +
          f.id +
          '"><span>' +
          f.id +
          "</span><div><strong>" +
          escapeHtml(f.title) +
          "</strong><p>" +
          escapeHtml(f.summary) +
          "</p></div></a>",
      )
      .join("") +
    '</div></section><section class="panel soft"><div class="small-label">Scope and confidence</div><h2>A code review, not a release certificate.</h2><p class="muted">The review traced the requested experiment paths and inspected saved inputs. Static syntax checks passed. A clean install and live model execution still need testing.</p>' +
    list([
      "No experiment code was changed during the review.",
      "No model calls, downloads, or Slurm jobs were run.",
      "The UI displays proposals and evidence only.",
    ]) +
    '<div class="actions">' +
    reportLink("validation", "Read the release checks") +
    reportLink("overview", "Full findings") +
    "</div></section></div>";
}

function familyCard(f) {
  return (
    '<a class="family-card" href="#experiment/' +
    f.id +
    '"><div class="topline"><span class="tag">' +
    escapeHtml(f.category) +
    '</span><span aria-hidden="true">↗</span></div><h3>' +
    escapeHtml(f.name) +
    "</h3><p>" +
    escapeHtml(f.concept) +
    '</p><div class="bottomline"><span><span class="run-number">' +
    num(f.paper_runs) +
    '</span> paper runs</span><span class="muted">View design & command</span></div></a>'
  );
}

function experiments() {
  main.innerHTML =
    heading(
      "Seven experiment presets",
      "Choose an experiment type",
      "Multi-agent is the umbrella for homogeneous, heterogeneous, and homogeneous-adversary games. The paper includes n = 2 cells in those designs too.",
    ) +
    '<div class="toolbar"><label class="search"><input id="family-search" aria-label="Search experiments" placeholder="Search models, designs, or setup…"></label><select id="family-category" aria-label="Experiment category"><option value="">All categories</option><option>Two-player</option><option>Multi-agent</option><option>Compute & teams</option></select><span id="family-count" class="filter-count"></span></div><div id="family-grid" class="family-grid"></div>';
  const filter = () => {
    const terms = document
      .getElementById("family-search")
      .value.toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    const category = document.getElementById("family-category").value;
    const rows = review.families.filter(
      (f) =>
        (!category || f.category === category) &&
        terms.every((term) => JSON.stringify(f).toLowerCase().includes(term)),
    );
    document.getElementById("family-grid").innerHTML =
      rows.map(familyCard).join("") ||
      '<div class="empty">No experiment matches this search.</div>';
    document.getElementById("family-count").textContent = rows.length + " of 7";
  };
  document.getElementById("family-search").addEventListener("input", filter);
  document.getElementById("family-category").addEventListener("change", filter);
  filter();
}

function shellQuote(value) {
  return /^[A-Za-z0-9_./:@=-]+$/.test(value)
    ? value
    : "'" + value.replace(/'/g, "'\"'\"'") + "'";
}

function commandPanel(f) {
  const multi = [
    "homogeneous",
    "heterogeneous",
    "homogeneous-adversary",
    "team",
  ].includes(f.id);
  return (
    '<section class="command-panel"><div class="command-head"><h2>Command preview</h2><span class="tag">Proposed only</span></div><div class="command-form">' +
    '<label>Scope<select id="command-scope"><option value="small">Small experiment</option><option value="paper">Paper sweep plan</option></select></label>' +
    '<label id="game-label">Game<select id="command-game"><option value="game1">Game 1 · items</option>' +
    (f.id === "team"
      ? ""
      : '<option value="game2">Game 2 · issues</option><option value="game3">Game 3 · projects</option>') +
    "</select></label>" +
    '<label id="seed-label">Seed<input id="command-seed" type="number" value="42" min="0" step="1"></label>' +
    (multi
      ? '<label id="agents-label">Agents<select id="command-agents"><option>2</option><option selected>4</option><option>6</option><option>8</option><option>10</option></select></label>'
      : "") +
    (f.id === "ttc"
      ? '<label id="effort-label">Model family<select id="command-effort"><option value="gpt5">GPT-5</option><option value="claude">Claude Sonnet 4.6</option><option value="gemini">Gemini 3 Flash</option></select></label>'
      : "") +
    '<label class="wide">Output directory<input id="command-output" value="/path/to/new-negotiation" spellcheck="false" autocomplete="off"></label></div>' +
    '<div class="command-output"><pre id="command-text"></pre></div><p id="command-error" class="command-error" hidden></p><div id="command-meta" class="command-meta"></div>' +
    '<div class="command-foot"><p id="command-explainer"></p><button id="copy-command">Copy proposed command</button></div></section>'
  );
}

function bindCommand(f) {
  let command = "";
  function update() {
    const paper = document.getElementById("command-scope").value === "paper";
    for (const id of [
      "game-label",
      "seed-label",
      "agents-label",
      "effort-label",
    ]) {
      const label = document.getElementById(id);
      if (label) label.hidden = paper;
    }
    const output = document.getElementById("command-output").value;
    const seed = document.getElementById("command-seed").value;
    const invalid =
      !output.startsWith("/") ||
      /[\r\n\0]/.test(output) ||
      (!paper && (!/^\d+$/.test(seed) || !Number.isSafeInteger(Number(seed))));
    const error = document.getElementById("command-error");
    error.hidden = !invalid;
    error.textContent =
      "Use an absolute output path and a nonnegative integer seed.";
    document.getElementById("copy-command").disabled = invalid;
    let args;
    let credentialNames = f.demo_keys;
    let explanation;
    if (paper) {
      args = ["bargain", "sweep", "--preset", f.preset, "--plan-only"];
      credentialNames = [];
      explanation =
        "Plans " +
        num(f.paper_runs) +
        " paper cells without model calls. Fixed manifests still need assembly and historical validation.";
      if (f.id === "team") {
        args.push("--controls", "/path/to/checked-control-bundle");
        explanation +=
          " The matched team plan requires the checked control bundle.";
      }
    } else {
      args = [
        "bargain",
        "run",
        f.id,
        "--game",
        document.getElementById("command-game").value,
      ];
      if (
        ["two-player", "two-player-llama", "homogeneous-adversary"].includes(
          f.id,
        )
      )
        args.push("--adversary", "gpt-4o-mini-2024-07-18");
      if (f.id === "homogeneous") args.push("--model", "gpt-5-nano");
      if (document.getElementById("command-agents"))
        args.push("--agents", document.getElementById("command-agents").value);
      if (f.id === "ttc") {
        const family = document.getElementById("command-effort").value;
        args.push("--family", family);
        credentialNames = [
          "OPENAI_API_KEY",
          ...(family === "claude"
            ? ["ANTHROPIC_API_KEY"]
            : family === "gemini"
              ? ["OPENROUTER_API_KEY"]
              : []),
        ];
      }
      args.push("--seed", seed);
      explanation =
        (f.id === "ttc"
          ? "Four negotiations: one per requested effort level."
          : "One negotiation with the selected seed.") +
        " Replace the example path. This is a UI preview, not an executable launcher.";
      if (f.id === "team") {
        explanation +=
          " This standalone example is not a historical matched rerun.";
        if (document.getElementById("command-agents").value === "2")
          explanation +=
            " With two agents, there is only one Nano agent and no team treatment.";
      }
      if (f.id === "heterogeneous")
        explanation +=
          " Doctor must inspect the actual sampled roster before execution.";
    }
    args.push("--output", output);
    command = args.map(shellQuote).join(" ");
    document.getElementById("command-text").textContent = command;
    document.getElementById("command-explainer").textContent = explanation;
    const meta = document.getElementById("command-meta");
    if (paper)
      meta.textContent =
        "Credentials for planning: none. Required input bundles must still be present.";
    else if (credentialNames === null)
      meta.textContent =
        "Credentials depend on the resolved sampled roster; the full pool uses OpenAI, Anthropic, and OpenRouter.";
    else
      meta.innerHTML =
        "Current routes for this example: " +
        credentialNames.map((key) => "<code>" + key + "</code>").join(" + ");
  }
  document
    .querySelectorAll(".command-form input,.command-form select")
    .forEach((input) => input.addEventListener("input", update));
  document
    .getElementById("copy-command")
    .addEventListener("click", () => copyText(command));
  update();
}

function familyDetail(id) {
  const f = byId(review.families, id);
  if (!f) throw new Error("Unknown experiment family.");
  main.innerHTML =
    '<a class="back-link" href="#experiments">← All experiment types</a>' +
    heading(f.category, f.name, f.concept) +
    '<div class="detail-lead"><section class="panel"><div class="small-label">Current launch path</div><p>' +
    escapeHtml(f.current) +
    "</p>" +
    sourceLinks(f.sources) +
    '</section><section class="panel soft"><div class="small-label">Selected paper collection</div><div class="large-count">' +
    num(f.paper_runs) +
    '</div><div class="grid-description">' +
    escapeHtml(f.grid) +
    "</div></section></div>" +
    commandPanel(f) +
    '<div class="two-col"><section class="panel"><div class="small-label">Proposed change</div><h2>Reuse the existing engine</h2><p class="muted">' +
    escapeHtml(f.proposal) +
    '</p><div class="notice">' +
    escapeHtml(f.blocker) +
    "</div>" +
    reportLink(f.report) +
    '</section><section class="panel"><div class="small-label">Keep these inputs explicit</div><h2>What must stay fixed?</h2>' +
    list(f.preserve) +
    "</section></div>" +
    '<section class="panel spaced-panel"><div class="section-title"><h2>Keys for the full current roster</h2><a class="text-link" href="#setup">Setup details →</a></div>' +
    keys(f.keys) +
    '<p class="muted source-note">These names describe the current inspected routes, not verified account access. The small example can require fewer keys. All selected paper routes use hosted APIs.</p></section>';
  bindCommand(f);
}

function findings(openId) {
  main.innerHTML =
    heading(
      "Verified code-path findings",
      "What blocks a public launch?",
      "These findings explain what the patch must address. They do not claim that every historical run encountered every failure path.",
    ) +
    '<div class="toolbar"><label class="search"><input id="finding-search" aria-label="Search findings" placeholder="Search paths, settings, providers, or resume…"></label><select id="finding-area" aria-label="Finding area"><option value="">All areas</option><option>Correctness</option><option>Portability</option><option>Credentials</option><option>Reproduction</option></select><span id="finding-count" class="filter-count"></span></div><div class="finding-list" id="finding-list"></div>';
  const filter = () => {
    const terms = document
      .getElementById("finding-search")
      .value.toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    const area = document.getElementById("finding-area").value;
    const rows = review.findings.filter(
      (f) =>
        (!area || f.area === area) &&
        terms.every((term) => JSON.stringify(f).toLowerCase().includes(term)),
    );
    document.getElementById("finding-count").textContent =
      rows.length + " findings";
    document.getElementById("finding-list").innerHTML =
      rows
        .map(
          (f) =>
            '<details class="finding" id="finding-' +
            f.id +
            '"' +
            (openId === f.id ? " open" : "") +
            '><summary><span class="finding-id">' +
            f.id +
            "</span><div><h3>" +
            escapeHtml(f.title) +
            "</h3><p>" +
            escapeHtml(f.summary) +
            '</p><div class="finding-tags"><span class="tag">' +
            f.area +
            '</span><span class="tag amber">' +
            f.priority +
            '</span></div></div></summary><div class="finding-body"><div class="two-col"><div><h4>Why it matters</h4><p>' +
            escapeHtml(f.why) +
            "</p></div><div><h4>Proposed fix</h4><p>" +
            escapeHtml(f.fix) +
            "</p></div></div>" +
            sourceLinks(f.sources) +
            '<div class="actions">' +
            reportLink(f.report) +
            "</div></div></details>",
        )
        .join("") || '<div class="empty">No finding matches this search.</div>';
  };
  document.getElementById("finding-search").addEventListener("input", filter);
  document.getElementById("finding-area").addEventListener("change", filter);
  filter();
  if (openId)
    document
      .getElementById("finding-" + openId)
      ?.scrollIntoView({ block: "center" });
}

function patch() {
  main.innerHTML =
    heading(
      "Implementation proposal",
      "One shared path, built in stages",
      "Add a thin launch package around the existing engine. Correct the shared behavior before exposing paid execution.",
    ) +
    '<div class="notice info">No implementation patch has been applied. The code shown in the source viewer is evidence from the inspected working files.</div>' +
    review.patch_stages
      .map(
        (stage) =>
          '<section class="patch-stage"><div class="stage-number">' +
          stage.number +
          '</div><div class="panel"><div class="stage-title"><h2>' +
          escapeHtml(stage.title) +
          '</h2><span class="tag plain">Proposed</span></div><p class="muted">' +
          escapeHtml(stage.purpose) +
          "</p>" +
          list(stage.changes) +
          '<div class="stage-gate"><strong>Acceptance check</strong><br>' +
          escapeHtml(stage.gate) +
          "</div>" +
          sourceLinks(stage.sources) +
          "</div></section>",
      )
      .join("") +
    '<section class="panel"><h2>Where the new interface belongs</h2><p class="muted">Install <code>bargain = strong_models_experiment.cli:main</code> and keep runtime imports on the execution path.</p>' +
    [
      [
        "strong_models_experiment/cli.py",
        "Parse the common command interface.",
      ],
      [
        "strong_models_experiment/launch/schema.py",
        "Validate and materialize scientific settings.",
      ],
      [
        "strong_models_experiment/launch/presets.py",
        "Expand seven experiment families and fixed plans.",
      ],
      [
        "strong_models_experiment/launch/execute.py",
        "Call the existing engine in one worker per negotiation.",
      ],
      [
        "strong_models_experiment/launch/store.py",
        "Own attempt files, status, and explicit resume.",
      ],
    ]
      .map(
        (row) =>
          '<div class="file-row"><code>' +
          escapeHtml(review.root + "/" + row[0]) +
          "</code><p>" +
          row[1] +
          "</p></div>",
      )
      .join("") +
    '<div class="report-tools">' +
    reportLink("implementation", "Read the complete file-level proposal") +
    '<a class="button small" href="/download?id=implementation">Download Markdown ↓</a></div></section>';
}

function setup() {
  main.innerHTML =
    heading(
      "External-user setup",
      "Keys stay private. Routes stay explicit.",
      "A local hosted-API experiment should not need Slurm, a GPU, or a personal directory on this cluster.",
    ) +
    '<div class="two-col"><section class="panel"><h2>A short setup sequence</h2>' +
    [
      [
        "Install the tested package",
        "Use the dependency group for the chosen providers. A clean wheel still needs validation.",
      ],
      [
        "Supply your own keys",
        "Use environment variables or an explicitly selected private environment file. Never put keys in the experiment JSON.",
      ],
      [
        "Check the selected plan",
        "Doctor should use the same models and settings as the intended run. Offline checks establish presence, not API access.",
      ],
      [
        "Run one small experiment",
        "Inspect the printed plan, output location, and terminal outcome before attempting a full sweep.",
      ],
    ]
      .map(
        (row, i) =>
          '<div class="setup-step"><span class="setup-number">0' +
          (i + 1) +
          "</span><div><h3>" +
          row[0] +
          "</h3><p>" +
          row[1] +
          "</p></div></div>",
      )
      .join("") +
    '</section><section class="panel soft"><div class="small-label">Important distinction</div><h2>Provider ≠ model developer</h2><p class="muted">The current TTC Gemini 3 Flash alias uses OpenRouter. It does not require a native Google key.</p>' +
    keys(["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]) +
    '<p class="muted">These three providers cover the current selected paper routes. A small run often needs only one or two.</p><div class="notice">The current template says <code>GEMINI_API_KEY</code>, but optional native Google execution reads <code>GOOGLE_API_KEY</code>.</div>' +
    reportLink("credentials", "Read the credential audit") +
    "</section></div>" +
    '<div class="section-title"><h2>Current route requirements</h2><small>Not a live authentication check</small></div><div class="table-wrap"><table><thead><tr><th>Selected experiment</th><th>Required variables</th></tr></thead><tbody>' +
    [
      ["Nano + native OpenAI adversary model", ["OPENAI_API_KEY"]],
      [
        "Nano or Llama + a mixed OpenAI/OpenRouter pair",
        ["OPENAI_API_KEY", "OPENROUTER_API_KEY"],
      ],
      [
        "Full two-player, Llama, homogeneous, or heterogeneous rosters",
        ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"],
      ],
      [
        "Full homogeneous-adversary roster",
        ["OPENAI_API_KEY", "OPENROUTER_API_KEY"],
      ],
      ["TTC · GPT-5", ["OPENAI_API_KEY"]],
      ["TTC · Claude Sonnet 4.6", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]],
      ["TTC · Gemini 3 Flash", ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]],
      ["Binding team with direct OpenAI override", ["OPENAI_API_KEY"]],
    ]
      .map(
        (row) =>
          "<tr><td>" +
          row[0] +
          "</td><td>" +
          row[1].map((value) => "<code>" + value + "</code>").join("<br>") +
          "</td></tr>",
      )
      .join("") +
    "</tbody></table></div>" +
    '<div class="two-col"><section class="panel"><span class="tag green">Default proposal</span><h2>Local API execution</h2>' +
    list([
      "Use explicit direct requests on a networked machine.",
      "Load credentials through one shared loader.",
      "Keep model identity and effort controls in the resolved plan.",
      "Write results only to the selected experiment directory.",
    ]) +
    '</section><section class="panel"><span class="tag amber">Separate release work</span><h2>Restricted cluster execution</h2>' +
    list([
      "Use an explicit site profile and private shared queue.",
      "Secure and redact request archives before public use.",
      "Preserve provider, endpoint, and account routing.",
      "Reject unsupported native-provider routes instead of silently changing providers.",
    ]) +
    sourceLinks(["queue", "monitor", "transport"]) +
    "</section></div>" +
    '<div class="section-title"><h2>What was actually validated?</h2></div><div class="check-list">' +
    review.validation
      .map(
        (check) =>
          '<section class="check"><span class="tag ' +
          (check.state === "Not tested" ? "amber" : "green") +
          '">' +
          escapeHtml(check.state) +
          "</span><h3>" +
          escapeHtml(check.title) +
          "</h3><p>" +
          escapeHtml(check.text) +
          "</p></section>",
      )
      .join("") +
    "</div>";
}

function reports() {
  main.innerHTML =
    heading(
      "Evidence library",
      "Read the source reviews",
      "Two summary documents and thirteen agent reports. Source links open approved code excerpts; other local paths can be copied.",
    ) +
    '<div class="toolbar"><label class="search"><input id="report-search" aria-label="Search reports" placeholder="Find a report…"></label><span id="report-count" class="filter-count"></span></div><div id="report-grid" class="three-col"></div>';
  const filter = () => {
    const term = document.getElementById("report-search").value.toLowerCase();
    const items = review.reports.filter((report) =>
      (report.title + " " + report.group + " " + report.filename)
        .toLowerCase()
        .includes(term),
    );
    document.getElementById("report-count").textContent =
      items.length + " of 15 reports";
    document.getElementById("report-grid").innerHTML =
      items
        .map(
          (report) =>
            '<a class="panel report-card" href="#report/' +
            report.id +
            '"><span class="tag">' +
            report.group +
            "</span><h3>" +
            escapeHtml(report.title) +
            " ↗</h3><p>" +
            escapeHtml(report.path) +
            "</p></a>",
        )
        .join("") || '<div class="empty">No report matches this search.</div>';
  };
  document.getElementById("report-search").addEventListener("input", filter);
  filter();
}

function inlineMarkdown(raw) {
  const slots = [];
  const slot = (html) => {
    slots.push(html);
    return "\uE000" + (slots.length - 1) + "\uE001";
  };
  let value = raw.replace(/`([^`]+)`/g, (_, code) =>
    slot("<code>" + escapeHtml(code) + "</code>"),
  );
  value = value.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, destination) => {
    const report = review.reports.find(
      (item) => destination.replace(/:\d+$/, "") === item.path,
    );
    const source = review.sources.find(
      (item) => destination === item.path + ":" + item.line,
    );
    if (report)
      return slot(
        '<a href="#report/' + report.id + '">' + escapeHtml(label) + "</a>",
      );
    if (source)
      return slot(
        '<a href="#source/' + source.id + '">' + escapeHtml(label) + "</a>",
      );
    if (destination.startsWith("/") && !destination.startsWith("//"))
      return slot(
        '<button class="local-ref" data-copy="' +
          escapeHtml(destination) +
          '" title="Copy source path: ' +
          escapeHtml(destination) +
          '">' +
          escapeHtml(label) +
          "</button>",
      );
    if (/^https:\/\//i.test(destination))
      return slot(
        '<a href="' +
          escapeHtml(destination) +
          '" target="_blank" rel="noopener noreferrer">' +
          escapeHtml(label) +
          "</a>",
      );
    return slot("<span>" + escapeHtml(label) + "</span>");
  });
  value = escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return value.replace(
    /\uE000(\d+)\uE001/g,
    (_, index) => slots[Number(index)],
  );
}

function renderMarkdown(text) {
  const lines = text.replace(/\r/g, "").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (/^```/.test(line)) {
      const code = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
      i++;
      blocks.push(
        "<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>",
      );
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.+)$/);
    if (h) {
      blocks.push(
        "<h" +
          h[1].length +
          ">" +
          inlineMarkdown(h[2]) +
          "</h" +
          h[1].length +
          ">",
      );
      i++;
      continue;
    }
    if (
      line.trim().startsWith("|") &&
      i + 1 < lines.length &&
      /^\s*\|[\s:|\-]+\|\s*$/.test(lines[i + 1])
    ) {
      const cells = (row) =>
        row
          .trim()
          .replace(/^\||\|$/g, "")
          .split("|")
          .map((cell) => inlineMarkdown(cell.trim()));
      const headers = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|"))
        rows.push(cells(lines[i++]));
      blocks.push(
        '<div class="table-wrap"><table><thead><tr>' +
          headers.map((c) => "<th>" + c + "</th>").join("") +
          "</tr></thead><tbody>" +
          rows
            .map(
              (row) =>
                "<tr>" +
                row.map((c) => "<td>" + c + "</td>").join("") +
                "</tr>",
            )
            .join("") +
          "</tbody></table></div>",
      );
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      function parseList(indent) {
        let html = "<ul>";
        while (i < lines.length) {
          const match = lines[i].match(/^(\s*)[-*]\s+(.+)$/);
          if (!match || match[1].length !== indent) break;
          html += "<li>" + inlineMarkdown(match[2]);
          i++;
          while (i < lines.length) {
            const nested = lines[i].match(/^(\s*)[-*]\s+(.+)$/);
            if (nested && nested[1].length > indent)
              html += parseList(nested[1].length);
            else break;
          }
          html += "</li>";
        }
        return html + "</ul>";
      }
      blocks.push(parseList(line.match(/^\s*/)[0].length));
      continue;
    }
    if (/^\d+\.\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i]))
        items.push(lines[i++].replace(/^\d+\.\s+/, ""));
      blocks.push(
        "<ol>" +
          items
            .map((item) => "<li>" + inlineMarkdown(item) + "</li>")
            .join("") +
          "</ol>",
      );
      continue;
    }
    const paragraph = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,6}\s|```|\s*[-*]\s|\d+\.\s|\|)/.test(lines[i])
    )
      paragraph.push(lines[i++]);
    blocks.push("<p>" + inlineMarkdown(paragraph.join(" ")) + "</p>");
  }
  return blocks.join("\n");
}

async function documentPage(id, serial) {
  const doc = await getJson("/api/report?id=" + encodeURIComponent(id));
  if (serial !== renderSerial) return;
  main.innerHTML =
    '<a class="back-link" href="#reports">← Source reports</a>' +
    heading(
      doc.group,
      doc.title,
      "Saved source review. Technical claims and current commands were established by inspection unless a report states otherwise.",
    ) +
    '<div class="report-tools"><a class="button" href="/download?id=' +
    encodeURIComponent(id) +
    '">Download Markdown ↓</a><button data-copy="' +
    escapeHtml(doc.path) +
    '">Copy report path</button><span class="tag plain">Read-only</span></div><div class="report-path">' +
    escapeHtml(doc.path) +
    '</div><article class="panel report-body">' +
    renderMarkdown(doc.text) +
    "</article>";
}

async function sourcePage(id, serial) {
  const source = await getJson("/api/source?id=" + encodeURIComponent(id));
  if (serial !== renderSerial) return;
  const content = source.text
    .split("\n")
    .map(
      (line, i) =>
        '<span class="source-line' +
        (source.start + i === source.line ? " highlight" : "") +
        '"><span class="line-number">' +
        (source.start + i) +
        "</span>" +
        escapeHtml(line) +
        "</span>",
    )
    .join("");
  main.innerHTML =
    '<a class="back-link" href="#findings">← Findings</a>' +
    heading(
      "Source evidence",
      source.label,
      "This is an excerpt from the working source loaded when the UI started. It is not a proposed replacement.",
    ) +
    '<div class="report-path">' +
    escapeHtml(source.path) +
    ":" +
    source.line +
    '</div><div class="report-tools"><button data-copy="' +
    escapeHtml(source.path + ":" + source.line) +
    '">Copy source reference</button><span class="tag plain">Lines ' +
    source.start +
    "–" +
    source.end +
    '</span></div><pre class="source-code"><code>' +
    content +
    '</code></pre><p class="source-note">Only explicitly indexed source excerpts are available. This viewer does not expose arbitrary files, credentials, or experiment logs.</p>';
}

async function route() {
  const serial = ++renderSerial;
  const raw = location.hash.slice(1) || "overview";
  const [page, id] = raw.split("/");
  const parent =
    {
      experiment: "experiments",
      finding: "findings",
      report: "reports",
      source: "reports",
    }[page] || page;
  document.querySelectorAll("nav a").forEach((link) => {
    const active = link.dataset.page === parent;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.getElementById("breadcrumb").textContent =
    {
      overview: "Launch review",
      experiments: "Experiments",
      findings: "Findings",
      patch: "Patch plan",
      setup: "Keys & setup",
      reports: "Source reports",
    }[parent] || "Launch review";
  main.innerHTML = '<div class="loading">Loading…</div>';
  window.scrollTo(0, 0);
  try {
    if (page === "overview") overview();
    else if (page === "experiments") experiments();
    else if (page === "experiment") familyDetail(id);
    else if (page === "findings") findings();
    else if (page === "finding") findings(id);
    else if (page === "patch") patch();
    else if (page === "setup") setup();
    else if (page === "reports") reports();
    else if (page === "report") await documentPage(id, serial);
    else if (page === "source") await sourcePage(id, serial);
    else throw new Error("This page does not exist.");
  } catch (error) {
    if (serial === renderSerial)
      main.innerHTML =
        '<div class="notice error"><h2>Could not load this page</h2><p>' +
        escapeHtml(error.message) +
        '</p><a href="#overview">Return to the overview</a></div>';
  }
}

main.addEventListener("click", (event) => {
  const copy = event.target.closest("[data-copy]");
  if (copy) copyText(copy.dataset.copy);
});

document.querySelector(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  main.focus();
  main.scrollIntoView();
});

async function refreshStatus() {
  try {
    const status = await getJson("/api/status");
    const warning = document.getElementById("stale-warning");
    warning.hidden = status.changed_inputs_since_load.length === 0;
    warning.textContent =
      "Source files or reports changed after this viewer loaded. The displayed snapshot is unchanged; restart the viewer to load the newer files.";
  } catch (error) {
    const warning = document.getElementById("stale-warning");
    warning.hidden = false;
    warning.textContent =
      "The review server is unreachable. Check the SSH tunnel and refresh this page.";
  }
}

(async () => {
  try {
    review = await getJson("/api/review");
    window.addEventListener("hashchange", route);
    await route();
    await refreshStatus();
    setInterval(refreshStatus, 30000);
  } catch (error) {
    main.innerHTML =
      '<div class="notice error"><h2>Review unavailable</h2><p>' +
      escapeHtml(error.message) +
      "</p><p>Check the server and SSH tunnel, then refresh.</p></div>";
  }
})();
