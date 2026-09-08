"use strict";
const app = document.getElementById("app");
const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (n) => Number(n).toLocaleString();
const labels = {
  needed: "Needed",
  "protected-other-git-worktree": "Other Git worktree",
  "conditional-retirement-candidate": "Conditional candidate",
  "unresolved-do-not-delete": "Unresolved · keep",
};
const badge = (s) =>
  `<span class="badge ${esc(s)}">${esc(labels[s] || s)}</span>`;
const link = (id) =>
  `<a class="badge" href="#audit/${encodeURIComponent(id)}">${esc(id)}</a>`;
const links = (ids) =>
  `<div class="auditlinks">${String(ids).split(";").filter(Boolean).map(link).join("")}</div>`;
const download = (name, label) =>
  `<a class="button" href="/download?name=${encodeURIComponent(name)}">${esc(label)}</a>`;
const heading = (tag, title, desc) =>
  `<div class="eyebrow">${esc(tag)}</div><h1>${esc(title)}</h1><p class="muted">${esc(desc)}</p>`;
const table = (headers, rows) =>
  `<div class="tablewrap"><table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.length ? rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") : `<tr><td colspan="${headers.length}">No matching records.</td></tr>`}</tbody></table></div>`;
const renderText = (text) =>
  esc(text)
    .split("\n")
    .map((l) =>
      l.startsWith("### ")
        ? `<h3>${l.slice(4)}</h3>`
        : l.startsWith("## ")
          ? `<h2>${l.slice(3)}</h2>`
          : l.startsWith("# ")
            ? `<h1>${l.slice(2)}</h1>`
            : /^\s*- /.test(l)
              ? `<div class="bullet">${l.replace(/^(\s*)- /, "$1• ")}</div>`
              : l,
    )
    .join("\n")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>");
async function api(path, params = {}) {
  const r = await fetch("/api/" + path + "?" + new URLSearchParams(params));
  if (!r.ok) throw Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}
let epoch = 0;
function filters() {
  return `<input id="search" aria-label="Search paths" placeholder="Search full path, filename or folder…"><select id="status" aria-label="File status"><option value="">All statuses</option>${Object.entries(
    labels,
  )
    .map(([k, v]) => `<option value="${k}">${v}</option>`)
    .join("")}</select>`;
}

async function overview(token) {
  const s = await api("status");
  if (token !== epoch) return;
  const c = s.status_counts;
  app.innerHTML =
    heading(
      "Audit overview",
      "What can we safely keep or remove?",
      "45 agent audits across three waves. Explore each result, its dependencies and the open questions.",
    ) +
    `<div class="notice"><strong>This is not a deletion certificate.</strong> ${fmt(c["unresolved-do-not-delete"])} inventoried code files remain unresolved. Source candidates still require a retirement decision.</div>${s.changed_inputs_since_load.length ? `<div class="notice error">Audit inputs changed after server startup: ${esc(s.changed_inputs_since_load.join(", "))}. Restart the viewer to review the new records.</div>` : ""}<div class="cards"><div class="card"><b>${fmt(c.needed)}</b><span>Code files with retention evidence</span></div><div class="card"><b>${fmt(c["unresolved-do-not-delete"])}</b><span>Unresolved code files</span></div><div class="card"><b>${s.source_candidates}</b><span>Conditional source candidates</span></div><div class="card"><b>${fmt(s.needed_files)}</b><span>Concrete files on the keep list</span></div></div><div class="grid"><section class="panel"><h2>Code inventory</h2><p class="muted">${fmt(s.code_files)} code/configuration filenames in the original snapshot.</p><div class="bar">${Object.entries(
      c,
    )
      .map(
        ([k, n]) =>
          `<div class="${esc(k)}" style="width:${(100 * n) / s.code_files}%" title="${esc(labels[k])}: ${n}"></div>`,
      )
      .join("")}</div><div class="legend">${Object.entries(c)
      .map(([k, n]) => `<div>${badge(k)}<strong>${fmt(n)}</strong></div>`)
      .join(
        "",
      )}</div></section><section class="panel"><h2>Coverage and limits</h2><p><strong>${s.figures} figures · ${s.tables} tables · ${s.agent_audits} audits</strong></p><p>${s.directory_paths} directory paths also have explicit subset-selection rules. Their entire contents are not certified as needed.</p><p>Runtime, result reproduction and historical provenance are separate reasons to retain a file.</p><p><a href="#review">Review the new consistency checks →</a></p><p><a href="#audits">Browse all experiment assignments →</a></p><p class="reviewstamp">Loaded ${esc(s.loaded_at)}<br>Read-only server · consistency checks ${s.review_passed ? "passed" : "need attention"}</p></section></div><section class="panel"><h2>Files that look old can still be required</h2><p>Current results use dated review scripts, recovery archives and old analysis modules. The audit recovered 27 launch/configuration files inside an earlier deletion archive.</p>${link("root_recovered_archives")} ${link("root_import_closure")}</section><div class="toolbar">${download("report.md", "Download full report")}${download("needed_files.csv", "Download keep list")}${download("code_status.csv", "Download code status")}</div>`;
}
async function audits(token) {
  const rows = await api("audits");
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      "Result provenance",
      "Experiment audits",
      "One report per assigned result family or supporting method. Open an audit to see its findings and exact file evidence.",
    ) +
    `<div class="toolbar"><input id="auditsearch" aria-label="Search audits" placeholder="Search game, model, method or audit ID…"></div><div id="results"></div>`;
  const draw = () => {
    const q = document.getElementById("auditsearch").value.toLowerCase();
    document.getElementById("results").innerHTML = table(
      ["Audit", "Result / method", "Needed entries", "Open questions"],
      rows
        .filter((r) => JSON.stringify(r).toLowerCase().includes(q))
        .map((r) => [
          link(r.id),
          esc(r.result),
          fmt(r.needed_entries),
          fmt(r.open_questions),
        ]),
    );
  };
  document.getElementById("auditsearch").oninput = draw;
  draw();
}
async function audit(id, token) {
  const d = await api("audit", { id });
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      id,
      d.result,
      "Positive dependency evidence and unresolved questions from this assignment.",
    ) +
    `<section class="panel"><h2>Finding</h2><p>${esc(d.summary)}</p></section><section class="panel"><h2>Open questions</h2>${d.unresolved.length ? `<ul>${d.unresolved.map((x) => `<li>${esc(typeof x === "string" ? x : JSON.stringify(x))}</li>`).join("")}</ul>` : "<p>No questions recorded by this audit; this is not a global completeness claim.</p>"}</section><div class="toolbar">${download(d.report_name, "Download this report")}<a class="button" href="#files?audit=${encodeURIComponent(id)}">Explore listed files</a></div><details><summary>Dependency entries (${d.needed.length})</summary>${table(
      ["Path", "Role", "Reason / evidence"],
      d.needed
        .slice(0, 150)
        .map((r) => [
          `<span class="path">${esc(r.path)}</span>`,
          esc(r.role),
          esc(r.reason) + "<br><small>" + esc(r.evidence) + "</small>",
        ]),
    )}${d.needed.length > 150 ? "<p>Showing 150 entries here. Use the file explorer or full dependency download for all entries.</p>" : ""}</details><section class="panel report">${renderText(d.report)}</section>`;
}
async function files(query, token) {
  const params = new URLSearchParams(query),
    audit = params.get("audit") || "";
  app.innerHTML =
    heading(
      "File-level evidence",
      "File explorer",
      "Search full paths. Click a file to see the reports and reasons that support its status.",
    ) +
    `<div class="toolbar"><select id="dataset" aria-label="Inventory"><option value="code">Code inventory (1,204)</option><option value="needed" ${audit ? "selected" : ""}>Concrete keep list (12,078)</option></select>${filters()}</div>${audit ? `<p>Filtered to ${link(audit)} · <a href="#files">Clear audit filter</a></p>` : ""}<div id="results"></div><div class="pager"><button id="prev">Previous</button><span id="range"></span><button id="next">Next</button></div>`;
  let offset = 0,
    total = 0,
    request = 0;
  const draw = async () => {
    const mine = ++request;
    const dataset = document.getElementById("dataset").value;
    document.getElementById("status").disabled = dataset === "needed";
    const data = await api("files", {
      dataset,
      q: document.getElementById("search").value,
      status: dataset === "code" ? document.getElementById("status").value : "",
      audit,
      offset,
      limit: 100,
    });
    if (token !== epoch || mine !== request) return;
    total = data.total;
    document.getElementById("results").innerHTML = table(
      ["Full path", dataset === "code" ? "Status" : "Retention role", "Audits"],
      data.rows.map((r) => [
        `<span class="path"><a href="#file/${encodeURIComponent(r.path)}">${esc(r.path)}</a></span>`,
        dataset === "code" ? badge(r.status) : esc(r.roles),
        links(r.audit_ids),
      ]),
    );
    document.getElementById("range").textContent =
      `${total ? offset + 1 : 0}–${Math.min(offset + 100, total)} of ${fmt(total)}`;
    document.getElementById("prev").disabled = offset === 0;
    document.getElementById("next").disabled = offset + 100 >= total;
  };
  const change = () => {
    offset = 0;
    draw().catch(showError);
  };
  document.getElementById("search").oninput = change;
  document.getElementById("dataset").onchange = change;
  document.getElementById("status").onchange = change;
  document.getElementById("prev").onclick = () => {
    offset = Math.max(0, offset - 100);
    draw().catch(showError);
  };
  document.getElementById("next").onclick = () => {
    offset += 100;
    draw().catch(showError);
  };
  await draw();
}
async function file(path, token) {
  const d = await api("file", { path });
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      "Decision evidence",
      "File detail",
      "This viewer shows audit evidence, not file contents.",
    ) +
    `<div class="codebox path">${esc(path)}</div><p>${d.code ? badge(d.code.status) : badge(d.keep ? "needed" : "conditional-retirement-candidate")} · Path ${d.exists_now ? "exists now" : "is missing now"}</p>${d.candidate ? `<section class="panel"><h2>Candidate rationale</h2><p>${esc(d.candidate.reason)}</p><p><strong>No deletion is authorized.</strong></p>${link(d.candidate.audit)}</section>` : ""}${
      !d.evidence.length
        ? '<div class="notice">No positive file-level dependency evidence is recorded here. This does not prove the file is unused.</div>'
        : table(
            ["Audit", "Role", "Reason", "Source evidence"],
            d.evidence.map((r) => [
              link(r.audit),
              esc(r.role),
              esc(r.reason),
              `<span class="path">${esc(r.evidence)}</span>`,
            ]),
          )
    }`;
}
async function candidates(token) {
  const rows = await api("candidates");
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      "Conditional retirement",
      "Cleanup candidates",
      "Seven generic planning helpers, two older plotting frontends, and 25 generated test-bytecode files.",
    ) +
    `<div class="notice"><strong>There are no delete or approval controls.</strong> Source candidates require confirmation that their standalone workflows are no longer needed. Old plot scripts are a separate decision from runtime cleanup.</div><div class="toolbar"><select id="kind" aria-label="Candidate type"><option value="candidate-only">Source candidates (9)</option><option value="safe-generated">Generated bytecode (25)</option><option value="">All candidates</option></select>${download("cleanup_candidates.json", "Download evidence")}</div><div id="results"></div>`;
  const draw = () => {
    const kind = document.getElementById("kind").value;
    document.getElementById("results").innerHTML = table(
      ["Exact path", "Reason", "Audit"],
      rows
        .filter((r) => !kind || r.confidence === kind)
        .map((r) => [
          `<span class="path"><a href="#file/${encodeURIComponent(r.path)}">${esc(r.path)}</a></span>`,
          esc(r.reason),
          link(r.audit),
        ]),
    );
  };
  document.getElementById("kind").onchange = draw;
  draw();
}
async function coverage(token) {
  const rows = await api("coverage");
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      "Paper → result → files",
      "Paper coverage",
      "All 30 figures and 10 numbered tables, linked to their audit assignments and source locations.",
    ) +
    table(
      ["Paper item", "Label / source", "Audits", "Graphic inputs"],
      rows.map((r) => [
        `${esc(r.kind)} ${esc(r.source_order_number)}`,
        `<strong>${esc(r.labels)}</strong><br><span class="path">${esc(r.source_path)}:${esc(r.line)}</span>`,
        links(r.audit_ids),
        `<span class="path">${esc(r.graphic_inputs || "Table embedded in LaTeX")}</span>`,
      ]),
    ) +
    `<div class="toolbar">${download("paper_coverage.csv", "Download coverage map")}</div>`;
}
async function review(token) {
  const [d, selections] = await Promise.all([api("review"), api("selections")]);
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      "Independent record checks",
      "Review of the audit",
      "Rechecked from the detailed reports and actual filesystem paths when this server started.",
    ) +
    `<div class="notice"><strong>${d.checks.filter((c) => c.passed).length}/${d.checks.length} record checks passed.</strong> Passing these checks does not certify scientific validity, complete dependency coverage, or safe deletion.</div><p class="reviewstamp">Checked ${esc(d.checked_at)}</p><div class="checks">${d.checks.map((c) => `<div class="check ${c.passed ? "" : "fail"}"><strong>${c.passed ? "PASS" : "NEEDS REVIEW"} · ${esc(c.name)}</strong><small>${esc(c.scope)}</small>${c.issues.length ? `<pre class="path">${esc(JSON.stringify(c.issues, null, 2))}</pre>` : ""}</div>`).join("")}</div><section class="panel"><h2>What remains unproven</h2><ul>${d.limits.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></section><details><summary>Directory-selection evidence (${selections.length} records)</summary>${table(
      ["Directory", "Audit", "Selection rule"],
      selections.map((s) => [
        `<span class="path">${esc(s.path)}</span>`,
        link(s.audit),
        esc(s.reason),
      ]),
    )}</details>`;
}
async function report(token) {
  const d = await api("report");
  if (token !== epoch) return;
  app.innerHTML =
    heading(
      "Final audit deliverable",
      "Full report",
      "The original report is preserved. Fresh record checks are available on the Review checks page.",
    ) +
    `<div class="toolbar">${download("report.md", "Download Markdown report")}${download("audit_index.csv", "Download audit index")}</div><section class="panel report">${renderText(d.text)}</section>`;
}
function showError(error) {
  app.innerHTML = `<div class="notice error"><h2>Could not load audit records</h2><p>${esc(error.message)}</p><p>No data was substituted. Check the server and reload this page.</p></div>`;
}
async function route() {
  const token = ++epoch;
  const hash = location.hash.slice(1) || "overview";
  const [path, query = ""] = hash.split("?");
  document
    .querySelectorAll("nav a")
    .forEach((a) =>
      a.classList.toggle(
        "active",
        a.hash ===
          "#" +
            (path.startsWith("audit/")
              ? "audits"
              : path.startsWith("file/")
                ? "files"
                : path),
      ),
    );
  app.innerHTML = '<div class="loading">Loading evidence…</div>';
  try {
    if (path === "overview") await overview(token);
    else if (path === "audits") await audits(token);
    else if (path.startsWith("audit/"))
      await audit(decodeURIComponent(path.slice(6)), token);
    else if (path === "files") await files(query, token);
    else if (path.startsWith("file/"))
      await file(decodeURIComponent(path.slice(5)), token);
    else if (path === "candidates") await candidates(token);
    else if (path === "coverage") await coverage(token);
    else if (path === "review") await review(token);
    else if (path === "report") await report(token);
    else throw Error("Unknown view");
  } catch (e) {
    if (token === epoch) showError(e);
  }
}
window.addEventListener("hashchange", route);
route();
