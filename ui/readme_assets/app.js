"use strict";

const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (char) =>
  ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[char]);
// This guide uses headings, paragraphs, lists, inline code, and fenced code.
// Escape source text before inserting markup. Raw HTML is displayed, never run.
function inline(text) {
  return text.split(/(`[^`]+`)/g).map((part) => part.startsWith("`") && part.endsWith("`")
    ? "<code>" + escapeHtml(part.slice(1, -1)) + "</code>"
    : escapeHtml(part).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")).join("");
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const sections = [];
  let section = {id:"overview", title:"Overview", html:[], source:[]};
  sections.push(section);
  for (let i = 0; i < lines.length;) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      const language = line.slice(3).trim();
      const code = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      if (i < lines.length) i++;
      section.source.push(code.join("\n"));
      section.html.push('<div class="code-block"><div class="code-toolbar"><span>' + escapeHtml(language || "text") + '</span><button class="copy" aria-label="Copy code block">Copy</button></div><pre><code>' + escapeHtml(code.join("\n")) + '</code></pre></div>');
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length, title = heading[2];
      if (level === 2) {
        const slug = title.toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-|-$/g,"");
        section = {id:"section-" + sections.length + "-" + slug, title, html:[], source:[]};
        sections.push(section);
      }
      section.html.push("<h" + level + ">" + (level === 2 ? '<a href="#' + section.id + '">' + inline(title) + '</a>' : inline(title)) + "</h" + level + ">");
      section.source.push(title);
      i++;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        const item = lines[i++].replace(/^[-*]\s+/, "");
        items.push("<li>" + inline(item) + "</li>");
        section.source.push(item);
      }
      section.html.push("<ul>" + items.join("") + "</ul>");
      continue;
    }
    const paragraph = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|[-*]\s)/.test(lines[i])) paragraph.push(lines[i++]);
    section.source.push(paragraph.join("\n"));
    section.html.push("<p>" + inline(paragraph.join("\n")) + "</p>");
  }
  return sections;
}

let sections = [], observer, toastTimer;
function toast(message) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("visible"), 2400);
}
function filter() {
  const query = document.getElementById("search").value.trim().toLowerCase();
  let visible = 0;
  for (const section of sections) {
    const matches = (section.title + "\n" + section.source.join("\n")).toLowerCase().includes(query);
    document.getElementById(section.id).hidden = !matches;
    document.querySelector('nav a[href="#' + section.id + '"]').hidden = !matches;
    if (matches) visible++;
  }
  document.getElementById("empty").hidden = visible !== 0;
}
async function loadDocument() {
  const button = document.getElementById("refresh"), doc = document.getElementById("document"), error = document.getElementById("error");
  button.disabled = true;
  doc.setAttribute("aria-busy", "true");
  error.hidden = true;
  try {
    const response = await fetch("/api/document", {cache:"no-store"});
    if (!response.ok) throw new Error("Could not load the guide (HTTP " + response.status + ").");
    const data = await response.json();
    sections = renderMarkdown(data.markdown);
    doc.innerHTML = sections.map((section) => '<section id="' + section.id + '">' + section.html.join("\n") + '</section>').join("\n");
    document.getElementById("nav").innerHTML = sections.map((section) => '<a href="#' + section.id + '">' + escapeHtml(section.title) + '</a>').join("");
    document.getElementById("source").textContent = data.path;
    document.getElementById("updated").textContent = "File modified " + new Date(data.modified_at * 1000).toLocaleString() + ". Refresh to read the latest version.";
    if (observer) observer.disconnect();
    observer = new IntersectionObserver((entries) => {
      const visible = entries.find((entry) => entry.isIntersecting);
      if (!visible) return;
      document.querySelectorAll("nav a").forEach((link) => {
        if (link.hash === "#" + visible.target.id) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }, {rootMargin:"-85px 0px -65% 0px"});
    document.querySelectorAll("#document section").forEach((section) => observer.observe(section));
    filter();
    const anchor = document.getElementById(location.hash.slice(1));
    if (anchor && !anchor.hidden) anchor.scrollIntoView();
  } catch (failure) {
    error.textContent = failure.message + " Any content still visible is from the previous successful load.";
    error.hidden = false;
    if (!sections.length) doc.textContent = "";
  } finally {
    button.disabled = false;
    doc.setAttribute("aria-busy", "false");
  }
}
document.getElementById("search").addEventListener("input", filter);
document.getElementById("refresh").addEventListener("click", loadDocument);
document.getElementById("document").addEventListener("click", async (event) => {
  const button = event.target.closest(".copy");
  if (!button) return;
  const code = button.closest(".code-block").querySelector("pre code").textContent;
  try { await navigator.clipboard.writeText(code); toast("Copied to clipboard"); }
  catch { toast("Clipboard unavailable. Select and copy the command directly."); }
});
loadDocument();
