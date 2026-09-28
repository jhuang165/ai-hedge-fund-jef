/* aihf web — the page. Plain DOM, no build step. Talks to /api/* and polls jobs. */
"use strict";

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, signed = true) => v == null ? "-" : `${signed && v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
const num = (v, d = 2) => v == null ? "-" : Number(v).toFixed(d);
const money = (v) => v == null ? "-" : (Math.abs(v) >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : Math.abs(v) >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`);
const signed = (v) => v == null ? "-" : `${v > 0 ? "+" : ""}${Number(v).toFixed(2)}`;
const cls = (v) => v == null ? "" : v > 0.05 ? "pos" : v < -0.05 ? "neg" : "flat";
const pill = (text, extra = "") => `<span class="pill ${esc(text)} ${extra}">${esc(text)}</span>`;

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json" },
    cache: "no-store",  // job polls and freshly created mandates must never hit the HTTP cache
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail ?? data));
  }
  return data;
}

function showError(el, err) {
  el.hidden = !err;
  el.textContent = err ? (err.message || String(err)) : "";
}

function progressUI(el, p, done) {
  el.hidden = false;
  const fill = $(".fill", el), label = $(".progress-label", el);
  if (done) { el.hidden = true; return; }
  if (p && p.n) {
    fill.classList.remove("indeterminate");
    fill.style.width = `${Math.round(100 * (p.i ?? 0) / p.n)}%`;
    label.textContent = `${p.i ?? 0}/${p.n}  ${p.label ?? ""}`;
  } else {
    fill.classList.add("indeterminate");
    label.textContent = (p && p.label) || "starting…";
  }
}

// Poll a job until it settles; onTick gets every snapshot.
function pollJob(id, onTick) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      let job;
      try { job = await api(`/api/jobs/${id}`); } catch (e) { return reject(e); }
      onTick(job);
      if (job.status === "done") return resolve(job);
      if (job.status === "failed") return reject(new Error(job.error || "job failed"));
      setTimeout(tick, 1200);
    };
    tick();
  });
}

const remember = (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { } };
const recall = (k) => { try { return localStorage.getItem(k); } catch { return null; } };

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const state = { status: null, mandates: [], strategies: [], models: [], reports: {} };

function route() {
  const view = (location.hash || "#research").slice(1);
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  $$(".topbar nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  if (view === "reports") loadReports();
  if (view === "settings") renderSettings();
  if (view === "fund") loadMandates();
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  state.status = await api("/api/status");
  $("#version").textContent = state.status.version;
  $("#env-path").textContent = state.status.mandates_dir.replace(/mandates$/, ".env");
  for (const sel of ["#research-model", "#fund-model"]) {
    const el = $(sel);
    el.innerHTML = state.status.llm_models
      .filter((m) => m.supported)
      .map((m) => `<option value="${esc(m.model)}" ${m.model === state.status.default_model ? "selected" : ""}>${esc(m.display_name)}${m.key_set ? "" : " (no key)"}</option>`)
      .join("");
  }
  const today = state.status.today;
  $("#research-date").value = today;
  $("#fund-date").value = today;
  const start = new Date(today); start.setDate(start.getDate() - 7 * 78);
  $("#fund-start").value = start.toISOString().slice(0, 10);

  const missing = state.status.keys.filter((k) => !k.set && ["FINANCIAL_DATASETS_API_KEY", "TAVILY_API_KEY"].includes(k.env_var));
  $("#key-warning").innerHTML = missing.length
    ? `missing ${missing.map((k) => `<code>${esc(k.env_var)}</code>`).join(", ")} — <a href="#settings">set keys</a>`
    : "";

  state.models = await api("/api/models");
  state.strategies = await api("/api/strategies");
  route();
  resumeJob("research");
  resumeJob("fund");
}

// ---------------------------------------------------------------------------
// Research
// ---------------------------------------------------------------------------

$("#research-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const targets = $("#research-targets").value.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
  const errEl = $("#research-error");
  showError(errEl, null);
  if (!targets.length) return showError(errEl, new Error("give at least one ticker"));
  const btn = $("#research-run"); btn.disabled = true;
  try {
    const job = await api("/api/research", { method: "POST", body: { targets, as_of: $("#research-date").value, model: $("#research-model").value } });
    remember("aihf.job.research", job.id);
    await followResearch(job.id);
  } catch (e) { showError(errEl, e); }
  finally { btn.disabled = false; }
});

async function followResearch(id) {
  const prog = $("#research-progress"), errEl = $("#research-error");
  $("#research-report").hidden = true;
  try {
    const job = await pollJob(id, (j) => {
      progressUI(prog, j.progress, j.status === "done" || j.status === "failed");
      if (j.progress && j.progress.done) renderResearchRows(j.progress.done, null, j.progress.failures || []);
    });
    const r = job.result;
    state.reports.current = Object.fromEntries(r.reports.map((rep) => [rep.ticker, rep]));
    renderResearchRows(r.reports.map(rowOf), r.reports, r.failures || []);
    $("#research-results-sub").textContent = `as of ${r.as_of} · ${r.reports.length} name${r.reports.length === 1 ? "" : "s"}`;
    if (r.reports.length === 1) showReport(r.reports[0], "#research-report");
  } catch (e) { progressUI(prog, null, true); showError(errEl, e); }
}

function rowOf(rep) {
  const desk = Object.fromEntries((rep.desk || []).map((s) => [s.model_name, s.metadata && s.metadata.abstained ? null : s.value]));
  return { ...rep, headline: (rep.thesis || "").split("\n")[0], desk, n_sources: (rep.sources || []).length };
}

function renderResearchRows(rows, full, failures) {
  const el = $("#research-table");
  if (!rows.length && !failures.length) { el.className = "empty"; el.textContent = "Nothing yet."; return; }
  el.className = "";
  const p = (r) => r.position ? `${r.position.side} ${Math.abs(r.position.shares)} @ ${num(r.position.cost_basis)}` : "-";
  const pnl = (r) => r.position && r.position.unrealized_pnl_pct != null ? `<span class="${cls(r.position.unrealized_pnl_pct)}">${pct(r.position.unrealized_pnl_pct)}</span>` : "-";
  const d = (r, k) => r.desk && r.desk[k] != null ? `<span class="${cls(r.desk[k])}">${signed(r.desk[k])}</span>` : `<span class="muted">-</span>`;
  el.innerHTML = `<table><thead><tr>
      <th>ticker</th><th>signal</th><th class="num">conf</th><th>action</th><th>position</th><th class="num">P&amp;L</th>
      <th class="num" title="12-1 momentum">mom</th><th class="num" title="quality-value">q-val</th><th class="num" title="insider flow">insiders</th><th>thesis</th>
    </tr></thead><tbody>${rows.map((r) => `
      <tr class="clickable" data-ticker="${esc(r.ticker)}">
        <td><b>${esc(r.ticker)}</b></td><td>${pill(r.signal)}</td><td class="num">${Math.round(r.confidence)}</td>
        <td>${pill(r.action)}</td><td>${esc(p(r))}</td><td class="num">${pnl(r)}</td>
        <td class="num">${d(r, "momentum")}</td><td class="num">${d(r, "quality-value")}</td><td class="num">${d(r, "insider-flow")}</td>
        <td class="ellipsis" title="${esc(r.headline)}">${esc(r.headline)}</td>
      </tr>`).join("")}
    ${failures.map((f) => `<tr><td><b>${esc(f.ticker)}</b></td><td colspan="9" class="neg">${esc(f.error)}</td></tr>`).join("")}
    </tbody></table>`;
  if (full) {
    $$("tr.clickable", el).forEach((tr) => tr.addEventListener("click", () => {
      $$("tr.selected", el).forEach((x) => x.classList.remove("selected"));
      tr.classList.add("selected");
      showReport(state.reports.current[tr.dataset.ticker], "#research-report");
    }));
  }
}

// ---------------------------------------------------------------------------
// Report detail (shared by Research and Reports)
// ---------------------------------------------------------------------------

function showReport(rep, target) {
  const el = $(target);
  el.hidden = false;
  el.innerHTML = renderReport(rep);
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderReport(r) {
  const t = r.technicals, p = r.position;
  const cite = (idx) => (idx || []).map((i) => `<a class="cite" href="#src-${i}" title="${esc((r.sources[i] || {}).title)}">[${i}]</a>`).join("");
  const claims = (list) => list.length ? `<ul class="claims">${list.map((c) => `<li>${esc(c.text)}${cite(c.source_indices)}</li>`).join("")}</ul>` : `<p class="muted">none</p>`;
  const gauge = (v) => {
    const w = Math.min(50, Math.abs(v) * 50);
    return `<div class="gauge"><div class="g ${v >= 0 ? "pos" : "neg"}" style="width:${w}%"></div></div><div class="val ${cls(v)}">${signed(v)}</div>`;
  };
  return `
    <div class="head">
      <span class="ticker">${esc(r.ticker)}</span>
      ${pill(r.signal)} <span class="muted">${Math.round(r.confidence)}% confidence</span>
      <span class="muted">as of ${esc(r.as_of)} · ${esc(r.model)}</span>
      ${t ? `<span class="muted">last close <b>${num(t.last_close)}</b> (${esc(t.last_bar_date)})</span>` : ""}
    </div>
    <div class="action">${pill(r.action, "big")}<p>${esc(r.action_rationale)}</p></div>
    ${p ? `<div class="grid">
      <div class="tile"><div class="k">position</div><div class="v">${esc(p.side)} ${Math.abs(p.shares)} sh</div></div>
      <div class="tile"><div class="k">avg cost</div><div class="v">${num(p.cost_basis)}</div></div>
      <div class="tile"><div class="k">market value</div><div class="v">${money(p.market_value)}</div></div>
      <div class="tile"><div class="k">unrealized</div><div class="v ${cls(p.unrealized_pnl_pct)}">${pct(p.unrealized_pnl_pct)} (${p.unrealized_pnl == null ? "-" : money(p.unrealized_pnl)})</div></div>
    </div>` : ""}
    <h3>Thesis</h3>
    <div class="thesis">${(r.thesis || "").split(/\n+/).map((para) => `<p>${esc(para)}</p>`).join("")}</div>
    <div class="two">
      <div><h3>Catalysts</h3>${claims(r.catalysts || [])}</div>
      <div><h3>Risks</h3>${claims(r.risks || [])}</div>
    </div>
    <h3>Desk readout</h3>
    <div class="desk">${(r.desk || []).map((s) => s.metadata && s.metadata.abstained
      ? `<div class="name">${esc(s.model_name)}</div><div class="muted">abstained</div><div></div><div class="reason">${esc(s.metadata.abstain_reason || "")}</div>`
      : `<div class="name">${esc(s.model_name)}</div>${gauge(s.value)}<div class="reason">${esc(s.reasoning || "no view")}</div>`).join("")}</div>
    ${t ? `<h3>Price action</h3><div class="grid">
      ${tile("1m", pct(t.ret_1m), cls(t.ret_1m))}${tile("3m", pct(t.ret_3m), cls(t.ret_3m))}${tile("6m", pct(t.ret_6m), cls(t.ret_6m))}${tile("12m", pct(t.ret_12m), cls(t.ret_12m))}
      ${tile("12-1 momentum", pct(t.momentum_12_1), cls(t.momentum_12_1))}${tile("vol (63d, ann.)", pct(t.vol_63d_ann, false))}${tile("max drawdown 1y", pct(t.max_drawdown_1y, false))}
      ${tile("from 52w high", pct(t.pct_from_52w_high))}${tile("above 52w low", pct(t.pct_from_52w_low))}${tile("RSI(14)", num(t.rsi_14, 0))}
      ${tile("SMA50 / SMA200", `${num(t.sma_50)} / ${num(t.sma_200)}`)}${tile("avg $ volume 20d", money(t.avg_dollar_volume_20d))}
    </div>` : ""}
    <h3>Sources</h3>
    <ul class="sources">${(r.sources || []).map((s, i) => `<li id="src-${i}"><span class="n">[${i}]</span><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title || s.url)}</a> <span class="muted">${esc(s.published_date || "")}</span></li>`).join("")}</ul>
    ${(r.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join("")}`;
}

const tile = (k, v, c = "") => `<div class="tile"><div class="k">${esc(k)}</div><div class="v ${c}">${v}</div></div>`;

// ---------------------------------------------------------------------------
// Reports (saved)
// ---------------------------------------------------------------------------

async function loadReports() {
  const rows = await api("/api/reports");
  const el = $("#reports-table");
  $("#reports-sub").textContent = rows.length ? `${rows.length} saved` : "";
  if (!rows.length) { el.className = "empty"; el.textContent = "No saved reports yet — every research run is saved here."; $("#scorecard").hidden = true; return; }
  loadScorecard();
  el.className = "";
  el.innerHTML = `<table><thead><tr><th>as of</th><th>ticker</th><th>signal</th><th class="num">conf</th><th>action</th><th>position</th><th>thesis</th><th>model</th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="clickable" data-id="${esc(r.id)}">
      <td>${esc(r.as_of)}</td><td><b>${esc(r.ticker)}</b></td><td>${pill(r.signal)}</td><td class="num">${Math.round(r.confidence)}</td><td>${pill(r.action)}</td>
      <td>${r.position ? esc(`${r.position.side} ${Math.abs(r.position.shares)} @ ${num(r.position.cost_basis)}`) : "-"}</td>
      <td class="ellipsis" title="${esc(r.headline)}">${esc(r.headline)}</td><td class="muted">${esc(r.model)}</td></tr>`).join("")}</tbody></table>`;
  $$("tr.clickable", el).forEach((tr) => tr.addEventListener("click", async () => {
    $$("tr.selected", el).forEach((x) => x.classList.remove("selected"));
    tr.classList.add("selected");
    showReport(await api(`/api/reports/${tr.dataset.id}`), "#reports-report");
  }));
}

async function loadScorecard() {
  const box = $("#scorecard");
  box.hidden = false;
  box.innerHTML = `<h2>Scorecard <span class="muted">grading…</span></h2>`;
  try {
    const c = await api("/api/scorecard");
    const stat = (k, v, cls = "") => tile(k, v, cls);
    const group = (title, g) => g.length ? `<h3>${esc(title)}</h3>
      <table><thead><tr><th>${esc(title)}</th><th class="num">calls</th><th class="num">hit rate</th><th class="num">avg excess</th><th class="num">median excess</th></tr></thead>
      <tbody>${g.map((r) => `<tr><td>${pill(r.key)}</td><td class="num">${r.n}</td><td class="num">${r.hit_rate == null ? "-" : pct(r.hit_rate, false)}</td>
        <td class="num ${cls(r.avg_excess_pct)}">${pct(r.avg_excess_pct)}</td><td class="num ${cls(r.median_excess_pct)}">${pct(r.median_excess_pct)}</td></tr>`).join("")}</tbody></table>` : "";
    box.innerHTML = `<h2>Scorecard <span class="muted">${c.horizon_days}-day forward excess return vs ${esc(c.benchmark)} · graded ${esc(c.graded_as_of)}</span></h2>
      <div class="grid">
        ${stat("graded", c.n_graded)}${stat("pending", c.n_pending)}${stat("skipped", c.n_skipped)}
        ${stat("hit rate", c.hit_rate == null ? "-" : pct(c.hit_rate, false))}
        ${stat("bull − bear spread", c.bull_bear_spread_pct == null ? "-" : pct(c.bull_bear_spread_pct), cls(c.bull_bear_spread_pct))}
        ${stat("rank IC", c.rank_ic == null ? "-" : num(c.rank_ic), cls(c.rank_ic))}
      </div>
      ${c.n_graded ? "" : `<p class="muted">Nothing old enough to grade yet: a call matures ${c.horizon_days} days after its report.</p>`}
      ${group("action", c.by_action)}${group("signal", c.by_signal)}`;
  } catch (e) {
    box.innerHTML = `<h2>Scorecard</h2><div class="warn">${esc(e.message || String(e))}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Fund
// ---------------------------------------------------------------------------

async function loadMandates(select) {
  state.mandates = await api("/api/mandates");
  const sel = $("#fund-mandate");
  const current = select || sel.value || recall("aihf.mandate");
  sel.innerHTML = state.mandates.map((m) => `<option value="${esc(m.file)}" ${m.error ? "disabled" : ""}>${esc(m.name)}${m.error ? " (invalid)" : ""}</option>`).join("");
  if (current && state.mandates.some((m) => m.file === current)) sel.value = current;
  renderMandateSummary();
}

function renderMandateSummary() {
  const m = state.mandates.find((x) => x.file === $("#fund-mandate").value);
  remember("aihf.mandate", m && m.file);
  const el = $("#fund-mandate-summary");
  if (!m || m.error) { el.textContent = m ? m.error : ""; return; }
  const s = m.spec;
  el.innerHTML = s.strategies.map((st) => `<b>${esc(st.display_name || st.name)}</b> ×${st.weight} <span class="muted">(${st.models.map((x) => esc(x.name)).join(", ")}${st.blend.market_neutral ? ", market-neutral" : ""})</span>`).join(" · ")
    + `<br>${s.rebalance} · vs ${esc(s.benchmark)} · ${money(s.capital)} · max position ${pct(s.risk.max_position_pct, false)} · max gross ${num(s.risk.max_gross_exposure, 1)}x`;
}
$("#fund-mandate").addEventListener("change", renderMandateSummary);

$("#fund-delete").addEventListener("click", async () => {
  const file = $("#fund-mandate").value;
  if (!file || !confirm(`Delete mandate ${file}?`)) return;
  try { await api(`/api/mandates/${encodeURIComponent(file)}`, { method: "DELETE" }); await loadMandates(); }
  catch (e) { showError($("#fund-error"), e); }
});

$("#fund-new-toggle").addEventListener("click", () => {
  const f = $("#fund-new");
  f.hidden = !f.hidden;
  if (!f.hidden) renderStrategyPicker();
});
$("#fund-new-cancel").addEventListener("click", () => { $("#fund-new").hidden = true; });

function renderStrategyPicker() {
  $("#new-strategies").innerHTML = state.strategies.map((s) => `
    <div class="s">
      <input type="checkbox" data-name="${esc(s.name)}">
      <div><div class="t">${esc(s.title)} ${pill(s.kind, "kind")}</div><div class="d">${esc(s.models.map((m) => m.name).join(", "))}${s.blend.market_neutral ? " · market-neutral" : ""} — ${esc(s.description)}</div></div>
      <input type="number" value="1" min="0.05" step="0.05" title="capital slice (relative)">
    </div>`).join("");
}

$("#fund-new").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const errEl = $("#new-error"); showError(errEl, null);
  const strategies = $$("#new-strategies .s").filter((row) => $("input[type=checkbox]", row).checked)
    .map((row) => ({ name: $("input[type=checkbox]", row).dataset.name, weight: Number($("input[type=number]", row).value) }));
  if (!strategies.length) return showError(errEl, new Error("pick at least one strategy"));
  try {
    const m = await api("/api/mandates", { method: "POST", body: {
      name: $("#new-name").value.trim(), strategies,
      risk: { max_position_pct: Number($("#new-maxpos").value), max_gross_exposure: Number($("#new-maxgross").value) },
      capital: Number($("#new-capital").value), rebalance: $("#new-rebalance").value, benchmark: $("#new-benchmark").value.trim().toUpperCase(),
    } });
    $("#fund-new").hidden = true;
    await loadMandates(m.file);
  } catch (e) { showError(errEl, e); }
});

$("#fund-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const mode = ev.submitter && ev.submitter.dataset.mode || "cycle";
  const errEl = $("#fund-error"); showError(errEl, null);
  const tickers = $("#fund-tickers").value.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
  if (!tickers.length) return showError(errEl, new Error("give at least one ticker"));
  const body = { mandate: $("#fund-mandate").value, tickers, as_of: $("#fund-date").value, model: $("#fund-model").value };
  if (mode === "backtest") body.start = $("#fund-start").value;
  $$("#fund-form button[type=submit]").forEach((b) => b.disabled = true);
  try {
    const job = await api(`/api/${mode}`, { method: "POST", body });
    remember("aihf.job.fund", job.id);
    await followFund(job.id);
  } catch (e) { showError(errEl, e); }
  finally { $$("#fund-form button[type=submit]").forEach((b) => b.disabled = false); }
});

async function followFund(id) {
  const prog = $("#fund-progress"), errEl = $("#fund-error"), out = $("#fund-results");
  try {
    const job = await pollJob(id, (j) => {
      progressUI(prog, j.progress, j.status === "done" || j.status === "failed");
      if (j.kind === "backtest" && j.progress && j.progress.nav && j.progress.nav.length) {
        out.hidden = false;
        out.innerHTML = `<h2>${esc(j.label)} <span class="muted">running…</span></h2>` + chart(j.progress.dates, j.progress.nav, null, j.request && j.request.capital);
      }
    });
    out.hidden = false;
    out.innerHTML = job.kind === "backtest" ? renderBacktest(job.result) : renderCycle(job.result);
    bindCycleLinks(out, job.result);
  } catch (e) { progressUI(prog, null, true); showError(errEl, e); }
}

function renderBacktest(r) {
  const m = r.metrics;
  const notices = [
    ...(m.kill_switch_date ? [`<div class="error">Kill-switch: drawdown limit hit on ${esc(m.kill_switch_date)}; the fund closed to flat and stayed there.</div>`] : []),
    ...((r.warnings || []).map((w) => `<div class="warn">⚠ ${esc(w)}</div>`)),
  ].join("");
  return `<h2>${esc(r.fund)} <span class="muted">${esc(r.start)} → ${esc(r.end)} · ${esc(r.rebalance)} · ${r.universe.map(esc).join(", ")}${r.universe_kind === "dated" ? " (dated universe)" : ""}</span></h2>
    ${notices}
    <div class="grid">
      ${tile("total return", pct(m.total_return_pct), cls(m.total_return_pct))}${tile(`${esc(r.benchmark)}`, pct(m.benchmark_return_pct), cls(m.benchmark_return_pct))}
      ${tile("excess", pct(m.excess_return_pct), cls(m.excess_return_pct))}${tile("annualized", pct(m.annualized_return_pct), cls(m.annualized_return_pct))}
      ${tile("sharpe", num(m.sharpe_ratio))}${tile("max drawdown", pct(m.max_drawdown_pct, false), "neg")}${tile("cycles", m.n_cycles)}${tile("orders", m.n_orders)}${tile("costs", pct(m.costs_pct, false), "neg")}${tile("dividends", money(m.total_dividends))}
    </div>
    ${chart(r.dates, r.nav, r.benchmark_nav, r.capital)}
    <div class="legend"><span><i style="background:var(--green)"></i>fund</span><span><i style="background:var(--accent)"></i>${esc(r.benchmark)}</span></div>
    ${renderAttribution(r.attribution)}
    <h3>Cycles</h3>
    <table><thead><tr><th>date</th><th class="num">NAV</th><th class="num">vs start</th><th class="num">orders</th><th class="num">clamps</th><th>book</th></tr></thead>
    <tbody>${r.records.map((rec, i) => `<tr class="clickable" data-i="${i}"><td>${esc(rec.as_of)}</td><td class="num">${money(rec.nav)}</td>
      <td class="num ${cls(rec.nav / r.capital - 1)}">${pct(rec.nav / r.capital - 1)}</td><td class="num">${rec.orders.length}</td><td class="num">${rec.clamps.length}</td>
      <td class="ellipsis">${esc(Object.entries(rec.final_weights).filter(([, w]) => Math.abs(w) > 1e-6).map(([t, w]) => `${t} ${pct(w)}`).join("  "))}</td></tr>`).join("")}</tbody></table>
    <div id="cycle-detail"></div>`;
}

function renderAttribution(a) {
  if (!a || !a.strategies) return "";
  const rows = a.strategies.map((s) => `<tr><td>${esc(s.name)}</td><td class="num">${pct(s.slice, false)}</td>
      <td class="num ${cls(s.total_return_pct)}">${pct(s.total_return_pct)}</td><td class="num">${num(s.sharpe_ratio)}</td>
      <td class="num neg">${pct(s.max_drawdown_pct, false)}</td><td class="num ${cls(s.contribution_pct)}">${pct(s.contribution_pct)}</td></tr>`).join("");
  return `<h3>Attribution</h3>
    <p class="muted">Each strategy's paper sleeve: its own target weights at the fund's marks, compounded on its capital slice. The residual is what the sleeves do not explain: master risk clamps, share sizing, cash drag, and trading costs.</p>
    <table><thead><tr><th>strategy</th><th class="num">slice</th><th class="num">sleeve return</th><th class="num">sharpe</th><th class="num">max drawdown</th><th class="num">contributed</th></tr></thead>
    <tbody>${rows}<tr><td class="muted">residual</td><td></td><td></td><td></td><td></td><td class="num ${cls(a.residual_pct)}">${pct(a.residual_pct)}</td></tr></tbody></table>`;
}

function bindCycleLinks(out, result) {
  if (!result.records) return;
  $$("tr.clickable", out).forEach((tr) => tr.addEventListener("click", () => {
    $$("tr.selected", out).forEach((x) => x.classList.remove("selected"));
    tr.classList.add("selected");
    $("#cycle-detail", out).innerHTML = renderCycle(result.records[Number(tr.dataset.i)]);
    $("#cycle-detail", out).scrollIntoView({ behavior: "smooth", block: "start" });
  }));
}

function renderCycle(rec) {
  const tickers = Array.from(new Set([...Object.keys(rec.target_weights), ...Object.keys(rec.positions)])).sort();
  return `<h2>${esc(rec.fund)} <span class="muted">@ ${esc(rec.as_of)}</span></h2>
    <div class="grid">
      ${tile("NAV", money(rec.nav))}${tile("cash", money(rec.cash))}${tile("equity before", money(rec.equity_before))}
      ${tile("orders", rec.orders.length)}${tile("risk clamps", rec.clamps.length)}${tile("skipped", rec.skipped.length)}
    </div>
    <h3>Book</h3>
    <table><thead><tr><th>ticker</th><th class="num">target</th><th class="num">after risk</th><th class="num">mark</th><th class="num">shares</th><th class="num">value</th></tr></thead>
    <tbody>${tickers.map((t) => `<tr><td><b>${esc(t)}</b></td><td class="num ${cls(rec.target_weights[t])}">${pct(rec.target_weights[t])}</td>
      <td class="num ${cls(rec.final_weights[t])}">${pct(rec.final_weights[t])}</td><td class="num">${num(rec.marks[t])}</td>
      <td class="num">${rec.positions[t] ?? 0}</td><td class="num">${money((rec.positions[t] ?? 0) * (rec.marks[t] ?? 0))}</td></tr>`).join("")}</tbody></table>
    ${rec.clamps.length ? `<h3>Risk clamps</h3><ul>${rec.clamps.map((c) => `<li><code>${esc(c.limit)}</code> ${esc(c.ticker || "book")}: ${pct(c.before)} → ${pct(c.after)}</li>`).join("")}</ul>` : ""}
    ${rec.orders.length ? `<h3>Orders</h3><table><thead><tr><th>side</th><th>ticker</th><th class="num">qty</th><th class="num">price</th></tr></thead>
      <tbody>${rec.orders.map((o) => `<tr><td class="${o.side === "buy" ? "pos" : "neg"}">${esc(o.side)}</td><td>${esc(o.ticker)}</td><td class="num">${o.quantity}</td><td class="num">${num(o.price)}</td></tr>`).join("")}</tbody></table>` : ""}
    ${rec.strategies.map((s) => `<h3>${esc(s.name)} <span class="muted">${pct(s.slice, false)} of capital</span></h3>
      <table><thead><tr><th>ticker</th><th>model</th><th class="num">view</th><th>thesis</th></tr></thead>
      <tbody>${s.signals.map((sig) => `<tr><td><b>${esc(sig.ticker)}</b></td><td><code>${esc(sig.model_name)}</code></td>
        <td class="num ${cls(sig.value)}">${sig.metadata && sig.metadata.abstained ? "<span class='muted'>abstain</span>" : signed(sig.value)}</td>
        <td class="reason-cell">${esc(sig.reasoning || "")}${sig.metadata && sig.metadata.cached ? " <span class='muted'>(cached)</span>" : ""}</td></tr>`).join("")}</tbody></table>`).join("")}
    ${rec.skipped.length ? `<div class="warn">skipped: ${rec.skipped.map((s) => `${esc(s.ticker)} (${esc(s.reason)})`).join(", ")}</div>` : ""}`;
}

// SVG line chart: fund NAV (green/red vs capital) and optional benchmark.
function chart(dates, nav, bench, capital) {
  const W = 900, H = 260, L = 64, R = 12, T = 12, B = 26;
  const all = [...nav, ...(bench || []), ...(capital ? [capital] : [])];
  const lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || 1;
  const x = (i, n) => L + (n > 1 ? (i / (n - 1)) : 0.5) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / span) * (H - T - B);
  const line = (vals) => vals.map((v, i) => `${x(i, vals.length).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const ticks = [lo, lo + span / 2, hi];
  const down = capital && nav[nav.length - 1] < capital;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${ticks.map((v) => `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="lbl" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${money(v)}</text>`).join("")}
    ${capital ? `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(capital)}" y2="${y(capital)}" stroke-dasharray="2 4"/>` : ""}
    ${bench ? `<polyline class="bench" points="${line(bench)}"/>` : ""}
    <polyline class="fund ${down ? "down" : ""}" points="${line(nav)}"/>
    <text class="lbl" x="${L}" y="${H - 8}">${esc(dates[0] || "")}</text>
    <text class="lbl" x="${W - R}" y="${H - 8}" text-anchor="end">${esc(dates[dates.length - 1] || "")}</text>
  </svg>`;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

async function renderSettings() {
  const keys = await api("/api/keys");
  $("#keys-table").innerHTML = `<table><thead><tr><th>key</th><th>for</th><th>status</th><th>set / replace</th></tr></thead><tbody>
    ${keys.map((k) => `<tr><td><code>${esc(k.env_var)}</code></td><td class="muted">${esc(k.label)}</td>
      <td>${k.set ? `<span class="pos">set</span> <span class="muted">${esc(k.masked)}</span>` : `<span class="muted">not set</span>`}</td>
      <td><form class="row tight key-form" data-env="${esc(k.env_var)}"><input type="password" placeholder="paste key" autocomplete="off" required><button class="ghost" type="submit">save</button></form></td></tr>`).join("")}
  </tbody></table>`;
  $$(".key-form").forEach((f) => f.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    try { await api("/api/keys", { method: "POST", body: { env_var: f.dataset.env, value: $("input", f).value } }); state.status = await api("/api/status"); renderSettings(); boot(); }
    catch (e) { alert(e.message); }
  }));
  $("#llm-table").innerHTML = `<table><thead><tr><th>model</th><th>id</th><th>provider</th><th>status</th></tr></thead><tbody>
    ${state.status.llm_models.map((m) => `<tr><td>${esc(m.display_name)}${m.model === state.status.default_model ? " <span class='muted'>(default)</span>" : ""}</td><td><code>${esc(m.model)}</code></td><td>${esc(m.provider)}</td>
      <td>${!m.supported ? "<span class='muted'>no client</span>" : m.key_set ? "<span class='pos'>ready</span>" : "<span class='flat'>key missing</span>"}</td></tr>`).join("")}</tbody></table>`;
  $("#alpha-table").innerHTML = `<table><thead><tr><th>model</th><th>name</th><th>kind</th></tr></thead><tbody>
    ${state.models.map((m) => `<tr><td>${esc(m.display_name)}</td><td><code>${esc(m.name)}</code></td><td>${pill(m.kind, "kind")}</td></tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------------------
// Resume a job after a reload
// ---------------------------------------------------------------------------

async function resumeJob(kind) {
  const id = recall(`aihf.job.${kind}`);
  if (!id) return;
  try {
    const job = await api(`/api/jobs/${id}`);
    if (kind === "research") followResearch(job.id); else followFund(job.id);
  } catch { remember(`aihf.job.${kind}`, null); }  // the server restarted; jobs are in-memory
}

boot().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<div class="error">${esc(e.message)}</div>`); });
