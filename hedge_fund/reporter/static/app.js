/* aihf reporter — dashboard. Vanilla JS, no build step.

   Two modes. Served by `aihf reporter` it talks to /api and can run a cycle
   or edit the watchlist. Exported by publish.py (<html data-mode="static">,
   e.g. on Firebase Hosting) it reads the same JSON from files under /data
   and is read-only. */
(() => {
  const $ = (id) => document.getElementById(id);
  const STATIC = document.documentElement.dataset.mode === "static";
  const state = { selected: null, reportId: null, status: null, latest: [] };
  const src = {
    status: () => (STATIC ? "/data/status.json" : "/api/status"),
    latest: () => (STATIC ? "/data/latest.json" : "/api/latest"),
    runs: (n) => (STATIC ? "/data/runs.json" : `/api/runs?limit=${n}`),
    history: (t, n) => (STATIC ? `/data/history/${encodeURIComponent(t)}.json` : `/api/reports?ticker=${encodeURIComponent(t)}&limit=${n}`),
    report: (id) => (STATIC ? `/data/reports/${id}.json` : `/api/reports/${id}`),
  };

  // ------------------------------------------------------------ helpers
  async function api(path, opts = {}) {
    const r = await fetch(path, { headers: { "Content-Type": "application/json" }, cache: "no-cache", ...opts });
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
  const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const pct = (v, d = 1) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);
  const money = (v) => {
    if (v == null) return "–";
    const a = Math.abs(v);
    if (a >= 1e12) return `${(v / 1e12).toFixed(2)}T`;
    if (a >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
    if (a >= 1e6) return `${(v / 1e6).toFixed(0)}M`;
    return v.toFixed(0);
  };
  function ago(iso) {
    if (!iso) return "never";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 90) return "just now";
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${(s / 3600).toFixed(1)} h ago`;
    return `${Math.round(s / 86400)} d ago`;
  }
  function inMin(iso) {
    if (!iso) return "–";
    const s = (new Date(iso).getTime() - Date.now()) / 1000;
    if (s <= 0) return "now";
    if (s < 3600) return `in ${Math.round(s / 60)} min`;
    return `in ${(s / 3600).toFixed(1)} h`;
  }
  const local = (iso) => (iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "–");
  const pill = (v, extra = "") => `<span class="pill ${esc(v)} ${extra}">${esc(v)}</span>`;
  const delta = (cur, prev) => {
    if (prev == null || cur == null) return "";
    const d = cur - prev;
    if (Math.abs(d) < 0.5) return `<span class="delta">= ${fmt(prev, 0)}</span>`;
    return `<span class="delta ${d > 0 ? "up" : "down"}">${d > 0 ? "▲" : "▼"} ${d > 0 ? "+" : ""}${fmt(d, 0)} from ${fmt(prev, 0)}</span>`;
  };

  // ------------------------------------------------------------- status
  async function loadStatus() {
    try {
      const s = await api(src.status());
      state.status = s;
      const sch = s.scheduler;
      const parts = [];
      if (STATIC) {
        // A mirror: say how fresh it is, and when the desk will refresh it.
        const age = (Date.now() - new Date(s.published_at).getTime()) / 36e5;
        $("llm-dot").className = `dot ${age < 3 ? "ok" : "bad"}`;
        parts.push(`published ${ago(s.published_at)}`);
        if (sch.running && sch.next_run_at) parts.push(`next cycle ${inMin(sch.next_run_at)}`);
        parts.push(`${s.llm.model} via wrapper`);
      } else {
        $("llm-dot").className = `dot ${s.llm.reachable ? "ok" : "bad"}`;
        parts.push(s.llm.reachable ? `${s.llm.model} via wrapper` : `wrapper unreachable: ${s.llm.detail}`);
        if (sch.current) parts.push(`running ${sch.current.done}/${sch.current.total} (${sch.current.phase}${sch.current.ticker ? " " + sch.current.ticker : ""})`);
        else if (sch.running) parts.push(`next cycle ${inMin(sch.next_run_at)}`);
        else parts.push("scheduler off");
        if (s.publish && s.publish.enabled) {
          const p = s.publish.last;
          parts.push(!p ? "not published yet" : p.ok ? `published ${ago(p.at)}` : `publish failed ${ago(p.at)}: ${p.detail}`);
        }
      }
      if (sch.last_run) parts.push(`last ${ago(sch.last_run.finished_at || sch.last_run.started_at)}`);
      $("status-text").textContent = parts.join(" · ");
      $("status").title = `every ${sch.interval_minutes} min ${sch.active_hours} ${sch.timezone} weekdays, every ${sch.off_hours_interval_minutes} min otherwise`;
      const running = !!sch.current;
      $("progress").hidden = !running;
      if (running) $("progress-label").textContent = `cycle #${sch.current.run_id} (${sch.current.trigger}): ${sch.current.phase} — ${sch.current.done}/${sch.current.total} done${sch.current.ticker ? " · " + sch.current.ticker : ""}`;
      $("run-now").disabled = running || !sch.running;
    } catch (e) {
      $("llm-dot").className = "dot bad";
      $("status-text").textContent = `status: ${e.message}`;
    }
  }

  // -------------------------------------------------------------- cards
  async function loadLatest() {
    try {
      state.latest = await api(src.latest());
    } catch (e) {
      $("cards").innerHTML = `<div class="error">${esc(e.message)}</div>`;
      return;
    }
    const box = $("cards");
    box.innerHTML = "";
    if (!state.latest.length) { box.innerHTML = '<div class="empty">No symbols yet — add one above.</div>'; return; }
    const n = state.latest.filter((x) => x.brief).length;
    $("dash-sub").textContent = `${state.latest.length} names · ${n} briefed`;
    const order = [...state.latest].sort((a, b) => (b.brief ? sc(b.brief) : -999) - (a.brief ? sc(a.brief) : -999));
    for (const row of order) box.appendChild(card(row));
  }
  const sc = (b) => ({ bullish: 1, neutral: 0, bearish: -1 }[b.signal] || 0) * b.confidence;

  function card(row) {
    const b = row.brief;
    const att = row.last_attempt;
    const failed = att && att.status === "failed";
    const q = b && b.quote;
    const stale = b && Date.now() - new Date(b.generated_at).getTime() > 6 * 3600e3;
    const c = el(`<div class="card ${failed ? "failed" : ""} ${stale ? "stale" : ""} ${state.selected === row.ticker ? "selected" : ""}" data-t="${esc(row.ticker)}">
      <button class="remove" title="remove from watchlist">×</button>
      <div class="top"><span class="ticker">${esc(row.ticker)}</span>
        ${q && q.price != null ? `<span class="price">${fmt(q.price)}</span>` : ""}
        ${q && q.change_pct != null ? `<span class="chg ${q.change_pct >= 0 ? "pos" : "neg"}">${pct(q.change_pct, 2)}</span>` : ""}</div>
      ${b ? `<div class="pills">${pill(b.signal)}<span class="conf">${fmt(b.confidence, 0)}</span>${pill(b.action)}${b.nothing_new ? pill("quiet", "quiet") : ""}${delta(b.confidence, b.prev_confidence)}</div>` : ""}
      <div class="headline">${b ? esc(b.headline || "(no headline)") : failed ? esc(att.error || "failed") : '<span class="muted">not briefed yet</span>'}</div>
      <div class="when"><span>${b ? "briefed " + ago(b.generated_at) : ""}</span><span>${failed ? "last attempt failed " + ago(att.generated_at) : ""}</span></div>
    </div>`);
    c.addEventListener("click", (ev) => { if (ev.target.classList.contains("remove")) return; select(row.ticker, row.report_id); });
    c.querySelector(".remove").addEventListener("click", async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Remove ${row.ticker} from the watchlist? Its history stays.`)) return;
      try { await api(`/api/symbols/${encodeURIComponent(row.ticker)}`, { method: "DELETE" }); await loadLatest(); }
      catch (e) { alert(e.message); }
    });
    return c;
  }

  // ------------------------------------------------------------- detail
  async function select(ticker, reportId) {
    state.selected = ticker;
    state.reportId = reportId;
    document.querySelectorAll(".card").forEach((c) => c.classList.toggle("selected", c.dataset.t === ticker));
    const box = $("detail");
    box.hidden = false;
    box.innerHTML = '<div class="empty">loading…</div>';
    try {
      const history = await api(src.history(ticker, 48));
      let id = reportId;
      if (id == null) { const ok = history.find((h) => h.status === "ok"); id = ok ? ok.id : null; }
      if (id == null) {
        const last = history[0];
        box.innerHTML = `<div class="head"><span class="ticker">${esc(ticker)}</span><button class="ghost close">close</button></div>` +
          (last ? `<div class="error">${esc(last.error || "no brief")}</div>` : '<div class="empty">No brief yet. Run a cycle.</div>');
        box.querySelector(".close").onclick = () => { box.hidden = true; state.selected = null; };
        return;
      }
      const row = await api(src.report(id));
      renderDetail(box, row, history);
      box.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      box.innerHTML = `<div class="error">${esc(e.message)}</div>`;
    }
  }

  function renderDetail(box, row, history) {
    const b = row.brief;
    const q = b.quote || {};
    const t = b.technicals || {};
    const srcs = b.sources || [];
    const cite = (idx) => (idx && idx.length ? `<span class="cite">${idx.map((i) => `[${i}]`).join("")}</span>` : "");
    const claims = (list) => (list && list.length ? `<ul class="list">${list.map((c) => `<li>${esc(c.text)}${cite(c.source_indices)}</li>`).join("")}</ul>` : '<div class="muted">none listed</div>');
    const bullets = (list) => (list && list.length ? `<ul class="list">${list.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<div class="muted">none</div>');
    const desk = (b.desk || []).map((s) => {
      const abst = s.metadata && s.metadata.abstained === true;
      const v = abst ? 0 : s.value;
      const w = Math.min(50, Math.abs(v) * 50);
      return `<div class="name">${esc(s.model_name)}</div>
        <div class="gauge">${abst ? "" : `<div class="g ${v >= 0 ? "pos" : "neg"}" style="width:${w}%"></div>`}</div>
        <div class="val">${abst ? "–" : (v >= 0 ? "+" : "") + fmt(v, 2)}</div>
        <div class="reason">${esc(abst ? "abstained — " + (s.metadata.abstain_reason || "") : s.reasoning || "")}</div>`;
    }).join("");
    const hist = history.map((h) => {
      const hb = h.brief;
      const label = hb ? `${local(h.generated_at)} ${hb.signal[0].toUpperCase()}${fmt(hb.confidence, 0)} ${hb.action}` : `${local(h.generated_at)} ✗`;
      return `<button class="${h.id === row.id ? "current" : ""}" data-id="${h.id}" title="${esc(hb ? hb.headline : h.error)}">${esc(label)}</button>`;
    }).join("");
    box.innerHTML = `
      <div class="head">
        <span class="ticker">${esc(b.ticker)}</span>
        ${pill(b.signal, "big")}<span class="conf">confidence ${fmt(b.confidence, 0)} ${delta(b.confidence, b.prev_confidence)}</span>
        ${pill(b.action, "big")}${b.nothing_new ? pill("quiet hour", "quiet") : ""}
        <span class="meta">${local(b.generated_at)} · ${esc(b.model)}${b.prev_generated_at ? " · prev " + ago(b.prev_generated_at) : ""}</span>
        <button class="ghost close">close</button>
      </div>
      <div class="headline">${esc(b.headline)}</div>
      <div class="grid">
        <div class="tile"><div class="k">last</div><div class="v ${q.change_pct >= 0 ? "pos" : "neg"}">${fmt(q.price)} ${q.change_pct != null ? pct(q.change_pct, 2) : ""}</div></div>
        <div class="tile"><div class="k">day range</div><div class="v">${fmt(q.day_low)} – ${fmt(q.day_high)}</div></div>
        <div class="tile"><div class="k">volume vs 3m</div><div class="v">${q.volume && q.avg_volume_3m ? Math.round((q.volume / q.avg_volume_3m) * 100) + "%" : "–"}</div></div>
        <div class="tile"><div class="k">mkt cap</div><div class="v">${money(q.market_cap)}</div></div>
        <div class="tile"><div class="k">1m / 3m</div><div class="v">${pct(t.ret_1m)} / ${pct(t.ret_3m)}</div></div>
        <div class="tile"><div class="k">RSI 14</div><div class="v">${fmt(t.rsi_14, 0)}</div></div>
        <div class="tile"><div class="k">SMA 50 / 200</div><div class="v">${fmt(t.sma_50, 0)} / ${fmt(t.sma_200, 0)}</div></div>
        <div class="tile"><div class="k">from 52w high</div><div class="v">${pct(t.pct_from_52w_high)}</div></div>
      </div>
      <h3>What's new</h3>${bullets(b.whats_new)}
      <h3>Brief</h3><div class="summary">${(b.summary || "").split(/\n\s*\n/).map((p) => `<p>${esc(p)}</p>`).join("")}</div>
      <div class="two">
        <div><h3>Catalysts</h3>${claims(b.catalysts)}</div>
        <div><h3>Risks</h3>${claims(b.risks)}</div>
      </div>
      <h3>Watch next</h3>${bullets(b.watch_next)}
      <h3>Sources (${srcs.length})</h3>
      <ul class="sources">${srcs.map((s, i) => `<li><span class="n">[${i}]</span><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a><span class="d">${esc(s.published || "")}</span></li>`).join("") || '<li class="muted">none cited</li>'}</ul>
      <h3>Quant desk</h3><div class="desk">${desk || '<div class="muted">no readout</div>'}</div>
      ${b.warnings && b.warnings.length ? `<div class="warn">⚠ ${esc(b.warnings.join("; "))}</div>` : ""}
      <h3>History</h3><div class="history">${hist}</div>
    `;
    box.querySelector(".close").onclick = () => { box.hidden = true; state.selected = null; document.querySelectorAll(".card").forEach((c) => c.classList.remove("selected")); };
    box.querySelectorAll(".history button").forEach((btn) => btn.addEventListener("click", () => select(b.ticker, Number(btn.dataset.id))));
  }

  // --------------------------------------------------------------- runs
  async function loadRuns() {
    try {
      const runs = await api(src.runs(20));
      $("runs").innerHTML = runs.length
        ? `<table><tr><th>#</th><th>started</th><th>trigger</th><th>status</th><th>ok</th><th>failed</th><th>took</th></tr>${runs.map((r) => {
            const took = r.finished_at ? Math.round((new Date(r.finished_at) - new Date(r.started_at)) / 60000) + " min" : "…";
            return `<tr><td>${r.id}</td><td>${local(r.started_at)}</td><td>${esc(r.trigger)}</td><td>${esc(r.status)}</td><td>${r.n_ok}</td><td>${r.n_failed}</td><td>${took}</td></tr>`;
          }).join("")}</table>`
        : '<div class="empty">no cycles yet</div>';
    } catch (e) { $("runs").innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  }

  // ------------------------------------------------------------- events
  $("run-now").addEventListener("click", async () => {
    try { await api("/api/run", { method: "POST", body: "{}" }); await loadStatus(); }
    catch (e) { alert(e.message); }
  });
  $("refresh").addEventListener("click", refresh);
  $("add-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const t = $("add-ticker").value.trim().toUpperCase();
    if (!t) return;
    try { await api("/api/symbols", { method: "POST", body: JSON.stringify({ ticker: t }) }); $("add-ticker").value = ""; await loadLatest(); }
    catch (e) { alert(e.message); }
  });

  async function refresh() { await Promise.all([loadStatus(), loadLatest(), loadRuns()]); }
  if (STATIC) document.body.classList.add("static");
  refresh();
  setInterval(loadStatus, 15000);
  setInterval(() => { loadLatest(); loadRuns(); }, 60000);
})();
