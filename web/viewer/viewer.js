// 2D map player for tracks.json (format: src/futsal/tracks.py).
// Data source: ?src=<url of a tracks.json>  or the server API (/api/matches, /api/matches/<id>/tracks).
(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("pitch"), ctx = canvas.getContext("2d");
  const MARGIN = 2.5;              // metres of run-off drawn around the lines
  const TRAIL_S = 3;
  let data = null, frame = 0, playing = false, lastTs = 0, selected = null, heat = null;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const teamColor = (t) => css(t === "A" ? "--team-a" : t === "B" ? "--team-b" : "--team-x");
  const params = new URLSearchParams(location.search);

  function status(msg) { $("status").textContent = msg || ""; }
  function fmt(s) { s = Math.max(0, Math.floor(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

  // ---------- data ----------
  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  }

  async function loadMatches(selectId) {
    const sel = $("match");
    let list = [];
    try { list = await getJSON("../api/matches"); } catch { sel.disabled = true; $("demo").disabled = true; return; }
    sel.innerHTML = list.length ? "" : "<option value=''>경기가 없습니다</option>";
    for (const m of list) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = `${m.title || m.id} · ${m.court} · ${statusKo(m.status)}`;
      sel.appendChild(o);
    }
    if (selectId) sel.value = selectId;
    if (sel.value) loadTracks(`../api/matches/${sel.value}/tracks`);
  }

  function statusKo(s) { return { created: "생성됨", uploading: "업로드 중", processing: "분석 중", done: "완료", failed: "실패" }[s] || s; }

  async function loadTracks(url) {
    status("불러오는 중…");
    try {
      data = await getJSON(url);
    } catch (e) {
      data = null; draw(); status(`결과가 아직 없습니다 (${e.message.slice(0, 60)})`); return;
    }
    frame = 0; selected = null; heat = null;
    $("seek").max = data.n_frames - 1; $("seek").value = 0;
    resize(); renderTable(); draw();
    status(`선수 ${data.players.length}명 · ${fmt(data.n_frames / data.fps)} · ${data.fps}Hz`);
  }

  // ---------- drawing ----------
  let scale = 10;
  function resize() {
    if (!data) return;
    const L = data.court.length, W = data.court.width;
    const cssW = canvas.clientWidth || canvas.parentElement.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    scale = cssW / (L + 2 * MARGIN);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round((W + 2 * MARGIN) * scale * dpr);
    canvas.style.height = `${(W + 2 * MARGIN) * scale}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  // pitch metres -> canvas px (y flipped so y=0, the bench side, is at the bottom)
  const X = (x) => (x + MARGIN) * scale;
  const Y = (y) => (data.court.width - y + MARGIN) * scale;

  function drawPitch() {
    const L = data.court.length, W = data.court.width;
    ctx.fillStyle = css("--grass");
    ctx.fillRect(0, 0, X(L + MARGIN), Y(-MARGIN));
    ctx.fillStyle = css("--grass-dark");
    for (let i = 0; i < 8; i++) if (i % 2) ctx.fillRect(X(i * L / 8), Y(W), (L / 8) * scale, W * scale);
    if (heat) drawHeat();
    ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.lineWidth = Math.max(1.5, scale * 0.08);
    for (const line of Court.markings(L, W)) {
      ctx.beginPath();
      line.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
    }
    ctx.fillStyle = "white";
    for (const [x, y] of Court.spots(L, W)) { ctx.beginPath(); ctx.arc(X(x), Y(y), Math.max(2, scale * .12), 0, 7); ctx.fill(); }
    for (const g of Court.goals(L, W)) {
      ctx.fillStyle = "rgba(255,255,255,.85)";
      ctx.fillRect(X(g.x), Y(g.y + g.h), g.w * scale, g.h * scale);
      ctx.strokeStyle = "#222"; ctx.lineWidth = Math.max(2, scale * .15);
      ctx.beginPath(); ctx.moveTo(X(g.line), Y(g.y)); ctx.lineTo(X(g.line), Y(g.y + g.h)); ctx.stroke();
    }
  }

  function drawHeat() {
    const { grid, nx, ny, cell, max } = heat;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = grid[j * nx + i] / max;
      if (v < 0.02) continue;
      ctx.fillStyle = `rgba(255, ${Math.round(220 * (1 - v))}, 0, ${0.15 + 0.6 * v})`;
      ctx.fillRect(X(i * cell), Y((j + 1) * cell), cell * scale + 0.5, cell * scale + 0.5);
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!data) return;
    drawPitch();
    const trail = $("trails").checked ? Math.round(TRAIL_S * data.fps) : 0;
    const r = Math.max(5, scale * 0.45);
    for (const p of data.players) {
      if (selected !== null && p.id !== selected) ctx.globalAlpha = 0.35;
      const col = teamColor(p.team);
      if (trail) {
        ctx.strokeStyle = col; ctx.lineWidth = Math.max(1.5, scale * .1); ctx.beginPath();
        let pen = false;
        for (let k = Math.max(0, frame - trail); k <= frame; k++) {
          const x = p.x[k], y = p.y[k];
          if (x == null) { pen = false; continue; }
          pen ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); pen = true;
        }
        ctx.stroke();
      }
      const x = p.x[frame], y = p.y[frame];
      if (x != null) {
        ctx.fillStyle = col; ctx.strokeStyle = "white"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X(x), Y(y), r, 0, 7); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "white"; ctx.font = `600 ${Math.round(r * 1.1)}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(p.id, X(x), Y(y) + 0.5);
      }
      ctx.globalAlpha = 1;
    }
    $("seek").value = frame;
    $("clock").textContent = `${fmt(frame / data.fps)} / ${fmt(data.n_frames / data.fps)}`;
  }

  // ---------- table + heatmap ----------
  function renderTable() {
    const tb = $("stats").querySelector("tbody");
    tb.innerHTML = "";
    const rows = [...data.players].sort((a, b) => (a.team > b.team ? 1 : a.team < b.team ? -1 : a.id - b.id));
    for (const p of rows) {
      const s = p.stats || {};
      const tr = document.createElement("tr");
      tr.dataset.id = p.id;
      tr.innerHTML = `<td>${p.id}</td><td><span class="chip" style="background:${teamColor(p.team)}"></span>${p.name || `${p.team}팀`}</td>
        <td class="num">${s.distance_m != null ? Math.round(s.distance_m) + " m" : "–"}</td>
        <td class="num">${s.max_speed_ms != null ? (s.max_speed_ms * 3.6).toFixed(1) + " km/h" : "–"}</td>
        <td class="num">${s.sprints ?? "–"}</td>`;
      tr.onclick = () => select(p.id === selected ? null : p.id);
      tb.appendChild(tr);
    }
  }

  function select(id) {
    selected = id; heat = null;
    for (const tr of $("stats").querySelectorAll("tbody tr")) tr.classList.toggle("sel", Number(tr.dataset.id) === id);
    if (id !== null) {
      const p = data.players.find((q) => q.id === id);
      const cell = 1, nx = Math.ceil(data.court.length / cell), ny = Math.ceil(data.court.width / cell);
      const grid = new Float32Array(nx * ny);
      for (let k = 0; k < p.x.length; k++) {
        if (p.x[k] == null) continue;
        const i = Math.min(nx - 1, Math.max(0, Math.floor(p.x[k] / cell)));
        const j = Math.min(ny - 1, Math.max(0, Math.floor(p.y[k] / cell)));
        grid[j * nx + i]++;
      }
      // light blur so the map reads as an area, not a scatter of cells
      const blur = new Float32Array(nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        let s = 0, w = 0;
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
          const a = i + di, b = j + dj;
          if (a < 0 || b < 0 || a >= nx || b >= ny) continue;
          const k = di || dj ? 0.5 : 1; s += grid[b * nx + a] * k; w += k;
        }
        blur[j * nx + i] = s / w;
      }
      heat = { grid: blur, nx, ny, cell, max: Math.max(...blur) || 1 };
    }
    draw();
  }

  // ---------- playback ----------
  function tick(ts) {
    if (!playing || !data) return;
    const dt = (ts - lastTs) / 1000; lastTs = ts;
    frame = Math.min(data.n_frames - 1, frame + Math.max(1, Math.round(dt * data.fps * Number($("speed").value))));
    if (frame >= data.n_frames - 1) setPlaying(false);
    draw();
    if (playing) requestAnimationFrame(tick);
  }
  function setPlaying(on) {
    playing = on && !!data;
    $("play").textContent = playing ? "❚❚" : "▶";
    $("play").setAttribute("aria-label", playing ? "일시정지" : "재생");
    if (playing) { if (frame >= data.n_frames - 1) frame = 0; lastTs = performance.now(); requestAnimationFrame(tick); }
  }

  $("play").onclick = () => setPlaying(!playing);
  $("seek").oninput = (e) => { frame = Number(e.target.value); draw(); };
  $("trails").onchange = draw;
  $("match").onchange = (e) => e.target.value && loadTracks(`../api/matches/${e.target.value}/tracks`);
  $("demo").onclick = async () => {
    status("데모 경기를 만드는 중…");
    try { const m = await getJSON("../api/demo?seconds=300", { method: "POST" }); await loadMatches(m.id); }
    catch (e) { status(`데모 생성 실패: ${e.message}`); }
  };
  window.addEventListener("resize", () => { resize(); draw(); });
  window.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target.tagName !== "INPUT") { e.preventDefault(); setPlaying(!playing); } });

  if (params.get("src")) { $("match").hidden = true; $("demo").hidden = true; loadTracks(params.get("src")); }
  else loadMatches(params.get("match"));
})();
