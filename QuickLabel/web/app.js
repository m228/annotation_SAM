"use strict";
// ── QuickLabel front-end ──────────────────────────────────────────
// Vanilla JS single-page client. Annotations are stored in image-pixel
// coordinates; the canvas draws them with a fit-to-view transform.

const $ = (id) => document.getElementById(id);

const state = {
  slug: null,
  project: null,
  currentImage: null,
  currentClassId: null,
  tool: "select",
  selectedAnnId: null,
  lastPrompt: "",
  // canvas view transform: screen = img*scale + off; `fit` is the zoom-to-fit
  // scale and doubles as the minimum zoom.
  view: { scale: 1, offX: 0, offY: 0, fit: 1 },
  // transient interaction
  imgEl: null,
  imgSeq: 0,            // monotonic token guarding async image loads
  drag: null,           // {x0,y0,x1,y1} during box drawing (image coords)
  pan: null,            // {sx,sy,offX,offY} during pan
  polyPoints: [],       // [{x,y}] vertices of the polygon being drawn
  polyCursor: null,     // {x,y} live cursor for the rubber-band segment
  samPoints: [],        // [{x,y,is_positive}]
  pending: null,        // {bbox, polygon} awaiting accept
  currentJobId: null,   // running SAM job (for progress/cancel)
  exportTarget: "yolo", // "yolo" | "coco" (chosen from the export dropdown)
};

// ── API helper ────────────────────────────────────────────────────
async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const res = await fetch(path, opt);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Full-screen blocking overlay — only for import / upload / export.
function busy(on, msg = "Обработка…") {
  $("busyMsg").textContent = msg || "Обработка…";
  $("busy").classList.toggle("hidden", !on);
}

// ── SAM card: lightweight top-right spinner (no screen dimming) that morphs
// into the accept/cancel confirm. Two sub-views in #pendingBar. ──
function showSamLoading(msg) {
  $("pendingConfirm").classList.add("hidden");
  $("pendingLoading").classList.remove("hidden");
  $("pendingBar").classList.remove("hidden");
  $("pendingLoadMsg").textContent = msg || "Обработка…";
  $("samProgressBar").style.width = "0%";
}

function updateSamProgress(progress) {
  if (!progress) return;
  $("pendingLoadMsg").textContent = progress.message || "Обработка…";
  $("samProgressBar").style.width = (progress.percent || 0) + "%";
}

class JobCancelled extends Error {}

// Start a SAM job, show the inline spinner + cancel, poll until it finishes.
async function runJob(url, body, label) {
  const { job_id } = await api("POST", url, body);
  state.currentJobId = job_id;
  showSamLoading(label || "Обработка…");
  try {
    while (true) {
      await sleep(350);
      const j = await api("GET", `/api/jobs/${job_id}`);
      updateSamProgress(j.progress);
      if (j.status === "done") return j.result;
      if (j.status === "cancelled") throw new JobCancelled();
      if (j.status === "error") throw new Error(j.error || "Ошибка задачи");
    }
  } finally {
    state.currentJobId = null;
    // Hide the loader. If no confirm view is about to show (set synchronously
    // by the caller before the next paint), hide the whole card too.
    $("pendingLoading").classList.add("hidden");
    if ($("pendingConfirm").classList.contains("hidden")) {
      $("pendingBar").classList.add("hidden");
    }
  }
}

async function cancelCurrentJob() {
  if (!state.currentJobId) return;
  $("pendingLoadMsg").textContent = "Отмена…";
  try { await api("POST", `/api/jobs/${state.currentJobId}/cancel`); } catch {}
}

// ── Projects ──────────────────────────────────────────────────────
async function loadProjects(selectSlug) {
  const list = await api("GET", "/api/projects");
  const sel = $("projectSelect");
  sel.innerHTML = '<option value="">— выберите проект —</option>' +
    list.map((p) => `<option value="${p.slug}">${escapeHtml(p.name)} (${p.images})</option>`).join("");
  if (selectSlug) { sel.value = selectSlug; await openProject(selectSlug); }
}

async function openProject(slug) {
  $("delProjectBtn").disabled = !slug;
  if (!slug) {
    $("workspace").classList.add("hidden"); state.slug = null;
    $("trainBtn").disabled = true; return;
  }
  state.slug = slug;
  await refreshProject();
  $("workspace").classList.remove("hidden");
  $("exportBtn").disabled = false;
  $("trainBtn").disabled = false;
  if (state.project.classes.length) selectClass(state.project.classes[0].id);
  else state.currentClassId = null;
  if (state.project.images.length) selectImage(state.project.images[0]);
  else { state.currentImage = null; draw(); }
}

async function deleteProject() {
  if (!state.slug) return;
  const name = state.project ? state.project.name : state.slug;
  if (!confirm(`Удалить проект «${name}» со всеми изображениями и аннотациями? Это действие необратимо.`)) return;
  await api("DELETE", `/api/projects/${state.slug}`);
  state.slug = null; state.project = null; state.currentImage = null; state.imgEl = null;
  $("workspace").classList.add("hidden");
  $("exportBtn").disabled = true;
  $("trainBtn").disabled = true;
  $("delProjectBtn").disabled = true;
  await loadProjects();
}

async function refreshProject() {
  state.project = await api("GET", `/api/projects/${state.slug}`);
  if (!state.project.static_rois) state.project.static_rois = [];
  // keep currentImage reference fresh
  if (state.currentImage) {
    state.currentImage = state.project.images.find((i) => i.id === state.currentImage.id) || null;
  }
  renderImageList();
  renderClassList();
  renderAnnList();
}

// ── Image list ────────────────────────────────────────────────────
function imageStatus(img) {
  const c = img.annotations.filter((a) => a.status === "confirmed").length;
  const s = img.annotations.filter((a) => a.status === "suggested").length;
  return { c, s };
}

function renderImageList() {
  const ul = $("imageList");
  ul.innerHTML = "";
  state.project.images.forEach((img) => {
    const { c, s } = imageStatus(img);
    const li = document.createElement("li");
    if (state.currentImage && img.id === state.currentImage.id) li.classList.add("active");
    let badge = "";
    if (c) badge = `<span class="badge done">${c}</span>`;
    else if (s) badge = `<span class="badge suggest">${s}?</span>`;
    li.innerHTML = `<span class="img-name">${escapeHtml(img.filename)}</span>${badge}` +
      `<button class="row-del" title="Удалить изображение">✕</button>`;
    li.onclick = (e) => { if (!e.target.classList.contains("row-del")) selectImage(img); };
    li.querySelector(".row-del").onclick = (e) => { e.stopPropagation(); deleteImage(img); };
    ul.appendChild(li);
  });
  $("imageCount").textContent = `${state.project.images.length}`;
}

async function deleteImage(img) {
  if (!confirm(`Удалить изображение «${img.filename}» и его аннотации?`)) return;
  await api("DELETE", `/api/projects/${state.slug}/image/${img.id}`);
  const wasCurrent = state.currentImage && state.currentImage.id === img.id;
  await refreshProject();
  if (wasCurrent) {
    if (state.project.images.length) selectImage(state.project.images[0]);
    else { state.currentImage = null; state.imgEl = null; draw(); }
  }
}

function selectImage(img) {
  const sameImage = state.currentImage && state.currentImage.id === img.id;
  state.currentImage = img;
  state.selectedAnnId = null;
  state.samPoints = [];
  state.pending = null;
  // Drop the previous frame so we never show image A under image B's contours.
  // (Skip the clear when re-selecting the same image, e.g. after a save.)
  if (!sameImage) state.imgEl = null;
  hidePending();
  // A monotonic token guards against out-of-order image loads when the user
  // switches images quickly: only the newest selection may set state.imgEl.
  const token = ++state.imgSeq;
  const el = new Image();
  el.onload = () => {
    if (token !== state.imgSeq) return;     // a newer image won the race
    state.imgEl = el; fitView(); draw();
  };
  el.onerror = () => {
    if (token !== state.imgSeq) return;
    state.imgEl = null; draw();
    console.error("QuickLabel: не удалось загрузить изображение", img.filename);
  };
  el.src = `/api/projects/${state.slug}/image/${img.id}?t=${Date.now()}`;
  draw();                       // reflect the cleared/!current state immediately
  renderImageList();
  renderAnnList();
  updateSahiForecast();
}

// ── Classes ───────────────────────────────────────────────────────
function renderClassList() {
  const ul = $("classList");
  ul.innerHTML = "";
  state.project.classes.forEach((c) => {
    const li = document.createElement("li");
    if (c.id === state.currentClassId) li.classList.add("active");
    li.innerHTML = `<input type="color" class="swatch class-color" value="${c.color}" title="Цвет класса">
      <span class="class-name">${escapeHtml(c.name)}</span>
      <button class="row-del" title="Удалить">✕</button>`;
    li.onclick = (e) => {
      if (e.target.classList.contains("row-del") || e.target.classList.contains("class-color")) return;
      selectClass(c.id);
    };
    const picker = li.querySelector(".class-color");
    picker.onclick = (e) => e.stopPropagation();             // don't select class
    picker.oninput = (e) => { c.color = e.target.value; draw(); };   // live preview
    picker.onchange = (e) => updateClassColor(c.id, e.target.value); // persist
    li.querySelector(".row-del").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Удалить класс «${c.name}» и все его аннотации?`)) return;
      await api("DELETE", `/api/projects/${state.slug}/classes/${c.id}`);
      if (state.currentClassId === c.id) state.currentClassId = null;
      await refreshProject(); draw();
    };
    ul.appendChild(li);
  });
  renderPropClasses();
}

// Selecting a class auto-fills the SAM 3 prompt fields with its name so the
// "find on current" / "propagate" flow defaults to searching for that class.
function selectClass(id) {
  state.currentClassId = id;
  const cls = classById(id);
  if (cls) {
    $("sam3Prompt").value = cls.name;
    $("propPrompt").value = cls.name;
  }
  renderClassList();
}

function renderPropClasses() {
  const sel = $("propClass");
  if (!sel) return;
  const prev = state.currentClassId;
  sel.innerHTML = state.project.classes
    .map((c) => `<option value="${c.id}">${escapeHtml(c.name)}</option>`).join("");
  if (prev !== null && prev !== undefined) sel.value = String(prev);
}

async function addClass() {
  const name = $("newClassName").value.trim();
  if (!name) return;
  const cls = await api("POST", `/api/projects/${state.slug}/classes`, { name });
  $("newClassName").value = "";
  await refreshProject();
  selectClass(cls.id);
}

function classById(id) { return state.project.classes.find((c) => c.id === id); }

async function updateClassColor(cid, color) {
  await api("PATCH", `/api/projects/${state.slug}/classes/${cid}`, { color });
  const c = classById(cid);
  if (c) c.color = color;
  draw();
}

// ── Annotations list ──────────────────────────────────────────────
function curAnns() { return state.currentImage ? state.currentImage.annotations : []; }
function staticRois() { return (state.project && state.project.static_rois) || []; }

// True if this static ROI is hidden ("apply to all EXCEPT this frame").
function roiHiddenHere(roi) {
  const id = state.currentImage && state.currentImage.id;
  return id && (roi.exceptions || []).includes(id);
}

// Static ROIs that should render on the current image.
function visibleStaticRois() { return staticRois().filter((r) => !roiHiddenHere(r)); }

// Convert a static ROI's normalised coords to pixels for the current image.
// Falls back to legacy pixel coords (for ROIs created before normalisation).
function roiOnImage(roi, img) {
  if (!img) return { polygon: roi.polygon || null, bbox: roi.bbox || null };
  let polygon = null, bbox = null;
  if (roi.polygon_norm && roi.polygon_norm.length >= 3) {
    polygon = roi.polygon_norm.map((p) => ({
      x: Math.round(p.x * img.width), y: Math.round(p.y * img.height),
    }));
  } else if (roi.polygon) {
    polygon = roi.polygon;
  }
  if (roi.bbox_norm) {
    const n = roi.bbox_norm;
    bbox = {
      x: Math.round(n.x * img.width), y: Math.round(n.y * img.height),
      width: Math.round(n.width * img.width), height: Math.round(n.height * img.height),
    };
  } else if (roi.bbox) {
    bbox = roi.bbox;
  }
  return { polygon, bbox };
}

// Everything shown on the canvas: static ROIs (under) + this image's anns.
function allDisplayAnns() { return visibleStaticRois().concat(curAnns()); }

function annListRow(a, isStatic) {
  const cls = classById(a.class_id);
  const li = document.createElement("li");
  if (a.status === "suggested") li.classList.add("suggested");
  if (isStatic) li.classList.add("static");
  if (isStatic && roiHiddenHere(a)) li.classList.add("hidden-here");
  if (a.id === state.selectedAnnId) li.classList.add("selected");
  const conf = a.confidence ? `<span class="ann-conf">${(a.confidence * 100).toFixed(0)}%</span>` : "";
  const hasPoly = (a.polygon_norm && a.polygon_norm.length >= 3) || (a.polygon && a.polygon.length >= 3);
  const shape = hasPoly ? "полигон" : "рамка";
  // Static ROIs get a second checkbox: "apply on this frame" (exception toggle).
  const excTitle = "Применять на этом кадре (снять — исключить эту рамку только тут)";
  const excChk = isStatic
    ? `<input type="checkbox" class="exc-chk" title="${excTitle}" ${roiHiddenHere(a) ? "" : "checked"}>`
    : "";
  li.innerHTML =
    `<input type="checkbox" class="static-chk" title="Статичная: применять ко всем кадрам" ${isStatic ? "checked" : ""}>` +
    excChk +
    `<span class="swatch" style="background:${cls ? cls.color : "#888"}"></span>` +
    `<span class="class-name">${cls ? escapeHtml(cls.name) : "?"} <small class="muted">${shape}${isStatic ? " · всем" : ""}</small></span>` +
    `${conf}<button class="row-del">✕</button>`;
  li.onclick = (e) => {
    const t = e.target.classList;
    if (t.contains("row-del") || t.contains("static-chk") || t.contains("exc-chk")) return;
    state.selectedAnnId = a.id; renderAnnList(); draw();
  };
  li.querySelector(".static-chk").onchange = () => toggleStatic(a, isStatic);
  const exc = li.querySelector(".exc-chk");
  if (exc) exc.onchange = () => toggleException(a);
  li.querySelector(".row-del").onclick = (e) => { e.stopPropagation(); removeAnn(a.id); };
  return li;
}

// Toggle whether a static ROI applies on the CURRENT frame. The ROI itself
// stays project-wide; we just add/remove this image_id from its exception list.
async function toggleException(roi) {
  if (!state.currentImage) return;
  const id = state.currentImage.id;
  const exc = roi.exceptions || [];
  roi.exceptions = exc.includes(id) ? exc.filter((x) => x !== id) : exc.concat([id]);
  await saveStaticRois();
  renderAnnList(); draw();
}

function renderAnnList() {
  const ul = $("annList");
  ul.innerHTML = "";
  staticRois().forEach((a) => ul.appendChild(annListRow(a, true)));
  let hasSuggest = false;
  curAnns().forEach((a) => {
    if (a.status === "suggested") hasSuggest = true;
    ul.appendChild(annListRow(a, false));
  });
  const total = staticRois().length + curAnns().length;
  $("annCount").textContent = total ? `${total}` : "";
  $("suggestActions").classList.toggle("hidden", !hasSuggest);
  // Bring the selected row into view (e.g. after picking a crystal on canvas).
  const selLi = ul.querySelector("li.selected");
  if (selLi) selLi.scrollIntoView({ block: "nearest" });
}

// Move an ROI between this image's annotations and the project-wide static set.
// Static ROIs are stored in NORMALISED 0..1 coords so they apply correctly to
// images of any size; we (de)normalise against the current image when toggling.
async function toggleStatic(a, wasStatic) {
  const img = state.currentImage;
  if (wasStatic) {
    // Static → per-image: scale back to pixel coords for *this* image.
    const disp = roiOnImage(a, img);
    const moved = Object.assign({}, a, {
      static: false, polygon: disp.polygon, bbox: disp.bbox,
    });
    delete moved.polygon_norm; delete moved.bbox_norm; delete moved.exceptions;
    state.project.static_rois = staticRois().filter((r) => r.id !== a.id);
    if (img) img.annotations.push(moved);
    await saveStaticRois();
    await saveAnns();
  } else {
    // Per-image → static: normalise pixel coords using the current image size.
    if (!img || !img.width || !img.height) return;
    const moved = Object.assign({}, a, { static: true, exceptions: [] });
    if (a.polygon && a.polygon.length >= 3) {
      moved.polygon_norm = a.polygon.map((p) => ({
        x: p.x / img.width, y: p.y / img.height,
      }));
    }
    if (a.bbox) {
      moved.bbox_norm = {
        x: a.bbox.x / img.width, y: a.bbox.y / img.height,
        width: a.bbox.width / img.width, height: a.bbox.height / img.height,
      };
    }
    // Drop the pixel copies so display always uses the normalised source.
    delete moved.polygon; delete moved.bbox;
    img.annotations = curAnns().filter((r) => r.id !== a.id);
    state.project.static_rois = staticRois().concat([moved]);
    await saveAnns();
    await saveStaticRois();
  }
  renderAnnList(); draw();
}

async function saveStaticRois() {
  const res = await api("PUT", `/api/projects/${state.slug}/static_rois`,
    { rois: staticRois() });
  state.project.static_rois = res.static_rois;
}

function removeAnn(id) {
  if (staticRois().some((r) => r.id === id)) {
    state.project.static_rois = staticRois().filter((r) => r.id !== id);
    if (state.selectedAnnId === id) state.selectedAnnId = null;
    saveStaticRois().then(() => { renderAnnList(); draw(); });
    return;
  }
  if (state.currentImage) {
    state.currentImage.annotations = curAnns().filter((a) => a.id !== id);
    if (state.selectedAnnId === id) state.selectedAnnId = null;
    saveAnns();
  }
}

async function saveAnns() {
  // Capture the image we're saving: the user may switch images before the
  // PUT resolves, and we must not write this image's result onto another one.
  const target = state.currentImage;
  if (!target) return;
  const updated = await api("PUT",
    `/api/projects/${state.slug}/image/${target.id}/annotations`,
    { annotations: target.annotations });
  target.annotations = updated.annotations;
  renderImageList();
  if (state.currentImage === target) { renderAnnList(); draw(); }
}

// ── Canvas view & drawing ─────────────────────────────────────────
// The canvas backing store matches the wrap size; the image is drawn with a
// fit-to-view transform that the user can zoom (wheel / buttons) and pan
// (middle- or right-drag).
function fitView() {
  if (!state.currentImage) return;
  const wrap = $("canvasWrap");
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const iw = state.currentImage.width, ih = state.currentImage.height;
  if (cw <= 0 || ch <= 0 || iw <= 0 || ih <= 0) return;  // layout not ready
  const c = $("canvas");
  c.width = cw; c.height = ch;
  const fit = Math.min(cw / iw, ch / ih);
  state.view = { scale: fit, fit, offX: (cw - iw * fit) / 2, offY: (ch - ih * fit) / 2 };
  updateZoomLabel();
}

function clampView() {
  const c = $("canvas"), v = state.view;
  const sw = state.currentImage.width * v.scale, sh = state.currentImage.height * v.scale;
  v.offX = sw <= c.width ? (c.width - sw) / 2 : Math.min(0, Math.max(c.width - sw, v.offX));
  v.offY = sh <= c.height ? (c.height - sh) / 2 : Math.min(0, Math.max(c.height - sh, v.offY));
}

function zoomAt(sx, sy, factor) {
  const v = state.view;
  const newScale = Math.max(v.fit, Math.min(v.fit * 40, v.scale * factor));
  const ix = (sx - v.offX) / v.scale, iy = (sy - v.offY) / v.scale;
  v.scale = newScale;
  v.offX = sx - ix * newScale;
  v.offY = sy - iy * newScale;
  clampView();
  updateZoomLabel();
  draw();
}

function updateZoomLabel() {
  const lbl = $("zoomLabel");
  if (lbl) lbl.textContent = Math.round(state.view.scale / state.view.fit * 100) + "%";
}

function imgToScreen(x, y) {
  return [x * state.view.scale + state.view.offX, y * state.view.scale + state.view.offY];
}
function screenToImg(sx, sy) {
  return [(sx - state.view.offX) / state.view.scale, (sy - state.view.offY) / state.view.scale];
}
function canvasPos(e) {
  const r = $("canvas").getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function draw() {
  const c = $("canvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  $("canvasHint").classList.toggle("hidden", !!state.currentImage);
  $("zoomCtl").classList.toggle("hidden", !state.currentImage);
  if (!state.currentImage || !state.imgEl) return;
  const v = state.view;
  ctx.drawImage(state.imgEl, v.offX, v.offY,
    state.currentImage.width * v.scale, state.currentImage.height * v.scale);

  // Draw non-selected first, then the selected one ON TOP so its highlighted
  // contour is never hidden under neighbouring crystals.
  const sel = state.selectedAnnId;
  let selDraw = null;
  for (const a of visibleStaticRois()) (a.id === sel ? (selDraw = [a, true]) : drawAnn(ctx, a, true));
  for (const a of curAnns()) (a.id === sel ? (selDraw = [a, false]) : drawAnn(ctx, a, false));
  if (selDraw) drawAnn(ctx, selDraw[0], selDraw[1]);

  // box being dragged
  if (state.drag) {
    const [x0, y0] = imgToScreen(state.drag.x0, state.drag.y0);
    const [x1, y1] = imgToScreen(state.drag.x1, state.drag.y1);
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.setLineDash([]);
  }
  // polygon being drawn (manual poly tool)
  if (state.polyPoints.length) drawPolyInProgress(ctx);
  // sam points
  for (const p of state.samPoints) {
    const [sx, sy] = imgToScreen(p.x, p.y);
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = p.is_positive ? "#3cb44b" : "#e6194b";
    ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // pending prediction
  if (state.pending) drawPending(ctx, state.pending);
}

function drawAnn(ctx, a, isStatic) {
  const cls = classById(a.class_id);
  const color = cls ? cls.color : "#888";
  const selected = a.id === state.selectedAnnId;
  const suggested = a.status === "suggested";
  // For static ROIs, scale from normalised coords using the current image size.
  const disp = isStatic ? roiOnImage(a, state.currentImage) : { polygon: a.polygon, bbox: a.bbox };

  // Trace the shape into the current path (polygon or bbox).
  let anchor = null;
  function tracePath() {
    ctx.beginPath();
    if (disp.polygon && disp.polygon.length >= 3) {
      disp.polygon.forEach((p, i) => {
        const [sx, sy] = imgToScreen(p.x, p.y);
        i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
      });
      ctx.closePath();
    } else if (disp.bbox) {
      const [x, y] = imgToScreen(disp.bbox.x, disp.bbox.y);
      ctx.rect(x, y, disp.bbox.width * state.view.scale, disp.bbox.height * state.view.scale);
    }
  }
  if (disp.polygon && disp.polygon.length >= 3) anchor = imgToScreen(disp.polygon[0].x, disp.polygon[0].y);
  else if (disp.bbox) anchor = imgToScreen(disp.bbox.x, disp.bbox.y);
  else return;

  // Fill (brighter when selected so the shape stands out in a dense cluster).
  tracePath();
  ctx.fillStyle = hexA(color, selected ? 0.35 : (suggested ? 0.10 : 0.18));
  ctx.fill();

  // Base outline in the class colour.
  ctx.setLineDash(suggested ? [6, 4] : (isStatic ? [12, 5] : []));
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeStyle = color;
  tracePath(); ctx.stroke();
  ctx.setLineDash([]);

  // Selection highlight: a high-contrast dashed contour drawn over the outline
  // (dark underlay + white dashes) so the picked crystal is unmistakable even
  // among overlapping red ones.
  if (selected) {
    tracePath();
    ctx.lineWidth = 4; ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.stroke();
    tracePath();
    ctx.lineWidth = 2; ctx.strokeStyle = "#ffffff"; ctx.setLineDash([7, 5]); ctx.stroke();
    ctx.setLineDash([]);
  }

  if (isStatic && anchor) {
    ctx.font = "14px sans-serif"; ctx.textBaseline = "top";
    ctx.fillStyle = color;
    ctx.fillText("📌", anchor[0] + 2, anchor[1] + 2);
  }
}

function drawPolyInProgress(ctx) {
  const cls = classById(state.currentClassId);
  const color = cls ? cls.color : "#fff";
  const pts = state.polyPoints.map((p) => imgToScreen(p.x, p.y));
  ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.setLineDash([4, 3]);
  ctx.beginPath();
  pts.forEach(([sx, sy], i) => (i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy)));
  // rubber-band segment to the cursor
  if (state.polyCursor) {
    const [cx, cy] = imgToScreen(state.polyCursor.x, state.polyCursor.y);
    ctx.lineTo(cx, cy);
  }
  ctx.stroke(); ctx.setLineDash([]);
  // vertices
  pts.forEach(([sx, sy], i) => {
    ctx.beginPath(); ctx.arc(sx, sy, i === 0 ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#fff" : color;   // first point larger (close target)
    ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
  });
}

function drawPending(ctx, p) {
  ctx.lineWidth = 2.5; ctx.strokeStyle = "#ffffff"; ctx.setLineDash([7, 4]);
  if (p.polygon && p.polygon.length >= 3) {
    ctx.beginPath();
    p.polygon.forEach((q, i) => {
      const [sx, sy] = imgToScreen(q.x, q.y);
      i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
    });
    ctx.closePath(); ctx.fillStyle = "rgba(255,255,255,.15)"; ctx.fill(); ctx.stroke();
  } else if (p.bbox) {
    const [x, y] = imgToScreen(p.bbox.x, p.bbox.y);
    ctx.strokeRect(x, y, p.bbox.width * state.view.scale, p.bbox.height * state.view.scale);
  }
  ctx.setLineDash([]);
}

// ── Tools ─────────────────────────────────────────────────────────
function setTool(t) {
  state.tool = t;
  state.samPoints = []; state.pending = null; hidePending();
  state.polyPoints = []; state.polyCursor = null;     // reset any half-drawn polygon
  document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
  $("canvas").style.cursor = t === "select" ? "default" : "crosshair";
  draw();
}

function requireClass() {
  if (state.currentClassId === null || state.currentClassId === undefined) {
    alert("Сначала создайте и выберите класс.");
    return false;
  }
  return true;
}

function newAnn(extra) {
  return Object.assign({
    class_id: state.currentClassId, source: "manual", status: "confirmed",
    type: "rect", bbox: null, polygon: null, confidence: 0, static: false,
  }, extra);
}

// canvas mouse handlers
function onMouseDown(e) {
  if (!state.currentImage) return;
  const [sx, sy] = canvasPos(e);

  // Middle- or right-button drag = pan (works with any tool).
  if (e.button === 1 || e.button === 2) {
    e.preventDefault();
    state.pan = { sx, sy, offX: state.view.offX, offY: state.view.offY };
    $("canvas").style.cursor = "grabbing";
    return;
  }
  if (e.button !== 0) return;  // only left button drives tools

  const [ix, iy] = screenToImg(sx, sy);

  if (state.tool === "select") {
    const hit = hitTest(ix, iy);
    state.selectedAnnId = hit ? hit.id : null;
    renderAnnList(); draw(); return;
  }
  if (state.tool === "sam-point") {
    if (!requireClass()) return;
    state.samPoints.push({ x: ix, y: iy, is_positive: !e.shiftKey });
    draw(); runSamPoints(); return;
  }
  if (state.tool === "poly") {
    if (!requireClass()) return;
    state.polyPoints.push({ x: Math.round(ix), y: Math.round(iy) });
    draw(); return;
  }
  if (state.tool === "box" || state.tool === "sam-box") {
    if (!requireClass()) return;
    state.drag = { x0: ix, y0: iy, x1: ix, y1: iy };
  }
}

function onMouseMove(e) {
  if (state.pan) {
    const [sx, sy] = canvasPos(e);
    state.view.offX = state.pan.offX + (sx - state.pan.sx);
    state.view.offY = state.pan.offY + (sy - state.pan.sy);
    clampView(); draw();
    return;
  }
  if (state.tool === "poly" && state.polyPoints.length) {
    const [sx, sy] = canvasPos(e);
    const [ix, iy] = screenToImg(sx, sy);
    state.polyCursor = { x: ix, y: iy };
    draw();
    return;
  }
  if (!state.drag) return;
  const [sx, sy] = canvasPos(e);
  [state.drag.x1, state.drag.y1] = screenToImg(sx, sy);
  draw();
}

async function onMouseUp() {
  if (state.pan) {
    state.pan = null;
    $("canvas").style.cursor = state.tool === "select" ? "default" : "crosshair";
    return;
  }
  if (!state.drag) return;
  const d = state.drag; state.drag = null;
  const bbox = normRect(d);
  if (bbox.width < 3 || bbox.height < 3) { draw(); return; }
  if (state.tool === "box") {
    state.currentImage.annotations.push(newAnn({ bbox }));
    await saveAnns();
  } else if (state.tool === "sam-box") {
    await runSamBox(bbox);
  }
}

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

function hitTest(ix, iy) {
  // topmost first: per-image annotations over visible static ROIs
  const items = curAnns().map((a) => [a, a.polygon, a.bbox])
    .concat(visibleStaticRois().map((a) => {
      const d = roiOnImage(a, state.currentImage);
      return [a, d.polygon, d.bbox];
    }));
  for (let i = items.length - 1; i >= 0; i--) {
    const [a, poly, bbox] = items[i];
    if (poly && poly.length >= 3) {
      if (pointInPolygon(ix, iy, poly)) return a;
    } else if (bbox && ix >= bbox.x && ix <= bbox.x + bbox.width &&
        iy >= bbox.y && iy <= bbox.y + bbox.height) return a;
  }
  return null;
}

// ── Manual polygon tool ───────────────────────────────────────────
function bboxOfPolygon(pts) {
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x = Math.min(...xs), y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

function closePolygon() {
  let pts = state.polyPoints.slice();
  // A double-click adds a near-duplicate final point — drop trailing duplicates.
  while (pts.length >= 2) {
    const a = pts[pts.length - 1], b = pts[pts.length - 2];
    if (Math.abs(a.x - b.x) <= 2 && Math.abs(a.y - b.y) <= 2) pts.pop();
    else break;
  }
  state.polyPoints = []; state.polyCursor = null;
  if (pts.length < 3) { draw(); return; }   // need a real polygon
  state.currentImage.annotations.push(newAnn({
    type: "poly", polygon: pts, bbox: bboxOfPolygon(pts),
  }));
  saveAnns();
}

function cancelPolygon() {
  state.polyPoints = []; state.polyCursor = null; draw();
}

// ── SAM 2 interactive ─────────────────────────────────────────────
async function runSamPoints() {
  try {
    const res = await runJob(`/api/projects/${state.slug}/sam2/points`,
      { image_id: state.currentImage.id, points: state.samPoints }, "SAM 2…");
    showPredictionAsPending(res.predictions);
  } catch (err) { if (!(err instanceof JobCancelled)) alert("SAM 2: " + err.message); }
}

async function runSamBox(bbox) {
  try {
    const res = await runJob(`/api/projects/${state.slug}/sam2/box`,
      { image_id: state.currentImage.id, box: bbox }, "SAM 2…");
    showPredictionAsPending(res.predictions);
  } catch (err) { if (!(err instanceof JobCancelled)) alert("SAM 2: " + err.message); }
}

function showPredictionAsPending(preds) {
  if (!preds || !preds.length) { state.pending = null; hidePending(); draw(); return; }
  const best = preds[0];
  state.pending = { bbox: best.bbox, polygon: best.polygon, confidence: best.confidence, source: "sam2" };
  $("pendingMsg").textContent = `Маска готова (уверенность ${(best.confidence * 100).toFixed(0)}%). Принять?`;
  $("pendingLoading").classList.add("hidden");
  $("pendingConfirm").classList.remove("hidden");
  $("pendingBar").classList.remove("hidden");
  draw();
}

async function acceptPending() {
  if (!state.pending) return;
  const p = state.pending;
  state.currentImage.annotations.push(newAnn({
    type: p.polygon && p.polygon.length >= 3 ? "poly" : "rect",
    bbox: p.bbox, polygon: p.polygon, source: p.source || "sam2",
    confidence: p.confidence || 0, status: "confirmed",
  }));
  state.pending = null; state.samPoints = []; hidePending();
  await saveAnns();
}

function cancelPending() {
  state.pending = null; state.samPoints = []; hidePending(); draw();
}

function hidePending() {
  $("pendingBar").classList.add("hidden");
  $("pendingLoading").classList.add("hidden");
  $("pendingConfirm").classList.add("hidden");
}

// ── SAM 3 auto on current image ───────────────────────────────────
async function runSam3Auto() {
  if (!state.currentImage) return;
  if (!requireClass()) return;
  const cls = classById(state.currentClassId);
  const prompt = ($("sam3Prompt").value.trim() || (cls && cls.name) || "").trim();
  if (!prompt) { alert("Введите запрос для SAM 3 или выберите класс."); return; }
  state.lastPrompt = prompt;
  const sahi = $("samSahi").checked;
  const body = {
    image_id: state.currentImage.id, class_id: state.currentClassId,
    text_prompt: prompt, confidence: state.project.settings.sam3_confidence ?? 0.5,
    sahi,
    slice_size: Number($("samSlice").value) || 512,
    overlap: Number($("samOverlap").value) || 0.2,
    iou: Number($("samIou").value) || 0.45,
    drop_edge: $("samDropEdge") ? $("samDropEdge").checked : true,
  };
  const label = sahi
    ? `SAM 3 (SAHI): прогон фото и тайлов…`
    : `SAM 3: поиск «${prompt}»…`;
  try {
    const res = await runJob(`/api/projects/${state.slug}/sam3/auto`, body, label);
    const note = res.passes > 1
      ? `${res.passes} проходов (1 основной + ${res.passes - 1} тайлов), сырых ${res.raw}` +
        `${res.edge_dropped ? `, откинуто на швах ${res.edge_dropped}` : ""}` +
        ` → после дедупликации ${res.predictions.length}, за ${res.elapsed} с`
      : `Фото размечено за ${res.elapsed} с — найдено: ${res.predictions.length}`;
    showSam3Time(note);
    if (!res.predictions.length) { alert("SAM 3 ничего не нашёл."); return; }
    for (const p of res.predictions) state.currentImage.annotations.push(p);
    await saveAnns();
  } catch (err) { if (!(err instanceof JobCancelled)) alert("SAM 3: " + err.message); }
}

// Forecast of SAHI passes for the current image (matches the backend tiling).
function sahiTileCount(w, h, slice, overlap) {
  slice = Math.max(64, slice || 512);
  const step = Math.max(1, Math.floor(slice * (1 - Math.min(0.9, Math.max(0, overlap)))));
  const nx = Math.ceil(Math.max(1, w - 1) / step);
  const ny = Math.ceil(Math.max(1, h - 1) / step);
  return nx * ny;
}
// Highlight the toolbar SAHI button when slicing is active.
function syncSahiToggle() {
  const btn = $("sahiToggleBtn");
  if (btn) btn.classList.toggle("active", $("samSahi").checked);
}

function updateSahiForecast() {
  const el = $("samSahiForecast");
  if (!el) return;
  const img = state.currentImage;
  if (!img) { el.textContent = "Откройте изображение для расчёта проходов."; return; }
  const tiles = sahiTileCount(img.width, img.height,
    Number($("samSlice").value) || 512, Number($("samOverlap").value) || 0.2);
  el.textContent = `Изображение ${img.width}×${img.height}. Проходов: 1 основной + ${tiles} тайлов = ${tiles + 1}.`;
}

// SAM 3 stopwatch readout (shown in the SAM 3 panel + console).
function showSam3Time(text) {
  const el = $("sam3Timer");
  if (el) el.textContent = "⏱ SAM 3: " + text;
  console.log("[SAM3]", text);
}

async function confirmAllSuggestions() {
  curAnns().forEach((a) => { if (a.status === "suggested") a.status = "confirmed"; });
  await saveAnns();
}
async function rejectAllSuggestions() {
  state.currentImage.annotations = curAnns().filter((a) => a.status !== "suggested");
  await saveAnns();
}

// Delete every annotation of the CURRENT image (static ROIs are project-wide и
// сюда не входят — их снимают галочкой «статичная»).
async function clearAnns() {
  if (!state.currentImage) return;
  const n = curAnns().length;
  if (!n) { alert("На этом кадре нет аннотаций."); return; }
  if (!confirm(`Удалить все аннотации этого кадра (${n})? Статичные рамки не затрагиваются.`)) return;
  state.currentImage.annotations = [];
  state.selectedAnnId = null;
  await saveAnns();
}

// ── Propagation across images ─────────────────────────────────────
async function propagate() {
  if (!state.currentImage) return;
  const classId = Number($("propClass").value);
  if (Number.isNaN(classId)) { alert("Сначала создайте и выберите класс."); return; }
  const cls = classById(classId);
  const prompt = ($("propPrompt").value.trim() || (cls && cls.name) || "").trim();
  if (!prompt) { alert("Укажите запрос SAM 3 или имя класса."); return; }
  const scope = $("propScope").value;
  const where = scope === "following" ? "следующих за текущим" : "всех";
  if (!confirm(`Найти «${prompt}» на ${where} изображениях и предложить как класс «${cls ? cls.name : classId}»?`)) return;
  try {
    const res = await runJob(`/api/projects/${state.slug}/sam3/propagate`, {
      from_image_id: state.currentImage.id, class_id: classId,
      text_prompt: prompt, confidence: state.project.settings.sam3_confidence ?? 0.5,
      scope,
    }, "SAM 3: авторазметка изображений…");
    await refreshProject();
    showSam3Time(`${res.processed} фото за ${res.elapsed} с (~${res.avg} с/фото)`);
    const note = res.cancelled ? " (отменено)" : "";
    alert(`Готово${note}. Предложено аннотаций: ${res.total_added} на ${res.images} изображениях.\n` +
          `Время: ${res.elapsed} с (~${res.avg} с/фото).\n` +
          `Пролистайте изображения и подтвердите/исправьте.`);
  } catch (err) { if (!(err instanceof JobCancelled)) alert("Авторазметка: " + err.message); }
}

// ── Navigation ────────────────────────────────────────────────────
function navigate(delta) {
  const imgs = state.project.images;
  if (!imgs.length || !state.currentImage) return;
  const idx = imgs.findIndex((i) => i.id === state.currentImage.id);
  const next = imgs[idx + delta];
  if (next) selectImage(next);
}

// ── Import / upload ───────────────────────────────────────────────
async function importFolder() {
  const folder = prompt("Путь к папке с изображениями:");
  if (!folder) return;
  try {
    busy(true, "Импорт изображений…");
    const res = await api("POST", `/api/projects/${state.slug}/import_folder`, { folder });
    await refreshProject();
    if (!state.currentImage && state.project.images.length) selectImage(state.project.images[0]);
    alert(`Добавлено изображений: ${res.added}`);
  } catch (err) { alert("Импорт: " + err.message); }
  finally { busy(false); }
}

async function importDataset() {
  const folder = prompt(
    "Путь к папке экспортированного датасета (YOLO или RF-DETR/COCO).\n" +
    "Формат определяется автоматически (data.yaml → YOLO, _annotations.coco.json → COCO).\n" +
    "Импортируются все части (train/val/test) как подтверждённая разметка:");
  if (!folder) return;
  try {
    busy(true, "Импорт разметки…");
    const res = await api("POST", `/api/projects/${state.slug}/import_dataset`,
      { folder, format: "auto", copy: true });
    await refreshProject();
    if (!state.currentImage && state.project.images.length) selectImage(state.project.images[0]);
    alert(`Импортировано (${(res.format || "").toUpperCase()}): изображений ${res.images}, ` +
      `аннотаций ${res.annotations}, новых классов ${res.classes}.`);
  } catch (err) { alert("Импорт разметки: " + err.message); }
  finally { busy(false); }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  busy(true, "Загрузка файлов…");
  try {
    const res = await fetch(`/api/projects/${state.slug}/upload`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const data = await res.json();
    await refreshProject();
    if (!state.currentImage && state.project.images.length) selectImage(state.project.images[0]);
    alert(`Загружено: ${data.added}`);
  } catch (err) { alert("Загрузка: " + err.message); }
  finally { busy(false); }
}

// ── Export ────────────────────────────────────────────────────────
function openExport(target) {
  state.exportTarget = target || "yolo";
  const isCoco = state.exportTarget === "coco";
  const s = state.project.settings;
  $("exportTitle").textContent = isCoco
    ? "Экспорт в RF-DETR (COCO JSON, сегментация)"
    : "Экспорт в формат YOLO";
  $("formatRow").classList.toggle("hidden", isCoco);   // YOLO-only label
  $("cocoNote").classList.toggle("hidden", !isCoco);
  $("exportFormat").value = s.export_format || "detect";
  $("valSplit").value = s.val_split ?? 0.1;
  $("augEnabled").checked = !!(s.augment && s.augment.enabled);
  $("exportResult").textContent = "";
  $("exportDialog").classList.remove("hidden");
}

async function doExport() {
  const out_dir = $("exportPath").value.trim();
  if (!out_dir) { alert("Укажите папку для экспорта."); return; }
  const angles = [...document.querySelectorAll(".angle:checked")].map((c) => Number(c.value));
  const body = {
    out_dir,
    target: state.exportTarget || "yolo",
    fmt: $("exportFormat").value,
    val_split: Number($("valSplit").value) || 0.1,
    augment: $("augEnabled").checked,
    angles,
    include_suggested: $("inclSuggested").checked,
  };
  try {
    busy(true, "Экспорт датасета…");
    const r = await api("POST", `/api/projects/${state.slug}/export`, body);
    const layout = r.target === "coco"
      ? "train/ и valid/ (+ _annotations.coco.json)"
      : "data.yaml создан";
    $("exportResult").textContent =
      `Готово → ${r.out_dir}\n` +
      `train: ${r.train} (из них аугментаций ${r.augmented}), val: ${r.val}\n` +
      `объектов: ${r.instances}, классов: ${r.classes}, формат: ${r.format}\n` +
      `${layout}.`;
    // persist export prefs (YOLO label format only)
    await api("PATCH", `/api/projects/${state.slug}/settings`, { patch: {
      export_format: body.fmt, val_split: body.val_split,
      augment: { enabled: body.augment, angles },
    }});
  } catch (err) { $("exportResult").textContent = "Ошибка: " + err.message; }
  finally { busy(false); }
}

// ── Helpers ───────────────────────────────────────────────────────
function normRect(d) {
  const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
  return { x: Math.round(x), y: Math.round(y),
           width: Math.round(Math.abs(d.x1 - d.x0)), height: Math.round(Math.abs(d.y1 - d.y0)) };
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function checkHealth() {
  try {
    const h = await api("GET", "/api/health");
    const el = $("samStatus");
    if (h.status === "error") { el.textContent = "AI: недоступен"; el.className = "sam-status err"; }
    else { el.textContent = `AI: ${h.device || "готов"}`; el.className = "sam-status ok"; }
  } catch { $("samStatus").textContent = "AI: ?"; }
}

// ── Wire up ───────────────────────────────────────────────────────
function init() {
  // Surface errors instead of letting them silently corrupt UI state.
  window.addEventListener("error", (e) => console.error("QuickLabel error:", e.message));
  window.addEventListener("unhandledrejection", (e) => console.error("QuickLabel async error:", e.reason));

  $("projectSelect").onchange = (e) => openProject(e.target.value);
  $("newProjectBtn").onclick = async () => {
    const name = prompt("Название проекта:");
    if (!name) return;
    const p = await api("POST", "/api/projects", { name });
    await loadProjects(p.slug);
  };
  $("delProjectBtn").onclick = deleteProject;
  $("importFolderBtn").onclick = importFolder;
  $("importDatasetBtn").onclick = importDataset;
  $("uploadBtn").onclick = () => $("uploadInput").click();
  $("uploadInput").onchange = (e) => uploadFiles([...e.target.files]);
  $("addClassBtn").onclick = addClass;
  $("newClassName").onkeydown = (e) => { if (e.key === "Enter") addClass(); };

  document.querySelectorAll(".tool").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
  $("sam3AutoBtn").onclick = runSam3Auto;
  $("sam3Prompt").onkeydown = (e) => { if (e.key === "Enter") runSam3Auto(); };
  $("samSahi").onchange = () => { syncSahiToggle(); updateSahiForecast(); };
  $("samSlice").oninput = updateSahiForecast;
  $("samOverlap").oninput = updateSahiForecast;
  $("sahiToggleBtn").onclick = (e) => {
    e.stopPropagation();
    const pop = $("sahiPopover");
    const willOpen = pop.classList.contains("hidden");
    pop.classList.toggle("hidden");
    if (willOpen) {
      // Fixed-position below the button (a clipping ancestor uses overflow:hidden).
      const r = $("sahiToggleBtn").getBoundingClientRect();
      pop.style.top = (r.bottom + 6) + "px";
      pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - 266)) + "px";
      updateSahiForecast();
    }
  };
  $("sahiPopover").onclick = (e) => e.stopPropagation();
  $("propagateBtn").onclick = propagate;
  $("propClass").onchange = (e) => {
    const cls = classById(Number(e.target.value));
    if (cls) $("propPrompt").value = cls.name;
  };
  $("pendingCancel").onclick = cancelCurrentJob;   // cancel during loading
  $("confirmAllBtn").onclick = confirmAllSuggestions;
  $("rejectAllBtn").onclick = rejectAllSuggestions;
  $("clearAnnsBtn").onclick = clearAnns;
  $("prevImg").onclick = () => navigate(-1);
  $("nextImg").onclick = () => navigate(1);

  $("acceptPending").onclick = acceptPending;
  $("cancelPending").onclick = cancelPending;

  // Export split-button: click toggles a target menu (YOLO / RF-DETR COCO).
  $("exportBtn").onclick = (e) => {
    e.stopPropagation();
    $("exportDropdown").classList.toggle("hidden");
  };
  $("exportDropdown").querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      $("exportDropdown").classList.add("hidden");
      openExport(b.dataset.target);
    };
  });
  document.addEventListener("click", () => {
    $("exportDropdown").classList.add("hidden");
    $("sahiPopover").classList.add("hidden");
  });
  $("doExportBtn").onclick = doExport;
  $("closeExportBtn").onclick = () => $("exportDialog").classList.add("hidden");

  // Training page
  $("trainBtn").onclick = openTrainView;
  $("trainBack").onclick = closeTrainView;
  $("backFromProgress").onclick = closeTrainView;
  $("newTrainBtn").onclick = showTrainSetup;
  $("startTrainBtn").onclick = startTraining;
  $("stopTrainBtn").onclick = stopTraining;
  document.querySelectorAll('input[name="framework"]').forEach((r) => { r.onchange = onFrameworkChange; });
  document.querySelectorAll('input[name="taskType"]').forEach((r) => {
    r.onchange = () => { renderModelCards(); };
  });
  $("valPct").oninput = updateSplitBar;
  $("testPct").oninput = updateSplitBar;
  $("tileEnabled").onchange = (e) => $("tileOpts").classList.toggle("hidden", !e.target.checked);

  // Test / inference dialog
  $("runTestBtn").onclick = runTest;
  $("closeTestBtn").onclick = () => $("testDialog").classList.add("hidden");
  $("runValBtn").onclick = runValidation;
  $("closeValBtn").onclick = () => $("valDialog").classList.add("hidden");
  $("testSahi").onchange = (e) => $("sahiRow").classList.toggle("hidden", !e.target.checked);

  const c = $("canvas");
  c.addEventListener("mousedown", onMouseDown);
  c.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  c.addEventListener("dblclick", (e) => {     // close polygon on double-click
    if (state.tool === "poly" && state.polyPoints.length) { e.preventDefault(); closePolygon(); }
  });
  c.addEventListener("contextmenu", (e) => e.preventDefault());  // right-drag pans
  c.addEventListener("wheel", (e) => {
    if (!state.currentImage) return;
    e.preventDefault();
    const [sx, sy] = canvasPos(e);
    zoomAt(sx, sy, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }, { passive: false });

  // Zoom control buttons (zoom toward canvas centre).
  $("zoomIn").onclick = () => zoomAt(c.width / 2, c.height / 2, 1.25);
  $("zoomOut").onclick = () => zoomAt(c.width / 2, c.height / 2, 1 / 1.25);
  $("zoomFit").onclick = () => { fitView(); draw(); };

  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "Enter" && state.polyPoints.length) closePolygon();
    else if (e.key === "Enter" && state.pending) acceptPending();
    else if (e.key === "Escape" && state.polyPoints.length) cancelPolygon();
    else if (e.key === "Escape") cancelPending();
    else if ((e.key === "Delete" || e.key === "Backspace") && state.selectedAnnId) {
      removeAnn(state.selectedAnnId);
    } else if (e.key === "ArrowRight") navigate(1);
    else if (e.key === "ArrowLeft") navigate(-1);
    else if (e.key === "1") setTool("select");
    else if (e.key === "2") setTool("box");
    else if (e.key === "3") setTool("poly");
    else if (e.key === "4") setTool("sam-point");
    else if (e.key === "5") setTool("sam-box");
    else if (state.currentImage && (e.key === "+" || e.key === "="))
      zoomAt($("canvas").width / 2, $("canvas").height / 2, 1.25);
    else if (state.currentImage && (e.key === "-" || e.key === "_"))
      zoomAt($("canvas").width / 2, $("canvas").height / 2, 1 / 1.25);
    else if (state.currentImage && e.key === "0") { fitView(); draw(); }
  });
  window.addEventListener("resize", () => { if (state.currentImage) { fitView(); draw(); } });

  setTool("select");
  loadProjects();
  checkHealth();
}

// ══════════════ Training page ══════════════
// Model line-up per framework. RF-DETR sizes share a backbone (see
// training_service); YOLO sizes differ. `imgsz` holds the default resolution
// per task so the form pre-fills a sensible value.
const MODEL_SPECS = {
  rfdetr: [
    { id: "RF-DETR-N", name: "RF-DETR-N", tags: ["Быстрая", "30M"], imgsz: { object_detection: 560, instance_segmentation: 312 } },
    { id: "RF-DETR-S", name: "RF-DETR-S", tags: ["Средняя", "32M"], imgsz: { object_detection: 560, instance_segmentation: 384 } },
    { id: "RF-DETR-M", name: "RF-DETR-M", tags: ["Медленнее", "34M"], imgsz: { object_detection: 560, instance_segmentation: 432 } },
    { id: "RF-DETR-L", name: "RF-DETR-L", tags: ["Большая", "34M"], imgsz: { object_detection: 704, instance_segmentation: 504 } },
    { id: "RF-DETR-XL", name: "RF-DETR-XL", tags: ["Очень большая", "126M", "rfdetr[plus]"], imgsz: { object_detection: 700, instance_segmentation: 700 } },
    { id: "RF-DETR-2XL", name: "RF-DETR-2XL", tags: ["Максимум", "127M", "rfdetr[plus]"], imgsz: { object_detection: 880, instance_segmentation: 880 } },
  ],
  yolo: [
    { id: "YOLO11n", name: "YOLO11n", tags: ["v11", "Быстрая"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO11s", name: "YOLO11s", tags: ["v11", "Средняя"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO11m", name: "YOLO11m", tags: ["v11", "Лучше"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO11l", name: "YOLO11l", tags: ["v11", "Большая"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO11x", name: "YOLO11x", tags: ["v11", "Максимум"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO26n", name: "YOLO26n", tags: ["v26", "Быстрая"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO26s", name: "YOLO26s", tags: ["v26", "Средняя"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO26m", name: "YOLO26m", tags: ["v26", "Лучше"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO26l", name: "YOLO26l", tags: ["v26", "Большая"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
    { id: "YOLO26x", name: "YOLO26x", tags: ["v26", "Максимум"], imgsz: { object_detection: 640, instance_segmentation: 640 } },
  ],
};
const FW_DEFAULTS = {
  rfdetr: { lr: 0.0001, batch: 4, epochs: 50 },
  yolo: { lr: 0.01, batch: 16, epochs: 100 },
};
const TRAIN_DONE = ["completed", "stopped", "error"];

function curFramework() { return document.querySelector('input[name="framework"]:checked').value; }
function curTask() { return document.querySelector('input[name="taskType"]:checked').value; }

async function openTrainView() {
  if (!state.slug) return;
  $("trainView").classList.remove("hidden");
  $("trainSubtitle").textContent =
    `Проект: ${state.project.name} · изображений: ${state.project.images.length} · классов: ${state.project.classes.length}`;
  await checkTrainDeps();
  renderModelCards();
  updateSplitBar();
  await loadTrainedModels();
  // Reopened while a run is live → jump straight to the dashboard.
  let st = null;
  try { st = await api("GET", "/api/train/status"); } catch {}
  if (st && st.running) { showTrainProgress(); startTrainPolling(); }
  else showTrainSetup();
}

function closeTrainView() { stopTrainPolling(); $("trainView").classList.add("hidden"); }
function showTrainSetup() {
  stopTrainPolling();
  $("startTrainBtn").disabled = !fwInstalled(curFramework());
  $("trainProgress").classList.add("hidden");
  $("trainSetup").classList.remove("hidden");
}
function showTrainProgress() {
  $("trainSetup").classList.add("hidden");
  $("trainProgress").classList.remove("hidden");
}

async function checkTrainDeps() {
  try {
    const d = await api("GET", "/api/train/check");
    state.depInfo = d;
    $("trainDevice").textContent = "устройство: " +
      (d.device === "cuda" ? (d.device_name || "GPU") : d.device);
    $("trainDevice").className = "sam-status " + (d.cuda ? "ok" : "");
    $("fwRfdetr").textContent = d.rfdetr ? "установлен" : "не установлен";
    $("fwRfdetr").className = d.rfdetr ? "ok-text" : "muted";
    $("fwYolo").textContent = d.ultralytics ? "установлен" : "не установлен";
    $("fwYolo").className = d.ultralytics ? "ok-text" : "muted";
    updateDepWarn();
  } catch (e) { console.error("train check", e); }
}

function fwInstalled(fw) {
  const d = state.depInfo || {};
  return fw === "yolo" ? !!d.ultralytics : !!d.rfdetr;
}

function updateDepWarn() {
  const fw = curFramework();
  const warn = $("depWarn");
  if (!state.depInfo || fwInstalled(fw)) {
    warn.classList.add("hidden"); $("startTrainBtn").disabled = false; return;
  }
  const pkg = fw === "yolo" ? "ultralytics" : "rfdetr==1.5.2";
  warn.textContent = `Пакет «${pkg}» не установлен — обучение для этого фреймворка недоступно.\n` +
    `Выполните в папке QuickLabel:  .\\setup.ps1 -WithTraining  (нужен интернет)\n` +
    `или вручную:  .\\.venv\\Scripts\\python.exe -m pip install ${pkg}`;
  warn.classList.remove("hidden");
  $("startTrainBtn").disabled = true;
}

function onFrameworkChange() {
  const def = FW_DEFAULTS[curFramework()];
  $("hpLr").value = def.lr;
  $("hpBatch").value = def.batch;
  $("hpEpochs").value = def.epochs;
  renderModelCards();
  updateDepWarn();
}

function renderModelCards() {
  const fw = curFramework(), task = curTask();
  const specs = MODEL_SPECS[fw];
  if (!specs.some((s) => s.id === state.trainModelId)) {
    state.trainModelId = specs[Math.min(1, specs.length - 1)].id;   // default S/s
  }
  const wrap = $("modelCards");
  wrap.innerHTML = "";
  specs.forEach((s) => {
    const div = document.createElement("div");
    div.className = "model-card" + (s.id === state.trainModelId ? " active" : "");
    let label = s.name;
    if (task === "instance_segmentation") {
      label = fw === "rfdetr" ? s.name.replace("RF-DETR", "RF-DETR-Seg") : s.name + "-seg";
    }
    div.innerHTML = `<div class="mc-name">${label}</div><div class="mc-tags">` +
      s.tags.map((t) => `<span class="mc-tag">${t}</span>`).join("") + "</div>";
    div.onclick = () => { state.trainModelId = s.id; renderModelCards(); };
    wrap.appendChild(div);
  });
  updateImgszDefault();
}

// Suggested batch size that fits an ~8 GB GPU, by model size (bigger → smaller batch).
function suggestedBatch(fw, id) {
  if (fw === "yolo") {
    if (/x$/i.test(id)) return 2;
    if (/l$/i.test(id)) return 4;
    if (/m$/i.test(id)) return 8;
    return 16;                       // n, s
  }
  const size = String(id).split("-").pop().toUpperCase();   // RF-DETR
  return { N: 4, S: 4, M: 4, L: 2, XL: 1, "2XL": 1 }[size] || 4;
}

function updateImgszDefault() {
  const fw = curFramework();
  const spec = MODEL_SPECS[fw].find((s) => s.id === state.trainModelId);
  if (spec) $("hpImgsz").value = spec.imgsz[curTask()] || spec.imgsz.object_detection;
  // Pre-fill a VRAM-safe batch for the chosen model size (user can still edit).
  $("hpBatch").value = suggestedBatch(fw, state.trainModelId);
}

function clampNum(v, lo, hi, dflt) {
  let n = Number(v); if (Number.isNaN(n)) n = dflt;
  return Math.min(hi, Math.max(lo, n));
}

function updateSplitBar() {
  let val = clampNum($("valPct").value, 0, 60, 10);
  let test = clampNum($("testPct").value, 0, 40, 0);
  if (val + test > 90) { test = Math.max(0, 90 - val); $("testPct").value = test; }
  const train = Math.max(0, 100 - val - test);
  $("segTrain").style.width = train + "%";
  $("segVal").style.width = val + "%";
  $("segTest").style.width = test + "%";
  $("trainPctLbl").textContent = train + "%";
  $("valPctLbl").textContent = val + "%";
  $("testPctLbl").textContent = test + "%";
}

async function startTraining() {
  const fw = curFramework();
  if (!fwInstalled(fw)) { alert("Фреймворк не установлен. Запустите setup.ps1."); return; }
  const augRot = $("augRot").checked, augFlip = $("augFlip").checked;
  const augBright = $("augBright").checked, augGray = $("augGray").checked;
  const body = {
    framework: fw,
    model_name: state.trainModelId,
    task_type: curTask(),
    epochs: Number($("hpEpochs").value) || 50,
    batch_size: Number($("hpBatch").value) || 4,
    image_size: Number($("hpImgsz").value) || 0,
    learning_rate: Number($("hpLr").value) || 0.0001,
    patience: Math.max(0, Number($("hpPatience").value) || 0),
    warmup_epochs: Math.max(0, Number($("hpWarmup").value) || 0),
    weight_decay: Math.max(0, Number($("hpWeightDecay").value) || 0),
    val_split: clampNum($("valPct").value, 0, 60, 10) / 100,
    test_split: clampNum($("testPct").value, 0, 40, 0) / 100,
    augment: augRot || augFlip || augBright || augGray,
    angles: augRot ? [90, 180, 270] : [],
    flip_h: augFlip,
    brightness: augBright,
    grayscale: augGray,
    tile: $("tileEnabled").checked,
    tile_size: clampNum($("tileSize").value, 128, 2048, 640),
    tile_overlap: clampNum($("tileOverlap").value, 0, 80, 20) / 100,
    tile_max_images: Math.max(0, Number($("tileMaxImages").value) || 0),
    tile_empty_ratio: clampNum($("tileEmpty").value, 0, 100, 15) / 100,
    include_suggested: $("trainIncl").checked,
  };
  if (body.epochs < 10 && !confirm(
      `Очень мало эпох (${body.epochs}). Модель почти не обучится: у YOLO первые ~3 эпохи — это только «разогрев» (warmup), ` +
      `и результат будет находить ~ноль объектов. Рекомендуется 50–100 эпох (и больше изображений). Всё равно запустить?`)) {
    return;
  }
  try {
    $("startTrainBtn").disabled = true;
    await api("POST", `/api/projects/${state.slug}/train`, body);
    $("stopTrainBtn").classList.remove("hidden");
    $("stopTrainBtn").disabled = false;
    $("newTrainBtn").classList.add("hidden");
    showTrainProgress();
    startTrainPolling();
  } catch (err) {
    alert("Не удалось запустить обучение: " + err.message);
    $("startTrainBtn").disabled = false;
  }
}

function startTrainPolling() {
  stopTrainPolling();
  state.trainPoll = setInterval(refreshTrainStatus, 1000);
  refreshTrainStatus();
}
function stopTrainPolling() {
  if (state.trainPoll) { clearInterval(state.trainPoll); state.trainPoll = null; }
}

async function refreshTrainStatus() {
  let st;
  try { st = await api("GET", "/api/train/status"); } catch { return; }
  renderTrainStatus(st);
  if (TRAIN_DONE.includes(st.status)) {
    stopTrainPolling();
    $("stopTrainBtn").classList.add("hidden");
    $("newTrainBtn").classList.remove("hidden");
    loadTrainedModels();
  }
}

function renderTrainStatus(st) {
  const labels = { preparing: "Подготовка", training: "Обучение", evaluating: "Валидация",
                   completed: "Завершено ✓", stopped: "Остановлено", error: "Ошибка", idle: "—" };
  const badge = $("progStatusBadge");
  badge.textContent = labels[st.status] || st.status || "—";
  badge.className = "status-badge " + (st.status || "");
  $("progModel").textContent =
    `${(st.framework || "").toUpperCase()} · ${st.model_name || ""} · ` +
    `${st.task === "instance_segmentation" ? "сегментация" : "детекция"}`;

  const pct = Math.max(0, Math.min(100, st.percentage || 0));
  $("progPct").textContent = pct.toFixed(1) + "%";
  $("progBar").style.width = pct + "%";
  $("progEpoch").textContent = `${st.current_epoch || 0} / ${st.total_epochs || 0}`;

  const ti = st.total_iterations || 0, ci = st.current_iteration || 0;
  $("iterBar").style.width = (ti ? Math.min(100, ci / ti * 100) : 0) + "%";
  if (st.status === "error") $("iterLbl").textContent = "⚠ " + (st.error || st.message || "Ошибка");
  else $("iterLbl").textContent = ti ? `Итерация ${ci} / ${ti}` : (st.message || "");

  // Show where the trained checkpoint was written (once available).
  const bar = $("modelPathBar");
  if (st.model_path) {
    $("modelPathText").textContent = st.model_path;
    $("copyModelPath").onclick = () => copyPath(st.model_path, $("copyModelPath"));
    bar.classList.remove("hidden");
  } else {
    bar.classList.add("hidden");
  }

  renderMetricCards(st);
  drawMetricChart(st.history || []);
  renderTrainLog(st.log || []);
}

async function copyPath(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    if (btn) { const old = btn.textContent; btn.textContent = "✓ Скопировано"; setTimeout(() => { btn.textContent = old; }, 1500); }
  } catch {
    // Clipboard API may be blocked on http; fall back to a prompt for manual copy.
    window.prompt("Путь к модели (Ctrl+C для копирования):", text);
  }
}

function metricCard(label, value, unit) {
  return `<div class="metric-card"><div class="mlabel">${label}</div>` +
    `<div class="mval">${value}<span class="munit">${unit ? " " + unit : ""}</span></div></div>`;
}
function renderMetricCards(st) {
  const num = (v, d = 1) => (v != null && !Number.isNaN(Number(v))) ? Number(v).toFixed(d) : "—";
  const cards = [
    metricCard("Прошло", fmtDur(st.elapsed_seconds), ""),
    metricCard("Осталось (ETA)", st.eta_seconds != null ? fmtDur(st.eta_seconds) : "—", ""),
    metricCard("Loss", num(st.loss, 3), ""),
    metricCard("mAP@50:95", num(st.map_50_95), "%"),
    metricCard("Best AP50", st.best_map_50 ? num(st.best_map_50) : "—", "%"),
    metricCard("Throughput", st.throughput ? num(st.throughput) : "—", "img/s"),
    metricCard("Learning rate", st.learning_rate != null ? fmtLr(st.learning_rate) : "—", ""),
  ];
  if (st.precision != null) cards.push(metricCard("Precision", num(st.precision), "%"));
  if (st.recall != null) cards.push(metricCard("Recall", num(st.recall), "%"));
  $("metricCards").innerHTML = cards.join("");
}
function fmtLr(v) { v = Number(v); return v && v < 0.001 ? v.toExponential(1) : v.toFixed(4); }
function fmtDur(s) {
  if (s == null) return "—";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}ч ${m}м`;
  if (m) return `${m}м ${sec}с`;
  return `${sec}с`;
}

function renderTrainLog(lines) {
  const el = $("trainLog");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  el.textContent = lines.join("\n");
  if (atBottom) el.scrollTop = el.scrollHeight;
}

// Vanilla-canvas line chart of metrics over epochs (no chart library).
// Left axis = mAP (0..100%), right axis = train loss (0..max). Three series:
// loss (orange), mAP@50 (blue), mAP@50:95 (green).
function drawMetricChart(history) {
  const cv = $("metricChart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || 800, cssH = 240;
  if (cv.width !== Math.round(cssW * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);
  const padL = 36, padR = 46, padT = 12, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const grid = "#3a3d4d", muted = "#9aa0b5";

  ctx.strokeStyle = grid; ctx.lineWidth = 1;
  ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = muted; ctx.fillText(String(100 - i * 25), padL - 4, y + 3);
  }
  if (!history.length) {
    ctx.fillStyle = muted; ctx.textAlign = "center";
    ctx.fillText("Метрики появятся после первой эпохи", W / 2, H / 2);
    return;
  }
  const n = history.length;
  const xs = (i) => padL + (n === 1 ? plotW / 2 : plotW * i / (n - 1));
  const losses = history.map((h) => h.loss).filter((v) => v != null);
  const maxLoss = losses.length ? Math.max(...losses) * 1.1 || 1 : 1;

  ctx.fillStyle = "#ff8a5c"; ctx.textAlign = "left";
  for (let i = 0; i <= 4; i++) {
    const y = padT + plotH * i / 4;
    ctx.fillText((maxLoss * (1 - i / 4)).toFixed(2), W - padR + 4, y + 3);
  }

  function series(key, color, scale) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    history.forEach((h, i) => {
      if (h[key] == null) return;
      const y = padT + plotH * (1 - scale(h[key]));
      if (!started) { ctx.moveTo(xs(i), y); started = true; } else ctx.lineTo(xs(i), y);
    });
    ctx.stroke();
    ctx.fillStyle = color;
    history.forEach((h, i) => {
      if (h[key] == null) return;
      const y = padT + plotH * (1 - scale(h[key]));
      ctx.beginPath(); ctx.arc(xs(i), y, 2.5, 0, Math.PI * 2); ctx.fill();
    });
  }
  const pctScale = (v) => Math.max(0, Math.min(1, v / 100));
  series("map_50", "#4f8cff", pctScale);
  series("map_50_95", "#3cb44b", pctScale);
  series("loss", "#ff8a5c", (v) => Math.max(0, Math.min(1, v / maxLoss)));

  ctx.fillStyle = muted; ctx.textAlign = "center";
  const step = Math.max(1, Math.ceil(n / 8));
  history.forEach((h, i) => {
    if (i % step === 0 || i === n - 1) ctx.fillText("эп." + h.epoch, xs(i), H - 7);
  });
}

async function loadTrainedModels() {
  try {
    const r = await api("GET", `/api/projects/${state.slug}/trained_models`);
    const list = r.models || [];
    state.trainedModels = list;
    const ul = $("trainedList"); ul.innerHTML = "";
    $("trainedEmpty").classList.toggle("hidden", list.length > 0);
    list.forEach((m) => {
      const li = document.createElement("li");
      li.className = "tm-item";
      li.title = "Показать метрики этой модели";
      const map = m.best_map_50 != null ? `mAP@50 ${Number(m.best_map_50).toFixed(1)}%` : "";
      li.innerHTML =
        `<button class="row-del tm-del" title="Удалить">✕</button>` +
        `<div class="tm-name">${escapeHtml(m.model_name || "?")}</div>` +
        `<div class="tm-meta">${(m.framework || "").toUpperCase()} · ` +
        `${m.task === "instance_segmentation" ? "seg" : "det"} · эпох: ${m.epochs || "?"}</div>` +
        (map ? `<div class="tm-meta">${map}</div>` : "") +
        `<span class="tm-badge ${m.status}">${m.status}</span>` +
        (m.model_path
          ? `<div class="tm-path" title="${escapeHtml(m.model_path)}">${escapeHtml(m.model_path)}</div>` +
            `<div class="tm-btns"><button class="tm-test">🔍 Проверить</button>` +
            `<button class="tm-val" title="Прогнать модель по её валидационной выборке">🖼 Валидация</button>` +
            `<button class="tm-copy" title="Копировать путь к модели">📋 путь</button></div>`
          : "");
      // Clicking the card (but not its buttons) shows this model's own metrics.
      li.onclick = () => showStoredModelMetrics(m);
      const del = li.querySelector(".tm-del");
      del.onclick = (e) => { e.stopPropagation(); deleteTrainedModel(m); };
      const copyBtn = li.querySelector(".tm-copy");
      if (copyBtn) copyBtn.onclick = (e) => { e.stopPropagation(); copyPath(m.model_path, copyBtn); };
      const testBtn = li.querySelector(".tm-test");
      if (testBtn) testBtn.onclick = (e) => { e.stopPropagation(); openTestDialog(m.run_id); };
      const valBtn = li.querySelector(".tm-val");
      if (valBtn) valBtn.onclick = (e) => { e.stopPropagation(); openValidationDialog(m); };
      ul.appendChild(li);
    });
  } catch (e) { console.error("trained models", e); }
}

// Render a finished model's stored metrics into the dashboard (read-only).
// Lets the user inspect ANY trained model, not just the most recent run.
function showStoredModelMetrics(m) {
  if (state.trainPoll) {
    alert("Идёт обучение — дождитесь его завершения, чтобы посмотреть метрики другой модели.");
    return;
  }
  const mt = m.metrics || {};
  const st = {
    status: m.status,
    framework: m.framework,
    model_name: m.model_name,
    task: m.task,
    percentage: 100,
    current_epoch: m.epochs,
    total_epochs: m.total_epochs || m.epochs,
    loss: mt.loss,
    map_50_95: mt.map_50_95 != null ? mt.map_50_95 : m.map_50_95,
    best_map_50: mt.best_map_50 != null ? mt.best_map_50 : m.best_map_50,
    precision: mt.precision,
    recall: mt.recall,
    learning_rate: mt.learning_rate,
    throughput: mt.throughput,
    elapsed_seconds: mt.elapsed_seconds,
    eta_seconds: null,
    model_path: m.model_path,
    history: m.history || [],
    log: m.log || [],
    error: m.error,
    message: m.message || "Сохранённые метрики обученной модели",
  };
  showTrainProgress();
  renderTrainStatus(st);
  // This is a past run, not a live one: no Stop, allow starting a new run.
  $("stopTrainBtn").classList.add("hidden");
  $("newTrainBtn").classList.remove("hidden");
  // Highlight the selected card.
  document.querySelectorAll("#trainedList .tm-item.active")
    .forEach((el) => el.classList.remove("active"));
  const idx = (state.trainedModels || []).indexOf(m);
  const li = $("trainedList").children[idx];
  if (li) li.classList.add("active");
}

async function deleteTrainedModel(m) {
  if (!confirm(`Удалить обученную модель «${m.model_name}» и её файлы?`)) return;
  try {
    await api("DELETE", `/api/projects/${state.slug}/trained_models/${m.run_id}`);
    await loadTrainedModels();
  } catch (err) { alert("Не удалось удалить: " + err.message); }
}

async function stopTraining() {
  if (!confirm("Остановить обучение? Текущая эпоха завершится, лучший чекпойнт сохранится.")) return;
  $("stopTrainBtn").disabled = true;
  try { await api("POST", "/api/train/stop"); } catch {}
  setTimeout(() => { $("stopTrainBtn").disabled = false; }, 2000);
}

// ── Test / inference dialog ───────────────────────────────────────
function openTestDialog(runId) {
  const models = (state.trainedModels || []).filter((m) => m.model_path);
  if (!models.length) { alert("Нет обученной модели с сохранённым файлом."); return; }
  $("testModel").innerHTML = models.map((m) =>
    `<option value="${m.run_id}">${escapeHtml(m.model_name || "?")} · ` +
    `${(m.framework || "").toUpperCase()} · ${m.task === "instance_segmentation" ? "seg" : "det"}</option>`).join("");
  if (runId) $("testModel").value = runId;
  // Project images (the dropdown sets image_id; empty = use the path field).
  const imgs = (state.project && state.project.images) || [];
  $("testImage").innerHTML = `<option value="">— выбрать изображение проекта —</option>` +
    imgs.map((im) => `<option value="${im.id}">${escapeHtml(im.filename)}</option>`).join("");
  if (state.currentImage) $("testImage").value = state.currentImage.id;
  $("testStatus").textContent = "";
  $("testResult").classList.add("hidden");
  $("testDialog").classList.remove("hidden");
}

async function runTest() {
  const run_id = $("testModel").value;
  if (!run_id) { alert("Выберите модель."); return; }
  const image_id = $("testImage").value || null;
  const image_path = $("testPath").value.trim() || null;
  if (!image_id && !image_path) { alert("Выберите изображение проекта или укажите путь к файлу."); return; }
  const body = {
    run_id, image_id: image_id || undefined, image_path: image_path || undefined,
    confidence: Number($("testConf").value) || 0.25,
    sahi: $("testSahi").checked,
    slice_size: Number($("testSlice").value) || 640,
    overlap: Number($("testOverlap").value) || 0.2,
    iou: Number($("testIou").value) || 0.45,
    drop_edge: $("testDropEdge").checked,
  };
  const btn = $("runTestBtn");
  btn.disabled = true;
  $("testStatus").textContent = body.sahi
    ? "Распознавание с нарезкой SAHI… (может занять время на больших фото)"
    : "Распознавание…";
  $("testResult").classList.add("hidden");
  try {
    const r = await api("POST", `/api/projects/${state.slug}/predict`, body);
    const per = r.per_class || {};
    const parts = Object.keys(per).map((k) => `${escapeHtml(k)}: ${per[k]}`).join(" · ");
    $("testCounts").textContent =
      `Найдено объектов: ${r.count}${parts ? " (" + parts + ")" : ""}` +
      `${r.sahi ? ` · SAHI, тайлов: ${r.tiles}` : ""}` +
      `${r.sahi && r.edge_dropped ? ` · отброшено на швах: ${r.edge_dropped}` : ""}` +
      ` · ${r.width}×${r.height}`;
    $("testImg").src = "data:image/jpeg;base64," + r.image_b64;
    $("testResult").classList.remove("hidden");
    $("testStatus").textContent = "";
  } catch (err) {
    $("testStatus").textContent = "Ошибка: " + err.message;
  } finally {
    btn.disabled = false;
  }
}

// ── Validation gallery (run a finished model on its own val set) ──
function openValidationDialog(m) {
  state.valRunId = m.run_id;
  $("valGrid").innerHTML = "";
  $("valStatus").textContent =
    `Модель: ${m.model_name || "?"} · ${(m.framework || "").toUpperCase()} · ` +
    `${m.task === "instance_segmentation" ? "seg" : "det"}. Нажмите «Проверить».`;
  $("valDialog").classList.remove("hidden");
}

async function runValidation() {
  const run_id = state.valRunId;
  if (!run_id) return;
  const body = {
    run_id,
    confidence: Number($("valConf").value) || 0.25,
    limit: Math.max(1, Math.min(48, Number($("valLimit").value) || 12)),
    sahi: $("valSahi").checked,
  };
  const btn = $("runValBtn");
  btn.disabled = true;
  $("valStatus").textContent = "Прогон по валидации… (модель загружается один раз, затем все фото)";
  $("valGrid").innerHTML = "";
  try {
    const r = await api("POST", `/api/projects/${state.slug}/trained_models/${run_id}/validate`, body);
    const results = r.results || [];
    $("valStatus").textContent =
      `Показано ${r.shown} из ${r.total_val} валидационных фото · ` +
      `всего найдено объектов: ${results.reduce((s, x) => s + (x.count || 0), 0)}`;
    $("valGrid").innerHTML = results.map((x) => {
      if (x.error) {
        return `<div class="val-cell"><div class="val-meta">${escapeHtml(x.name || "")}: ${escapeHtml(x.error)}</div></div>`;
      }
      const per = x.per_class || {};
      const parts = Object.keys(per).map((k) => `${escapeHtml(k)}: ${per[k]}`).join(", ");
      return `<div class="val-cell">` +
        `<img src="data:image/jpeg;base64,${x.image_b64}" alt="${escapeHtml(x.name || "")}" />` +
        `<div class="val-meta">${escapeHtml(x.name || "")} — объектов: ${x.count || 0}` +
        `${parts ? ` (${parts})` : ""}</div></div>`;
    }).join("");
  } catch (err) {
    $("valStatus").textContent = "Ошибка: " + err.message;
  } finally {
    btn.disabled = false;
  }
}

init();
