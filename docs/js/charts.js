// Small hand-written SVG charts. Every chart is drawn at its container's pixel width so labels
// stay at text size on a phone, and redrawn when that width changes. Colours come from CSS
// classes (site.css), never from here.

const NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, parent = null, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  if (text !== null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

export function linear([d0, d1], [r0, r1]) {
  const f = (x) => r0 + ((x - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.domain = [d0, d1];
  return f;
}

export function sqrtScale([d0, d1], [r0, r1]) {
  const s0 = Math.sqrt(d0);
  const s1 = Math.sqrt(d1);
  return (x) => r0 + ((Math.sqrt(Math.max(x, 0)) - s0) / (s1 - s0 || 1)) * (r1 - r0);
}

function frame(container, width, height, label) {
  container.replaceChildren();
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
    "aria-label": label,
  });
  container.appendChild(svg);
  return svg;
}

/** Draw now and again whenever the container's width changes. */
export function responsive(container, draw) {
  let last = 0;
  const run = () => {
    const w = Math.round(container.clientWidth);
    if (w && w !== last) {
      last = w;
      draw(w);
    }
  };
  new ResizeObserver(run).observe(container);
  run();
  return () => {
    last = 0;
    run();
  };
}

const narrow = (w) => w < 520;

/**
 * Dot strip: one row per model, a dot per window with its CI, the mean as a tick.
 * rows: [{group, label, points: [{name, value, lo, hi, hollow}], mean, chance}]
 */
export function dotStrip(container, rows, { label, fmt }) {
  return responsive(container, (width) => {
    const small = narrow(width);
    const labelW = small ? 112 : 196;
    const valueW = small ? 44 : 56;
    const rowH = small ? 28 : 36;
    const groupH = small ? 26 : 32;
    const top = 8;
    const axisH = 28;
    let y = top;
    const layout = rows.map((r, i) => {
      if (r.group && (i === 0 || rows[i - 1].group !== r.group)) y += groupH;
      const at = y + rowH / 2;
      y += rowH;
      return { ...r, at };
    });
    const height = y + axisH;
    const x = linear([0, 1], [labelW, width - valueW - 8]);
    const svg = frame(container, width, height, label);
    const ax = el("g", { class: "ax" }, svg);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      el("line", { x1: x(t), x2: x(t), y1: top, y2: y }, ax);
      el("text", { x: x(t), y: y + 18, "text-anchor": "middle" }, ax, t === 0 ? "0" : String(t));
    }
    let prevGroup = null;
    for (const r of layout) {
      if (r.group && r.group !== prevGroup) {
        el("text", { x: 0, y: r.at - rowH / 2 - 8, class: "lbl lbl--group" }, svg, small && r.groupShort ? r.groupShort : r.group);
        prevGroup = r.group;
      }
      el("text", { x: 0, y: r.at + 4, class: "lbl" }, svg, small && r.short ? r.short : r.label);
      if (r.chance !== undefined) {
        const cx = x(r.chance);
        el("line", { x1: cx, x2: cx, y1: r.at - 8, y2: r.at + 8, class: "chance" }, svg);
      }
      for (const p of r.points ?? []) {
        if (p.lo !== undefined) {
          el("line", { x1: x(p.lo), x2: x(p.hi), y1: r.at, y2: r.at, class: "ci s-ink" }, svg);
        }
        el("circle", { cx: x(p.value), cy: r.at, r: 4, class: p.hollow ? "hollow s-ink" : "s-ink", "stroke-width": 1.5 }, svg);
      }
      if (r.mean !== undefined) {
        const mx = x(r.mean);
        if (r.chance === undefined) el("line", { x1: mx, x2: mx, y1: r.at - 10, y2: r.at + 10, class: "ci s-cool" }, svg);
        el("text", { x: width, y: r.at + 4, "text-anchor": "end", class: "lbl lbl--num" }, svg, fmt(r.mean));
      }
    }
  });
}

/**
 * Grouped CIs per window: bands along x, a few series inside each band.
 * bands: [{name, sub: [lines], shade, chance}], series: [{key, cls, hollow}],
 * values[band][series] = {value, lo, hi}
 */
export function bandCI(container, { bands, series, values, label, shadeLabel, subPrefix = [], subShort = [], fmt }) {
  return responsive(container, (width) => {
    const small = narrow(width);
    const left = 36;
    const top = 28;
    const plotH = small ? 220 : 300;
    const subLines = bands[0].sub.length;
    const bottom = 22 + subLines * 16;
    const height = top + plotH + bottom;
    const bw = (width - left) / bands.length;
    const y = linear([0, 1], [top + plotH, top]);
    const svg = frame(container, width, height, label);
    bands.forEach((b, i) => {
      if (b.shade) el("rect", { x: left + i * bw, y: 0, width: bw, height: top + plotH, class: "band" }, svg);
    });
    const firstShade = bands.findIndex((b) => b.shade);
    if (firstShade >= 0 && shadeLabel) {
      el("text", { x: left + firstShade * bw + 8, y: 16, class: "lbl lbl--ink" }, svg, shadeLabel);
    }
    const ax = el("g", { class: "ax" }, svg);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      el("line", { x1: left, x2: width, y1: y(t), y2: y(t) }, ax);
      el("text", { x: left - 6, y: y(t) + 4, "text-anchor": "end" }, ax, t === 0 ? "0" : String(t));
    }
    bands.forEach((b, i) => {
      const x0 = left + i * bw;
      const pad = bw * 0.18;
      const step = (bw - 2 * pad) / Math.max(series.length - 1, 1);
      if (b.chance !== undefined) {
        el("line", { x1: x0 + pad - 8, x2: x0 + bw - pad + 8, y1: y(b.chance), y2: y(b.chance), class: "chance" }, svg);
      }
      series.forEach((s, j) => {
        const v = values[i][j];
        if (!v) return;
        const cx = x0 + pad + j * step;
        el("line", { x1: cx, x2: cx, y1: y(v.lo), y2: y(v.hi), class: `ci ${s.cls}` }, svg);
        el("circle", { cx, cy: y(v.value), r: small ? 3.5 : 4.5, class: s.hollow ? `hollow ${s.cls}` : s.cls, "stroke-width": 1.5 }, svg);
      });
      const cx = x0 + bw / 2;
      el("text", { x: cx, y: top + plotH + 18, "text-anchor": "middle", class: "lbl lbl--ink" }, svg, b.name);
      b.sub.forEach((line, k) => {
        const text = small ? line : `${subPrefix[k] ?? ""}${line}`;
        el("text", { x: cx, y: top + plotH + 34 + k * 16, "text-anchor": "middle", class: "lbl" }, svg, text);
      });
    });
    if (small) {
      subShort.forEach((p, k) => {
        if (p) el("text", { x: 0, y: top + plotH + 34 + k * 16, class: "lbl" }, svg, p);
      });
    }
    void fmt;
  });
}

/**
 * Lines over an index axis (months). series: [{values, cls, area}] with null gaps;
 * xLabels: [{i, text}]; events: [{i, text, cls, row}]; shades: [{from, to, text}].
 */
export function lines(container, { n, series, yMax, yTicks, yFmt, xLabels, events = [], shades = [], label, height: h }) {
  return responsive(container, (width) => {
    const small = narrow(width);
    const left = small ? 44 : 56;
    const right = 8;
    const top = events.length || shades.length ? 44 : 12;
    const plotH = h ?? (small ? 200 : 260);
    const height = top + plotH + 26;
    const x = linear([0, n - 1], [left, width - right]);
    const y = linear([0, yMax], [top + plotH, top]);
    const svg = frame(container, width, height, label);
    for (const s of shades) {
      el("rect", { x: x(s.from), y: top, width: Math.max(x(s.to) - x(s.from), 1), height: plotH, class: "band" }, svg);
      if (s.text) el("text", { x: x(s.from) + 4, y: top + 14, class: "lbl" }, svg, s.text);
    }
    const ax = el("g", { class: "ax" }, svg);
    for (const t of yTicks) {
      el("line", { x1: left, x2: width - right, y1: y(t), y2: y(t) }, ax);
      el("text", { x: left - 6, y: y(t) + 4, "text-anchor": "end" }, ax, yFmt(t));
    }
    const every = small ? Math.ceil(xLabels.length / 6) : 1;
    xLabels.forEach((l, k) => {
      if (k % every) return;
      el("text", { x: x(l.i), y: top + plotH + 18, "text-anchor": "middle" }, ax, l.text);
    });
    for (const s of series) {
      const segs = [];
      let cur = [];
      s.values.forEach((v, i) => {
        if (v === null || v === undefined) {
          if (cur.length) segs.push(cur);
          cur = [];
        } else cur.push([x(i), y(v)]);
      });
      if (cur.length) segs.push(cur);
      for (const seg of segs) {
        const d = seg.map(([a, b], k) => `${k ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join("");
        if (s.area) {
          const base = y(0).toFixed(1);
          el("path", { d: `${d}L${seg.at(-1)[0].toFixed(1)},${base}L${seg[0][0].toFixed(1)},${base}Z`, class: "area" }, svg);
        }
        el("path", { d, class: `ln ${s.cls}` }, svg);
      }
    }
    for (const e of events) {
      const ex = x(e.i);
      el("line", { x1: ex, x2: ex, y1: top - 6 - (e.row ?? 0) * 16, y2: top + plotH, class: `event ${e.cls ?? ""}` }, svg);
      const anchor = ex > width * 0.75 ? "end" : "start";
      const text = small && e.short ? e.short : e.text;
      if (text) el("text", { x: anchor === "end" ? ex - 4 : ex + 4, y: top - 8 - (e.row ?? 0) * 16, "text-anchor": anchor, class: "lbl lbl--ink" }, svg, text);
    }
  });
}

/** Reliability: square, square-root axes, diagonal. series: [{points: [[x, y]], cls}] */
export function reliability(container, { series, label }) {
  return responsive(container, (width) => {
    const size = Math.min(width, 560);
    const left = 44;
    const bottom = 32;
    const top = 8;
    const right = 8;
    const plot = size - left - right;
    const height = top + plot + bottom;
    const x = sqrtScale([0, 1], [left, left + plot]);
    const y = sqrtScale([0, 1], [top + plot, top]);
    const svg = frame(container, size, height, label);
    const ax = el("g", { class: "ax" }, svg);
    for (const t of [0, 0.01, 0.05, 0.25, 0.5, 1]) {
      el("line", { x1: x(t), x2: x(t), y1: top, y2: top + plot }, ax);
      el("line", { x1: left, x2: left + plot, y1: y(t), y2: y(t) }, ax);
      const txt = t === 0 ? "0" : String(t);
      el("text", { x: x(t), y: top + plot + 18, "text-anchor": "middle" }, ax, txt);
      el("text", { x: left - 6, y: y(t) + 4, "text-anchor": "end" }, ax, txt);
    }
    el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: "diag" }, svg);
    for (const s of series) {
      const d = s.points.map(([a, b], k) => `${k ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join("");
      el("path", { d, class: `ln ${s.cls}` }, svg);
      for (const [a, b] of s.points) el("circle", { cx: x(a), cy: y(b), r: 3.5, class: s.cls }, svg);
    }
  });
}
