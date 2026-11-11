// Fills the page from docs/data/*.json. Every number shown is read here at load time.
import { dotStrip, bandCI, lines, reliability } from "./charts.js";
import { initScorer } from "./try.js";
import { initThreshold } from "./threshold.js";
import { f3, pct, int, monthName, MONTHS, h } from "./util.js";

const FILES = ["metrics", "curves", "calibration", "timeline", "windows", "robustness", "scorer", "manifest", "v1"];

const quarter = (date) => {
  const [y, m] = date.split("-").map(Number);
  return `Q${Math.floor((m - 1) / 3) + 1} ${y}`;
};
const listJoin = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs.at(-1)}`);

function setK(key, text) {
  for (const node of document.querySelectorAll(`[data-k="${key}"]`)) node.textContent = text;
}
function setText(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}
function table(head, rows, { numFrom = 1, caption = null, rowClass = () => null } = {}) {
  return h(
    "table",
    { class: "data" },
    caption ? h("caption", {}, caption) : null,
    h("thead", {}, h("tr", {}, head.map((c, i) => h("th", { scope: "col", class: i >= numFrom ? "num" : null }, c)))),
    h(
      "tbody",
      {},
      rows.map((r, ri) =>
        h(
          "tr",
          { class: rowClass(ri) },
          r.map((c, i) => (i === 0 ? h("th", { scope: "row" }, c) : h("td", { class: i >= numFrom ? "num" : null }, c))),
        ),
      ),
    ),
  );
}
const ciText = (point, ci) => `${f3(point)} (${f3(ci[0])} to ${f3(ci[1])})`;

function cellIndex(metrics) {
  const map = new Map();
  for (const c of metrics.cells) map.set(`${c.model}__${c.featureset}__${c.window}`, c);
  return map;
}

// --- 1. the answer ----------------------------------------------------------------------

function answer(d, cells) {
  const { metrics, windows } = d;
  const low = new Set(windows.windows.filter((w) => w.low_narrative_share).map((w) => w.window));
  const aw = metrics.across_windows;
  setK("struct", f3(aw.lr__struct.pr_auc.mean));
  setK("both", f3(aw.lr__both.pr_auc.mean));
  const fs = metrics.featureset_labels;
  const models = [
    ["lr", metrics.model_labels.lr],
    ["lgbm", metrics.model_labels.lgbm],
  ];
  const rows = [];
  for (const [m, name] of models) {
    for (const s of ["text", "struct", "both"]) {
      rows.push({
        group: name,
        label: fs[s],
        model: m,
        set: s,
        points: metrics.windows.map((w) => {
          const c = cells.get(`${m}__${s}__${w}`);
          return { name: w, value: c.point.pr_auc, lo: c.ci95.pr_auc[0], hi: c.ci95.pr_auc[1], hollow: low.has(w) };
        }),
        mean: aw[`${m}__${s}`].pr_auc.mean,
      });
    }
  }
  const base = metrics.model_labels.company_rate;
  rows.push({
    group: "Baselines",
    label: base,
    model: "company_rate",
    set: "struct",
    points: metrics.windows.map((w) => {
      const c = cells.get(`company_rate__struct__${w}`);
      return { name: w, value: c.point.pr_auc, lo: c.ci95.pr_auc[0], hi: c.ci95.pr_auc[1], hollow: low.has(w) };
    }),
    mean: aw.company_rate__struct.pr_auc.mean,
  });
  rows.push({
    group: "Baselines",
    label: "Chance (test late rate)",
    points: [],
    chance: aw.company_rate__struct.prevalence.mean,
    mean: aw.company_rate__struct.prevalence.mean,
  });
  for (const r of rows) {
    r.full = r.label;
    r.label = r.label.replace("Narrative and structured fields", "Both").replace("Structured fields only", "Structured").replace("Narrative only", "Narrative").replace("Company late rate alone", "Company rate").replace("Chance (test late rate)", "Chance");
  }
  dotStrip(document.getElementById("grid-chart"), rows, {
    label: `Test PR-AUC by model and features across ${metrics.windows.length} windows. Mean for logistic regression: narrative only ${f3(aw.lr__text.pr_auc.mean)}, structured only ${f3(aw.lr__struct.pr_auc.mean)}, both ${f3(aw.lr__both.pr_auc.mean)}.`,
    fmt: f3,
  });

  // Paired CIs on adding the narrative to the structured fields.
  const parts = models.map(([m, name]) => {
    const comps = metrics.comparisons.filter((c) => c.a[0] === m && c.a[1] === "both" && c.b[0] === m && c.b[1] === "struct");
    const below = comps.filter((c) => c.metrics.pr_auc.ci95[1] < 0).length;
    const above = comps.filter((c) => c.metrics.pr_auc.ci95[0] > 0).length;
    return `${name} below zero in ${below}, above in ${above}`;
  });
  const lowNames = [...low];
  setText(
    "grid-note",
    `Hollow dots: ${listJoin(lowNames)}, tested in 2026. Blue tick: the mean.`,
  );

  const head = ["Model and features", ...metrics.windows, "Mean"];
  const body = rows
    .filter((r) => r.points.length)
    .map((r) => [
      `${r.group === "Baselines" ? "" : `${r.group}, `}${r.full}`,
      ...metrics.windows.map((w) => {
        const c = cells.get(`${r.model}__${r.set}__${w}`);
        return ciText(c.point.pr_auc, c.ci95.pr_auc);
      }),
      f3(r.mean),
    ]);
  body.push(["Chance (test late rate)", ...windows.windows.map((w) => f3(w.test.prevalence)), f3(aw.company_rate__struct.prevalence.mean)]);
  document.getElementById("grid-table").append(
    table(head, body, {
      caption: `Paired 95% CI of adding the narrative to the structured fields: ${parts.join("; ")}, of ${metrics.windows.length} windows.`,
    }),
  );
}

// --- 4. over time -----------------------------------------------------------------------

function overTime(d, cells) {
  const { metrics, windows, robustness } = d;
  const ws = windows.windows;
  const low = ws.filter((w) => w.low_narrative_share);
  setText(
    "time-lead",
    `Each window trains on a year and is tested once on a later quarter. ${listJoin(low.map((w) => w.window))} test on 2026.`,
  );
  const series = [
    { key: "lr__text", cls: "s-muted", label: metrics.featureset_labels.text },
    { key: "lr__struct", cls: "s-ink", label: metrics.featureset_labels.struct },
    { key: "lr__both", cls: "s-cool", label: metrics.featureset_labels.both },
    { key: "company_rate__struct", cls: "s-ink", hollow: true, label: metrics.model_labels.company_rate },
  ];
  const bands = ws.map((w) => ({
    name: w.window,
    sub: [quarter(w.dates.test[0]).replace(" 20", " \u2019"), pct(w.test.prevalence), pct(w.narrative_share.test, 0)],
    shade: w.low_narrative_share,
    chance: w.test.prevalence,
  }));
  const values = ws.map((w) =>
    series.map((s) => {
      const c = cells.get(`${s.key}__${w.window}`);
      return c ? { value: c.point.pr_auc, lo: c.ci95.pr_auc[0], hi: c.ci95.pr_auc[1] } : null;
    }),
  );
  const chart = document.getElementById("time-chart");
  bandCI(chart, {
    bands,
    series,
    values,
    label: `Test PR-AUC per window for logistic regression; ${listJoin(low.map((w) => w.window))} are shaded as the low narrative share period.`,
    shadeLabel: "Few narratives",
    subPrefix: ["", "late ", "narrative "],
    subShort: ["", "late", "narr."],
    fmt: f3,
  });
  chart.after(
    h(
      "p",
      { class: "keys" },
      series.map((s) => h("span", {}, h("span", { class: `swatch ${s.cls}` }), s.label)),
    ),
  );
  const head = ["Window", "Test period", "Test rows", "Late (chance)", "Narrative share", ...series.map((s) => s.label)];
  const body = ws.map((w, i) => [
    w.window + (w.low_narrative_share ? " (2026)" : ""),
    `${w.dates.test[0]} to ${w.dates.test[1]}`,
    int(w.test.rows),
    pct(w.test.prevalence),
    pct(w.narrative_share.test),
    ...values[i].map((v) => (v ? ciText(v.value, [v.lo, v.hi]) : "n/a")),
  ]);
  document.getElementById("time-table").append(table(head, body, { numFrom: 2, rowClass: (i) => (ws[i].low_narrative_share ? "flag" : null) }));

  // Drift by test month.
  const months = [];
  const shades = [];
  for (const w of ws) {
    const rows = robustness.drift.windows[w.window];
    const from = months.length;
    for (const r of rows) months.push({ ...r, window: w.window });
    if (w.low_narrative_share) shades.push({ from: from - 0.5, to: months.length - 0.5 });
  }
  const merged = shades.length ? [{ from: Math.max(shades[0].from, 0), to: Math.min(shades.at(-1).to, months.length - 1), text: "Few narratives" }] : [];
  const yMax = Math.max(...months.map((m) => Math.max(m.prevalence, m["lr__both:mean_score"]))) * 1.15;
  const step = yMax > 0.04 ? 0.01 : 0.005;
  const ticks = [];
  for (let t = 0; t <= yMax; t += step) ticks.push(+t.toFixed(4));
  const driftChart = document.getElementById("drift-chart");
  lines(driftChart, {
    n: months.length,
    series: [
      { values: months.map((m) => m.prevalence), cls: "s-ink" },
      { values: months.map((m) => m["lr__both:mean_score"]), cls: "s-cool" },
    ],
    yMax,
    yTicks: ticks,
    yFmt: (t) => pct(t, t < 0.01 && t > 0 ? 1 : 0),
    xLabels: months.map((m, i) => ({ i, text: MONTHS[+m.month.slice(5) - 1] })),
    shades: merged,
    label: "Late rate and mean model score per test month; the two move together, including the jump in 2026.",
    height: 200,
  });
  driftChart.after(
    h("p", { class: "keys" }, h("span", {}, h("span", { class: "swatch s-ink" }), "Late rate"), h("span", {}, h("span", { class: "swatch s-cool" }), "Mean score")),
  );
  document.getElementById("drift-table").append(
    table(
      ["Month", "Window", "Test rows", "Late rate", "Mean score", "PR-AUC"],
      months.map((m) => [monthName(m.month), m.window, int(m.rows), pct(m.prevalence, 2), pct(m["lr__both:mean_score"], 2), f3(m["lr__both:pr_auc"])]),
      { numFrom: 2 },
    ),
  );
}

// --- 5. calibration -----------------------------------------------------------------------

function calibrationSection(d) {
  const { calibration, metrics } = d;
  const order = [
    ["lr_balanced", "s-muted"],
    ["lr_balanced_platt", "s-cool"],
    ["lr_balanced_isotonic", "s-ink"],
    ["lr", "s-accent"],
  ];
  const select = document.getElementById("c-window");
  for (const w of metrics.windows) select.append(h("option", { value: w }, w));
  select.value = metrics.windows.at(-1);
  const chart = document.getElementById("cal-chart");
  const draw = () => {
    const w = select.value;
    const cellsFor = new Map(calibration.cells.filter((c) => c.window === w && c.featureset === "both").map((c) => [c.model, c]));
    const shown = order.filter(([m]) => cellsFor.has(m) && m !== "lr_balanced_isotonic");
    chart.replaceChildren();
    reliability(chart, {
      series: shown.map(([m, cls]) => ({ points: cellsFor.get(m).reliability.map(([x, y]) => [x, y]), cls })),
      label: `Reliability for window ${w}: the class-weighted model sits far right of the diagonal, the recalibrated one on it.`,
    });
    const stats = document.getElementById("cal-stats");
    stats.replaceChildren(
      ...table(
        ["Model, both feature sets", "ECE", "Brier", "Log loss"],
        order
          .filter(([m]) => cellsFor.has(m))
          .map(([m, cls]) => {
            const c = cellsFor.get(m);
            return [h("span", {}, h("span", { class: `swatch ${cls}` }), metrics.model_labels[m]), f3(c.ece.point), f3(c.brier.point), f3(c.log_loss.point)];
          }),
        { caption: `Window ${w}, test split` },
      ).childNodes,
    );
    const binTable = document.getElementById("cal-table");
    binTable.replaceChildren(
      table(
        ["Model", "Bin", "Mean score", "Observed late rate", "Rows"],
        shown.flatMap(([m]) => cellsFor.get(m).reliability.map(([x, y, n], i) => [metrics.model_labels[m], String(i + 1), x.toFixed(4), y.toFixed(4), int(n)])),
        { numFrom: 1 },
      ),
    );
  };
  select.addEventListener("change", draw);
  draw();
}

// --- 6. across products -------------------------------------------------------------------

const SHORT = {
  "Checking or savings account": "Checking or savings",
  "Credit reporting or other personal consumer reports": "Credit reporting",
  "Money transfer, virtual currency, or money service": "Money transfer",
  "Payday loan, title loan, personal loan, or advance loan": "Payday or personal loan",
};

function products(d) {
  const cp = d.robustness.cross_product;
  const names = cp.products.map((p) => SHORT[p] ?? p);
  const seg = document.getElementById("p-set");
  const labels = d.metrics.featureset_labels;
  for (const s of Object.keys(cp.featuresets)) {
    seg.append(
      h("input", { type: "radio", name: "pset", id: `p-${s}`, value: s, checked: s === "both" }),
      h("label", { for: `p-${s}` }, labels[s] ?? s),
    );
  }
  const heat = document.getElementById("p-heat");
  const draw = () => {
    const s = seg.querySelector("input:checked").value;
    const fs = cp.featuresets[s];
    heat.replaceChildren(
      h("caption", { class: "visually-hidden" }, `Test PR-AUC, ${labels[s]}, rows trained on, columns tested on`),
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Trained on, tested on"), names.map((n) => h("th", { scope: "col", class: "num" }, n)))),
      h(
        "tbody",
        {},
        names.map((n, i) =>
          h(
            "tr",
            {},
            h("th", { scope: "row" }, n),
            fs.pr_auc[i].map((v, j) =>
              h(
                "td",
                { style: `--v:${v.toFixed(3)}`, class: i === j ? "diag-cell" : null },
                f3(v),
              ),
            ),
          ),
        ),
      ),
      h(
        "tfoot",
        {},
        h("tr", {}, h("th", { scope: "row" }, `Pooled ${cp.window} model`), fs.pooled.map((v) => h("td", { style: "--v:0" }, f3(v)))),
        h("tr", {}, h("th", { scope: "row" }, "Chance (late rate)"), fs.chance.map((v) => h("td", { style: "--v:0" }, f3(v)))),
      ),
    );
    document.getElementById("p-ci").replaceChildren(
      table(
        ["Trained on", ...names],
        names.map((n, i) => [n, ...fs.pr_auc[i].map((v, j) => ciText(v, fs.pr_auc_ci95[i][j]))]),
        { caption: `${labels[s]}: test PR-AUC with 95% CI, rows trained on, columns tested on.` },
      ),
    );
  };
  seg.addEventListener("change", draw);
  draw();
  setText(
    "p-note",
    `${cp.window} test, logistic regression. Outlined cells are in-domain.`,
  );
}

// --- 7. timeline ----------------------------------------------------------------------------

function timelineSection(d) {
  const t = d.timeline;
  // The month of the live download is still arriving; leave it out of the charts.
  const live = d.manifest.files.find((f) => !f.name.startsWith("CCDB_Export_"));
  const months = t.months.filter((m) => m.month < live.downloaded.slice(0, 7));
  const idx = (ym) => months.findIndex((m) => m.month === ym);
  const end = t.data_end.slice(0, 7);
  const from = t.received_from.slice(0, 7);
  const lastNarr = [...months].reverse().find((m) => m.narratives > 0).month;
  const prevMonth = (ym) => {
    const [y, m] = ym.split("-").map(Number);
    return m === 1 ? `${y - 1}-12` : `${y}-${String(m - 1).padStart(2, "0")}`;
  };
  const disputesEnd = prevMonth(from);
  const yearAgo = `${+end.slice(0, 4) - 1}${end.slice(4)}`;
  const a = months[idx(yearAgo)];
  const b = months[idx(end)];
  setText(
    "data-lead",
    `Narratives covered ${pct(a.narrative_share, 0)} of complaints in ${monthName(yearAgo)}, ${pct(b.narrative_share, 0)} by ${monthName(end)}. data_end is the last month with ${pct(t.data_end_rule.min_ratio, 0)} of its trailing ${t.data_end_rule.trailing_months}-month median.`,
  );
  const years = [];
  months.forEach((m, i) => {
    if (m.month.endsWith("-01") && +m.month.slice(0, 4) % 2 === 1) years.push({ i, text: m.month.slice(0, 4) });
  });
  const firstTest = idx(t.windows[0].test[0].slice(0, 7));
  const lastTest = idx(t.windows.at(-1).test[1].slice(0, 7));
  const events = [
    { i: idx(disputesEnd), text: `Disputes end, ${monthName(disputesEnd)}`, short: "Disputes end", row: 0 },
    { i: idx(end), text: `data_end, ${monthName(end)}`, short: "data_end", cls: "event--data", row: 1 },
    { i: idx(lastNarr), text: `Archive ends, ${monthName(lastNarr)}`, short: "Archive ends", row: 0 },
  ];
  const maxC = Math.max(...months.map((m) => m.complaints));
  const step = maxC > 600000 ? 200000 : 100000;
  const ticks = [];
  for (let v = 0; v <= maxC * 1.05; v += step) ticks.push(v);
  lines(document.getElementById("volume-chart"), {
    n: months.length,
    series: [
      { values: months.map((m) => m.narratives), cls: "s-cool", area: true },
      { values: months.map((m) => m.complaints), cls: "s-ink" },
    ],
    yMax: maxC * 1.05,
    yTicks: ticks,
    yFmt: (v) => (v ? `${v / 1000}k` : "0"),
    xLabels: years,
    events,
    shades: [{ from: firstTest, to: lastTest, text: "" }],
    label: `Complaints per month from ${monthName(months[0].month)} to ${monthName(months.at(-1).month)}, and those with a narrative, which collapse from late 2025.`,
  });
  document.getElementById("volume-chart").after(
    h(
      "p",
      { class: "keys" },
      h("span", {}, h("span", { class: "swatch s-ink" }), "All complaints"),
      h("span", {}, h("span", { class: "swatch s-cool" }), "With a narrative"),
      h("span", {}, h("span", { class: "swatch s-muted" }), "Shaded: the test quarters"),
    ),
  );
  const lateMax = Math.max(...months.map((m) => Math.max(m.late_rate ?? 0, m.complaints_late_rate))) * 1.1;
  const lticks = [];
  for (let v = 0; v <= lateMax; v += 0.01) lticks.push(+v.toFixed(2));
  lines(document.getElementById("late-chart"), {
    n: months.length,
    series: [
      { values: months.map((m) => m.complaints_late_rate), cls: "s-ink" },
      { values: months.map((m) => (m.month >= from && m.month <= end ? m.late_rate : null)), cls: "s-cool" },
    ],
    yMax: lateMax,
    yTicks: lticks,
    yFmt: (v) => pct(v, 0),
    xLabels: years,
    events: events.map((e) => ({ ...e, text: "", short: "", row: 0 })),
    label: "Share of complaints answered late each month, for complaints with a narrative and for all complaints.",
    height: 180,
  });
  document.getElementById("late-chart").after(
    h(
      "p",
      { class: "keys" },
      h("span", {}, h("span", { class: "swatch s-ink" }), "All complaints"),
      h("span", {}, h("span", { class: "swatch s-cool" }), "With a narrative (the label)"),
    ),
  );
  document.getElementById("data-table").append(
    table(
      ["Month", "Complaints", "With a narrative", "Narrative share", "Late, all", "Late, with narrative"],
      months.map((m) => [
        monthName(m.month),
        int(m.complaints),
        int(m.narratives),
        pct(m.narrative_share),
        pct(m.complaints_late_rate, 2),
        m.late_rate === null ? "n/a" : pct(m.late_rate, 2),
      ]),
    ),
  );
}

// --- 8 to 10. v1, limits, run it ----------------------------------------------------------

function history(d) {
  const v1 = d.v1;
  const best = v1.models[0];
  setText(
    "v1-lead",
    `v1 reported PR-AUC ${f3(best.pr_auc)} on one split, threshold chosen on ${v1.threshold.chosen_on}. Quoted as history.`,
  );
  const t = document.getElementById("v1-fixes");
  t.append(
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "v1"), h("th", { scope: "col" }, "Now"))),
    h("tbody", {}, v1.fixes.map((f) => h("tr", {}, h("td", {}, f.v1), h("td", {}, f.v2)))),
  );
}

function limits(d) {
  const { windows, metrics, manifest } = d;
  const low = windows.windows.filter((w) => w.low_narrative_share);
  const first = windows.windows[0];
  setText(
    "limit-shift",
    `${listJoin(low.map((w) => w.window))}: narrative on ${listJoin(low.map((w) => pct(w.narrative_share.test)))}, late ${listJoin(low.map((w) => pct(w.test.prevalence)))}. ${first.window}: ${pct(first.narrative_share.test)} and ${pct(first.test.prevalence)}.`,
  );
  const notRun = metrics.not_run;
  const models = [...new Set(notRun.map((c) => c.split("__")[0]))].map((m) => (metrics.model_labels[m] ?? m).split(/[ ,]/)[0]);
  setText(
    "limit-deep",
    notRun.length
      ? `${listJoin(models)} read “not run” until the Colab run.`
      : "Every planned model has been run.",
  );
  const live = manifest.files.find((f) => !f.name.startsWith("CCDB_Export_"));
  const date = new Date(live.downloaded).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" });
  setText("limit-live", `The live file changes daily; this is the ${date} copy.`);
}

function run(d) {
  const files = d.manifest.files;
  const bytes = files.reduce((s, f) => s + f.bytes, 0);
  setText("run-lead", "Every raw file is pinned by its SHA-256.");
  const archive = files.filter((f) => f.name.startsWith("CCDB_Export_"));
  document.getElementById("run-figures").replaceChildren(
    ...[
      ["Archive files", String(archive.length)],
      ["Live file", String(files.length - archive.length)],
      ["Zipped", `${(bytes / 1e9).toFixed(2)} GB`],
    ].map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v))),
  );
  const t = document.getElementById("manifest");
  const dates = [...new Set(files.map((f) => f.downloaded.slice(0, 10)))];
  setText("manifest-caption", `The data manifest, downloaded ${listJoin(dates)}`);
  const label = (name) => {
    const m = name.match(/^CCDB_Export_(\d+)_(.*)\.zip$/);
    return m ? [`Archive ${m[1]}`, m[2].replaceAll("_", " ")] : [name.replace(/\.zip$/, ""), "Live file, every complaint"];
  };
  t.append(
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "File"), h("th", { scope: "col" }, "Covers"), h("th", { scope: "col", class: "num" }, "MB"), h("th", { scope: "col" }, "SHA-256, first 12"))),
    h(
      "tbody",
      {},
      files.map((f) =>
        h(
          "tr",
          {},
          h("td", {}, h("a", { href: f.url }, label(f.name)[0])),
          h("td", {}, label(f.name)[1]),
          h("td", { class: "num" }, (f.bytes / 1e6).toFixed(1)),
          h("td", { class: "num" }, f.sha256.slice(0, 12)),
        ),
      ),
    ),
  );
}

// --- load --------------------------------------------------------------------------------

async function main() {
  // Start every fetch at once; each section draws as soon as the files it reads arrive, so the
  // answer does not wait for the larger curve file. All data and the scorer's model are fetched
  // during load, so the page keeps working offline afterwards.
  const pending = Object.fromEntries(FILES.map((n) => [n, fetch(`data/${n}.json`).then((r) => r.json())]));
  const d = {};
  const need = async (...names) => {
    for (const n of names) d[n] ??= await pending[n];
  };
  let cells = null;
  const withCells = async () => {
    await need("metrics");
    cells ??= cellIndex(d.metrics);
    return cells;
  };
  const steps = [
    [["metrics", "windows"], async () => answer(d, await withCells())],
    [["scorer"], () => initScorer(d)],
    [["curves", "metrics", "windows"], () => initThreshold(d)],
    [["metrics", "windows", "robustness"], async () => overTime(d, await withCells())],
    [["calibration", "metrics"], () => calibrationSection(d)],
    [["robustness", "metrics"], () => products(d)],
    [["timeline", "manifest"], () => timelineSection(d)],
    [["v1"], () => history(d)],
    [["windows", "metrics", "manifest"], () => limits(d)],
    [["manifest"], () => run(d)],
  ];
  await Promise.all(
    steps.map(async ([names, step]) => {
      try {
        await need(...names);
        await step();
      } catch (err) {
        console.error(err);
      }
    }),
  );
}

main();
