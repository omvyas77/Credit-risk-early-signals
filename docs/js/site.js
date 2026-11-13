// Fills the page from docs/data/*.json. Every number shown is read here at load time.
import { dotStrip, bandCI, lines, reliability } from "./charts.js";
import { initScorer } from "./try.js";
import { initThreshold } from "./threshold.js";
import { f3, pct, int, monthName, MONTHS, h, quarter, SEES, SEES_SHORT, MODEL, MODEL_SHORT } from "./util.js";

const FILES = ["metrics", "curves", "calibration", "timeline", "windows", "robustness", "scorer", "manifest", "v1"];

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
  const qname = new Map(windows.windows.map((w) => [w.window, quarter(w.dates.test[0])]));
  const models = [
    ["lr", MODEL.lr],
    ["lgbm", MODEL.lgbm],
  ];
  const rows = [];
  for (const [m, name] of models) {
    for (const s of ["text", "struct", "both"]) {
      rows.push({
        group: name,
        groupShort: MODEL_SHORT[m],
        label: SEES[s],
        short: SEES_SHORT[s],
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
  rows.push({
    group: "Without a model",
    groupShort: "No model",
    label: MODEL.company_rate.replace(", no model", ""),
    short: MODEL_SHORT.company_rate,
    model: "company_rate",
    set: "struct",
    points: metrics.windows.map((w) => {
      const c = cells.get(`company_rate__struct__${w}`);
      return { name: w, value: c.point.pr_auc, lo: c.ci95.pr_auc[0], hi: c.ci95.pr_auc[1], hollow: low.has(w) };
    }),
    mean: aw.company_rate__struct.pr_auc.mean,
  });
  rows.push({
    group: "Without a model",
    groupShort: "No model",
    label: "Guessing (the share late)",
    short: "Guessing",
    points: [],
    chance: aw.company_rate__struct.prevalence.mean,
    mean: aw.company_rate__struct.prevalence.mean,
  });
  dotStrip(document.getElementById("grid-chart"), rows, {
    label: `Score from 0 to 1 in each of ${metrics.windows.length} test quarters. Average for the linear model: text only ${f3(aw.lr__text.pr_auc.mean)}, company, product and issue ${f3(aw.lr__struct.pr_auc.mean)}, both together ${f3(aw.lr__both.pr_auc.mean)}. Guessing scores ${f3(aw.company_rate__struct.prevalence.mean)}.`,
    fmt: f3,
  });

  // Paired CIs on adding the narrative to the structured fields.
  const parts = models.map(([m, name]) => {
    const comps = metrics.comparisons.filter((c) => c.a[0] === m && c.a[1] === "both" && c.b[0] === m && c.b[1] === "struct");
    const below = comps.filter((c) => c.metrics.pr_auc.ci95[1] < 0).length;
    const above = comps.filter((c) => c.metrics.pr_auc.ci95[0] > 0).length;
    return `${MODEL_SHORT[m]}, worse in ${below} and better in ${above}`;
  });
  const lowNames = [...low].map((w) => qname.get(w));
  setText("grid-note", `A dot is one test quarter; hollow ones are ${listJoin(lowNames)}. Blue tick: the average.`);

  // How to read the numbers.
  const ws = windows.windows;
  setText(
    "def-score",
    `From 0 to 1: how well a model ranks the complaints answered late first. Guessing scores about ${f3(aw.company_rate__struct.prevalence.mean)}.`,
  );
  setText(
    "def-quarter",
    `A model learns from a year of complaints, then is tested once on a later quarter it never saw: ${qname.get(ws[0].window)} to ${qname.get(ws.at(-1).window)} (${ws[0].window} to ${ws.at(-1).window} in the code).`,
  );

  const head = ["Model, what it sees", ...metrics.windows.map((w) => `${qname.get(w)} (${w})`), "Average"];
  const body = rows
    .filter((r) => r.points.length)
    .map((r) => [
      r.model === "company_rate" ? MODEL.company_rate : `${r.group}, ${r.label}`,
      ...metrics.windows.map((w) => {
        const c = cells.get(`${r.model}__${r.set}__${w}`);
        return ciText(c.point.pr_auc, c.ci95.pr_auc);
      }),
      f3(r.mean),
    ]);
  body.push(["Guessing (the share late)", ...windows.windows.map((w) => f3(w.test.prevalence)), f3(aw.company_rate__struct.prevalence.mean)]);
  document.getElementById("grid-table").append(
    table(head, body, {
      caption: `Score with 95% range. Adding the text to company, product and issue, judged by a paired 95% range: ${parts.join("; ")} of ${metrics.windows.length} quarters.`,
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
    `Each model is scored once on a quarter it never saw. In ${listJoin(low.map((w) => quarter(w.dates.test[0])))}, few complaints had published text, and more of those were late.`,
  );
  const series = [
    { key: "lr__text", cls: "s-muted", label: SEES.text },
    { key: "lr__struct", cls: "s-ink", label: SEES.struct },
    { key: "lr__both", cls: "s-cool", label: SEES.both },
    { key: "company_rate__struct", cls: "s-ink", hollow: true, label: MODEL.company_rate },
  ];
  const bands = ws.map((w) => ({
    name: quarter(w.dates.test[0]),
    sub: [pct(w.test.prevalence), pct(w.narrative_share.test, 0)],
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
    label: `Score in each test quarter for the linear model; ${listJoin(low.map((w) => quarter(w.dates.test[0])))} are shaded: few complaints had published text.`,
    shadeLabel: "Little published text",
    subPrefix: ["late ", "with text "],
    subShort: ["late", "text"],
    fmt: f3,
  });
  chart.after(
    h(
      "p",
      { class: "keys" },
      series.map((s) => h("span", {}, h("span", { class: `swatch ${s.cls}` }), s.label)),
    ),
  );
  const head = ["Test quarter", "Dates", "Complaints", "Answered late (guessing)", "With published text", ...series.map((s) => s.label)];
  const body = ws.map((w, i) => [
    `${quarter(w.dates.test[0])} (${w.window})`,
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
  const merged = shades.length ? [{ from: Math.max(shades[0].from, 0), to: Math.min(shades.at(-1).to, months.length - 1), text: "Little published text" }] : [];
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
    label: "The share answered late and the model's average chance in each test month; the two move together, including the jump in 2026.",
    height: 200,
  });
  driftChart.after(
    h("p", { class: "keys" }, h("span", {}, h("span", { class: "swatch s-ink" }), "Share answered late"), h("span", {}, h("span", { class: "swatch s-cool" }), "Model’s average chance")),
  );
  document.getElementById("drift-table").append(
    table(
      ["Month", "Test quarter", "Complaints", "Answered late", "Average chance", "Score"],
      months.map((m) => [monthName(m.month), quarter(`${m.month}-01`), int(m.rows), pct(m.prevalence, 2), pct(m["lr__both:mean_score"], 2), f3(m["lr__both:pr_auc"])]),
      { numFrom: 2 },
    ),
  );
}

// --- 5. calibration -----------------------------------------------------------------------

function calibrationSection(d) {
  const { calibration, metrics, windows } = d;
  const qname = new Map(windows.windows.map((w) => [w.window, quarter(w.dates.test[0])]));
  const CAL = {
    lr_balanced: "Weighted, uncorrected",
    lr_balanced_platt: "Weighted, corrected (Platt)",
    lr_balanced_isotonic: "Weighted, corrected (isotonic)",
    lr: "Unweighted",
  };
  const order = [
    ["lr_balanced", "s-muted"],
    ["lr_balanced_platt", "s-cool"],
    ["lr_balanced_isotonic", "s-ink"],
    ["lr", "s-accent"],
  ];
  const select = document.getElementById("c-window");
  for (const w of metrics.windows) select.append(h("option", { value: w }, qname.get(w)));
  select.value = metrics.windows.at(-1);
  const chart = document.getElementById("cal-chart");
  const draw = () => {
    const w = select.value;
    const cellsFor = new Map(calibration.cells.filter((c) => c.window === w && c.featureset === "both").map((c) => [c.model, c]));
    const shown = order.filter(([m]) => cellsFor.has(m) && m !== "lr_balanced_isotonic");
    chart.replaceChildren();
    reliability(chart, {
      series: shown.map(([m, cls]) => ({ points: cellsFor.get(m).reliability.map(([x, y]) => [x, y]), cls })),
      label: `Honest chances, ${qname.get(w)}: the weighted, uncorrected model sits far right of the diagonal; the corrected one sits on it.`,
    });
    const stats = document.getElementById("cal-stats");
    stats.replaceChildren(
      ...table(
        ["Linear model, both together", "Average gap (ECE)", "Brier", "Log loss"],
        order
          .filter(([m]) => cellsFor.has(m))
          .map(([m, cls]) => {
            const c = cellsFor.get(m);
            return [h("span", {}, h("span", { class: `swatch ${cls}` }), CAL[m]), f3(c.ece.point), f3(c.brier.point), f3(c.log_loss.point)];
          }),
        { caption: `${qname.get(w)}; lower is better` },
      ).childNodes,
    );
    const binTable = document.getElementById("cal-table");
    binTable.replaceChildren(
      table(
        ["Model", "Group", "Chance given", "Share actually late", "Complaints"],
        shown.flatMap(([m]) => cellsFor.get(m).reliability.map(([x, y, n], i) => [CAL[m], String(i + 1), x.toFixed(4), y.toFixed(4), int(n)])),
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
  const labels = SEES;
  const win = d.windows.windows.find((w) => w.window === cp.window);
  const q = quarter(win.dates.test[0]);
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
      h("caption", { class: "visually-hidden" }, `Score, ${labels[s]}; rows: where the model learned, columns: where it was tested`),
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Learned on, tested on"), names.map((n) => h("th", { scope: "col", class: "num" }, n)))),
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
        h("tr", {}, h("th", { scope: "row" }, "One model for all products"), fs.pooled.map((v) => h("td", { style: "--v:0" }, f3(v)))),
        h("tr", {}, h("th", { scope: "row" }, "Guessing (the share late)"), fs.chance.map((v) => h("td", { style: "--v:0" }, f3(v)))),
      ),
    );
    document.getElementById("p-ci").replaceChildren(
      table(
        ["Learned on", ...names],
        names.map((n, i) => [n, ...fs.pr_auc[i].map((v, j) => ciText(v, fs.pr_auc_ci95[i][j]))]),
        { caption: `${labels[s]}: score with 95% range; rows: where it learned, columns: where it was tested.` },
      ),
    );
  };
  seg.addEventListener("change", draw);
  draw();
  setText(
    "p-note",
    `${q} test quarter, linear model. Outlined cells learned and were tested on the same product.`,
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
    `Complaints kept coming, but the share published with the consumer’s text fell from ${pct(a.narrative_share, 0)} in ${monthName(yearAgo)} to ${pct(b.narrative_share, 0)} by ${monthName(end)}. The data stops in ${monthName(end)}, the last month with at least ${pct(t.data_end_rule.min_ratio, 0)} of the usual amount of text.`,
  );
  const years = [];
  months.forEach((m, i) => {
    if (m.month.endsWith("-01") && +m.month.slice(0, 4) % 2 === 1) years.push({ i, text: m.month.slice(0, 4) });
  });
  const firstTest = idx(t.windows[0].test[0].slice(0, 7));
  const lastTest = idx(t.windows.at(-1).test[1].slice(0, 7));
  const events = [
    { i: idx(disputesEnd), text: `Disputes end, ${monthName(disputesEnd)}`, short: "Disputes end", row: 0 },
    { i: idx(end), text: `Data used ends, ${monthName(end)}`, short: "Data ends", cls: "event--data", row: 1 },
    { i: idx(lastNarr), text: `Text archive ends, ${monthName(lastNarr)}`, short: "Text ends", row: 0 },
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
    label: `Complaints per month from ${monthName(months[0].month)} to ${monthName(months.at(-1).month)}, and those published with the consumer’s text, which collapse from late 2025.`,
  });
  document.getElementById("volume-chart").after(
    h(
      "p",
      { class: "keys" },
      h("span", {}, h("span", { class: "swatch s-ink" }), "All complaints"),
      h("span", {}, h("span", { class: "swatch s-cool" }), "With published text"),
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
    label: "Share of complaints answered late each month, for complaints with published text and for all complaints.",
    height: 180,
  });
  document.getElementById("late-chart").after(
    h(
      "p",
      { class: "keys" },
      h("span", {}, h("span", { class: "swatch s-ink" }), "All complaints"),
      h("span", {}, h("span", { class: "swatch s-cool" }), "With published text (what is predicted)"),
    ),
  );
  document.getElementById("data-table").append(
    table(
      ["Month", "Complaints", "With published text", "Share with text", "Late, all", "Late, with text"],
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
    `The first version scored ${f3(best.pr_auc)} on one test period, with its cut-off picked on that ${v1.threshold.chosen_on} data. Quoted as history.`,
  );
  const t = document.getElementById("v1-fixes");
  t.append(
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "The first version"), h("th", { scope: "col" }, "Now"))),
    h("tbody", {}, v1.fixes.map((f) => h("tr", {}, h("td", {}, f.v1), h("td", {}, f.v2)))),
  );
}

function limits(d) {
  const { windows, metrics, manifest } = d;
  const low = windows.windows.filter((w) => w.low_narrative_share);
  const first = windows.windows[0];
  setText(
    "limit-shift",
    `By ${quarter(low.at(-1).dates.test[0])}, ${pct(low.at(-1).narrative_share.test, 0)} of complaints had published text and ${pct(low.at(-1).test.prevalence, 0)} of those were late; in ${quarter(first.dates.test[0])}, ${pct(first.narrative_share.test, 0)} and ${pct(first.test.prevalence, 0)}.`,
  );
  const notRun = metrics.not_run;
  const models = [...new Set(notRun.map((c) => c.split("__")[0]))].map((m) => (metrics.model_labels[m] ?? m).split(/[ ,]/)[0]);
  setText(
    "limit-deep",
    notRun.length
      ? `${notRun.length} planned language-model results were not run for lack of a GPU (${listJoin(models)}); they read “not run”.`
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
    [["scorer", "windows"], () => initScorer(d)],
    [["curves", "metrics", "windows"], () => initThreshold(d)],
    [["metrics", "windows", "robustness"], async () => overTime(d, await withCells())],
    [["calibration", "metrics", "windows"], () => calibrationSection(d)],
    [["robustness", "metrics", "windows"], () => products(d)],
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
