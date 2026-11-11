// Threshold workbench: a slider over the validation threshold grid of logistic regression on
// narrative and structured fields, with live test counts and editable costs.
import { f3, pct, int, h } from "./util.js";

const MODEL = "lr__both";

function monthsBetween(a, b) {
  const [y0, m0] = a.split("-").map(Number);
  const [y1, m1] = b.split("-").map(Number);
  return (y1 - y0) * 12 + (m1 - m0) + 1;
}

export function initThreshold(d) {
  const { curves, metrics, windows } = d;
  const cols = curves.columns.thresholds;
  const at = Object.fromEntries(cols.map((c, i) => [c, i]));
  const select = document.getElementById("t-window");
  const slider = document.getElementById("t-slider");
  const valueOut = document.getElementById("t-value");
  const hint = document.getElementById("t-hint");
  const fp = document.getElementById("t-fp");
  const fn = document.getElementById("t-fn");
  const best = document.getElementById("t-best");
  const figures = document.getElementById("t-figures");
  const caption = document.getElementById("t-caption");
  const cell = (id) => document.getElementById(id);

  fp.value = metrics.costs.fp;
  fn.value = metrics.costs.fn;
  for (const w of windows.windows) {
    select.append(h("option", { value: w.window }, `${w.window}${w.low_narrative_share ? ", 2026" : ""}`));
  }
  select.value = windows.windows.at(-1).window;

  let grid = [];
  let win = null;
  const costs = () => [Math.max(+fp.value || 0, 0), Math.max(+fn.value || 0, 0)];
  const cheapest = () => {
    const [cfp, cfn] = costs();
    let bestI = 0;
    let bestCost = Infinity;
    grid.forEach((r, i) => {
      const c = cfp * r[at.val_fp] + cfn * r[at.val_fn];
      if (c < bestCost) {
        bestCost = c;
        bestI = i;
      }
    });
    return bestI;
  };

  function load() {
    win = windows.windows.find((w) => w.window === select.value);
    grid = [...curves.cells[`${MODEL}__${win.window}`].thresholds].sort((a, b) => b[0] - a[0]);
    slider.max = String(grid.length - 1);
    slider.value = String(cheapest());
    render();
  }

  function render() {
    const r = grid[+slider.value];
    const [cfp, cfn] = costs();
    const tp = r[at.test_tp];
    const fpN = r[at.test_fp];
    const fnN = r[at.test_fn];
    const tn = r[at.test_tn];
    const n = tp + fpN + fnN + tn;
    const valN = r[at.val_tp] + r[at.val_fp] + r[at.val_fn] + r[at.val_tn];
    const valFlag = (r[at.val_tp] + r[at.val_fp]) / valN;
    const months = monthsBetween(win.dates.test[0].slice(0, 7), win.dates.test[1].slice(0, 7));
    const precision = tp + fpN ? tp / (tp + fpN) : null;
    const recall = tp / (tp + fnN);
    const cost = ((cfp * fpN + cfn * fnN) / n) * 1000;
    valueOut.textContent = r[0] < 0.001 ? r[0].toExponential(1) : f3(r[0]);
    slider.setAttribute("aria-valuetext", `threshold ${valueOut.textContent}, flags ${pct(valFlag)} of validation complaints`);
    const isBest = +slider.value === cheapest();
    hint.textContent = `Flags ${pct(valFlag)} on validation${isBest ? ", the cheapest there" : ""}.`;
    figures.replaceChildren(
      ...[
        ["Precision", precision === null ? "n/a" : pct(precision, 0)],
        ["Recall", pct(recall, 0)],
        ["Flagged a month", int((tp + fpN) / months)],
        ["Cost per 1,000", cost.toFixed(0)],
      ].map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v))),
    );
    caption.textContent = `${win.window} test: ${int(n)} complaints, ${int(tp + fnN)} late.`;
    cell("t-tp").textContent = int(tp);
    cell("t-fn-n").textContent = int(fnN);
    cell("t-fp-n").textContent = int(fpN);
    cell("t-tn").textContent = int(tn);
  }

  select.addEventListener("change", load);
  slider.addEventListener("input", render);
  for (const input of [fp, fn]) input.addEventListener("input", render);
  best.addEventListener("click", () => {
    slider.value = String(cheapest());
    render();
    slider.focus();
  });
  load();
}
