// The in-browser scorer: text plus product, issue and company pickers. The model file is
// fetched once at load, so scoring keeps working offline afterwards.
import { loadModel, score } from "./scorer.js";
import { f3, pct, h, quarter } from "./util.js";

// Written by hand for this page; none of these is a real complaint.
const EXAMPLES = [
  {
    name: "Student loan",
    text: "My servicer put my payment on the wrong loan and says I am past due. Nobody answers.",
    product: "Student loan",
    issue: "Dealing with your lender or servicer",
    company: "MOHELA",
  },
  {
    name: "Credit report",
    text: "An account I never opened is on my credit report. I disputed it a month ago.",
    product: "Credit reporting or other personal consumer reports",
    issue: "Incorrect information on your report",
    company: "EQUIFAX, INC.",
  },
  {
    name: "Debt collector",
    text: "A collector keeps calling me at work about a debt I do not owe and threatened to sue.",
    product: "Debt collection",
    issue: "Attempts to collect debt not owed",
    company: "Portfolio Recovery Associates, LLC",
  },
  {
    name: "Money transfer",
    text: "Someone sent money from my account that I did not authorize, and the app will not refund me.",
    product: "Money transfer, virtual currency, or money service",
    issue: "Fraud or scam",
    company: "Block, Inc.",
  },
];

const ANY_PRODUCT = "";
const prob = (p) => (p < 0.001 ? "under 0.1%" : pct(p, p < 0.1 ? 1 : 0));
const signed = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`;

function option(value, label) {
  return h("option", { value }, label);
}

export async function initScorer(d) {
  const form = document.getElementById("scorer-form");
  const text = document.getElementById("s-text");
  const product = document.getElementById("s-product");
  const issue = document.getElementById("s-issue");
  const company = document.getElementById("s-company");
  const out = document.getElementById("scorer-out");
  const probNode = document.getElementById("s-prob");
  const compare = document.getElementById("s-compare");
  const readback = document.getElementById("s-readback");
  const chips = document.getElementById("examples");

  const info = d.scorer;
  const m = info.metrics;
  document.getElementById("scorer-note").textContent =
    `A smaller copy of the linear model. On ${quarter(d.windows.windows.find((w) => w.window === info.window).dates.test[0])} it scores ${f3(m.scorer.point.pr_auc)}; the full model, ${f3(m.full.point.pr_auc)}. Not for credit decisions.`;

  out.setAttribute("aria-busy", "true");
  probNode.textContent = "Loading the model";
  let model;
  try {
    model = await loadModel("scorer/model.json");
  } catch (err) {
    probNode.textContent = "The model did not load";
    out.removeAttribute("aria-busy");
    throw err;
  }
  out.removeAttribute("aria-busy");
  const b = model.bundle;
  const cat = b.categorical;

  product.append(option(ANY_PRODUCT, "Not chosen (average)"), ...Object.keys(cat.product).sort().map((p) => option(p, p)));
  const fillIssues = () => {
    const keep = issue.value;
    const list = product.value ? (cat.issues_by_product[product.value] ?? Object.keys(cat.issue)) : Object.keys(cat.issue);
    issue.replaceChildren(option("", "Not chosen (average)"), ...[...list].sort().map((i) => option(i, i)));
    issue.value = list.includes(keep) ? keep : "";
  };
  fillIssues();
  company.append(
    option("", "Not on the list (median history)"),
    ...Object.keys(b.companies)
      .sort((a, c) => a.localeCompare(c, "en", { sensitivity: "base" }))
      .map((c) => option(c, c)),
  );

  const chipNodes = EXAMPLES.map((ex, i) => {
    const btn = h("button", { type: "button", class: "chip", "aria-pressed": "false" }, ex.name);
    btn.addEventListener("click", () => {
      text.value = ex.text;
      product.value = ex.product;
      fillIssues();
      issue.value = ex.issue;
      company.value = ex.company;
      chipNodes.forEach((n, k) => n.setAttribute("aria-pressed", String(k === i)));
      update();
    });
    return btn;
  });
  chips.append(...chipNodes);

  const bar = (label, p, muted) =>
    h(
      "div",
      { class: `bar-row${muted ? " bar-row--muted" : ""}` },
      h("span", {}, label),
      h("span", { class: "measured" }, prob(p)),
      h("span", { class: "bar", "aria-hidden": "true" }, h("i", { style: `transform: scaleX(${Math.min(Math.max(p, 0.004), 1).toFixed(4)})` })),
    );

  function renderReadback(value, terms) {
    const weight = new Map();
    for (const t of terms) if (!t.term.includes(" ")) weight.set(t.term, t.contribution);
    const nodes = [];
    const re = /[A-Za-z]+(?:'[A-Za-z]+)*/g;
    let last = 0;
    let match;
    while ((match = re.exec(value))) {
      if (match.index > last) nodes.push(value.slice(last, match.index));
      const w = weight.get(match[0].toLowerCase());
      if (w !== undefined && Math.abs(w) >= 0.01) {
        nodes.push(h("mark", { class: w > 0 ? "up" : "down", title: `log-odds ${signed(w)}` }, match[0]));
      } else nodes.push(match[0]);
      last = match.index + match[0].length;
    }
    nodes.push(value.slice(last));
    readback.replaceChildren(...(value.trim() ? nodes : ["Write or pick a complaint to see its words weighed."]));
  }

  // Coalesce bursts of typing into one score; a timer, not a frame, so it also runs in a
  // background tab.
  let pending = 0;
  function update() {
    clearTimeout(pending);
    pending = setTimeout(() => {
      const input = {
        text: text.value,
        product: product.value || undefined,
        issue: issue.value || undefined,
        company: company.value || undefined,
      };
      const full = score(model, input);
      const alone = score(model, { text: text.value });
      probNode.textContent = prob(full.probability);
      const p = full.parts;
      compare.replaceChildren(
        bar("Text alone", alone.probability, true),
        bar("With company, product, issue", full.probability, false),
      );
      document.getElementById("s-parts").replaceChildren(
          ...[
            ["Company history", p.company],
            ["Product", p.product],
            ["Issue", p.issue],
            ["Words", p.text],
            ["Urgency and harm phrases", p.lexicon],
          ].map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", { class: "measured" }, signed(v)))),
      );
      renderReadback(text.value, full.terms);
    }, 30);
  }

  form.addEventListener("input", (e) => {
    if (e.target === text) chipNodes.forEach((n) => n.setAttribute("aria-pressed", "false"));
    update();
  });
  product.addEventListener("change", () => {
    fillIssues();
    update();
  });
  form.addEventListener("submit", (e) => e.preventDefault());
  chipNodes[0].click();
}
