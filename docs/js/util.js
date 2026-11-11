// Formatting and DOM helpers shared by the page modules.

export const f3 = (x) => (x === null || x === undefined ? "n/a" : x.toFixed(3));
export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
export const int = (x) => Math.round(x).toLocaleString("en-US");
export const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const monthName = (ym, long = false) => {
  const [y, m] = ym.split("-").map(Number);
  const name = long
    ? new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-US", { month: "long", timeZone: "UTC" })
    : MONTHS[m - 1];
  return `${name} ${y}`;
};

/** Build an element tree from a tag, attributes and children (strings become text). */
export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c);
  return node;
}
