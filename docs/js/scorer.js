// In-browser scorer for complaint-handling risk: the chance that a company misses the CFPB's
// 15-day response window. Not for credit decisions.
//
// Reproduces src/cres/text.py (tokenizer spec version 1) and the logistic regression in
// docs/scorer/model.json, which src/cres/scorer.py writes. The parity test
// (tests/test_scorer_parity.py) checks this file against the Python reference on 1,000
// synthetic texts. Change both files together, then regenerate the fixtures and rerun it.
//
// Tokenizer spec (text.py):
//   1. Unicode NFKC normalisation.
//   2. URLs (http://, https:// or www., any case, up to the next whitespace) and emails
//      (non-space runs containing @) become a space.
//   3. Every run of two or more capital X's (CFPB redactions) becomes a space.
//   4. Lowercase.
//   5. Tokens are runs of a-z joined by single apostrophes (don't, company's).
//   6. Features are unigrams plus adjacent-pair bigrams joined by one space.
// NFKC and lowercasing use the browser's Unicode tables; characters added to Unicode after
// version 15 may normalise differently from Python 3.12.

export const TOKENIZER_VERSION = 1;
export const TOKENIZER_HASH = "2a539f97bd248695";

// Python's \s for str patterns (str.isspace), which differs from JavaScript's \s: it adds
// U+001C to U+001F and U+0085, and leaves out U+FEFF.
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const URL = new RegExp(`(?:[hH][tT][tT][pP][sS]?://|[wW][wW][wW]\\.)[^${WS}]*`, "g");
const EMAIL = new RegExp(`[^${WS}]*@[^${WS}]*`, "g");
const REDACTION = /X{2,}/g;
const TOKEN = /[a-z]+(?:'[a-z]+)*/g;

/** Steps 1 to 3 of the spec: the text before lowercasing and tokenising. */
export function clean(text) {
  return text.normalize("NFKC").replace(URL, " ").replace(EMAIL, " ").replace(REDACTION, " ");
}

/** Tokens of `text` per the spec. */
export function tokens(text) {
  return clean(text).toLowerCase().match(TOKEN) ?? [];
}

/** Unigrams followed by bigrams. */
export function ngrams(toks) {
  const out = toks.slice();
  for (let i = 0; i + 1 < toks.length; i++) out.push(`${toks[i]} ${toks[i + 1]}`);
  return out;
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** One regex per lexicon group, longest phrases first, matched on whole tokens. */
function lexiconPatterns(lexicon) {
  const out = {};
  for (const [group, phrases] of Object.entries(lexicon)) {
    const alts = phrases
      .slice()
      .sort((a, b) => b.length - a.length) // stable, as Python's sorted()
      .map(escapeRegExp)
      .join("|");
    out[group] = new RegExp(`(?<= )(?:${alts})(?= )`, "g");
  }
  return out;
}

/** Non-overlapping occurrences of each group's phrases in the token list. */
export function lexiconCounts(toks, patterns) {
  const joined = ` ${toks.join(" ")} `;
  const out = {};
  for (const [group, re] of Object.entries(patterns)) out[group] = (joined.match(re) ?? []).length;
  return out;
}

function transform(value, how) {
  if (how === "log1p") return Math.sign(value) * Math.log1p(Math.abs(value));
  if (how === "logit") {
    const v = Math.min(Math.max(value, 1e-9), 1 - 1e-9);
    return Math.log(v / (1 - v));
  }
  if (how === "none") return value;
  throw new Error(`unknown transform ${how}`);
}

/**
 * Check a parsed model.json and index it for scoring. Throws if the bundle was built for a
 * different tokenizer than this file implements.
 */
export function prepare(bundle) {
  if (bundle.tokenizer?.hash !== TOKENIZER_HASH || bundle.tokenizer?.version !== TOKENIZER_VERSION) {
    throw new Error(
      `model.json was built for tokenizer ${bundle.tokenizer?.hash}, scorer.js implements ${TOKENIZER_HASH}`,
    );
  }
  return {
    bundle,
    // Maps, not plain objects: terms such as "constructor" must not hit Object.prototype.
    vocabulary: new Map(Object.entries(bundle.vocabulary)),
    companies: new Map(Object.entries(bundle.companies)),
    product: new Map(Object.entries(bundle.categorical.product)),
    issue: new Map(Object.entries(bundle.categorical.issue)),
    patterns: lexiconPatterns(bundle.tokenizer.lexicon),
  };
}

/** Fetch and prepare model.json. */
export async function loadModel(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: ${response.status}`);
  return prepare(await response.json());
}

/**
 * Score one complaint.
 *   model: from prepare() or loadModel()
 *   input: { text, product, issue, company, daysToCompany } — all optional. A product, issue
 *          or company the model doesn't list falls back to the bundle's defaults.
 * Returns { probability, logit, parts, tokens, lexicon, terms }. `parts` are the log-odds
 * contributions that sum to `logit`; `terms` lists each vocabulary term found, in order of
 * first appearance, with its count, TF-IDF weight and contribution.
 */
export function score(model, input = {}) {
  const { bundle, vocabulary } = model;
  const toks = tokens(input.text ?? "");
  const counts = new Map();
  for (const g of ngrams(toks)) {
    if (vocabulary.has(g)) counts.set(g, (counts.get(g) ?? 0) + 1);
  }
  const weights = new Map();
  let sumSquares = 0;
  for (const [g, n] of counts) {
    const w = (1 + Math.log(n)) * vocabulary.get(g)[0];
    weights.set(g, w);
    sumSquares += w * w;
  }
  const norm = Math.sqrt(sumSquares) || 1;
  const terms = [];
  let textPart = 0;
  for (const [g, w] of weights) {
    const x = w / norm;
    const contribution = x * vocabulary.get(g)[1];
    terms.push({ term: g, count: counts.get(g), tfidf: x, contribution });
    textPart += contribution;
  }
  const lexicon = lexiconCounts(toks, model.patterns);
  const [lateRate, logVolume] = model.companies.get(input.company) ?? bundle.company_default;
  const values = {
    days_to_company: input.daysToCompany ?? bundle.defaults.days_to_company,
    company_late_rate: lateRate,
    company_log_volume: logVolume,
    ...lexicon,
  };
  const numeric = {};
  for (const n of bundle.numeric) {
    numeric[n.name] = (n.coef * (transform(values[n.name], n.transform) - n.mean)) / n.scale;
  }
  const cat = bundle.categorical;
  const dropdown = (variable, choice) => {
    const pair = model[variable].get(choice);
    return pair ? pair[0] + pair[1] : cat[`${variable}_default`];
  };
  let lexiconPart = 0;
  for (const group of Object.keys(bundle.tokenizer.lexicon)) lexiconPart += numeric[group];
  const parts = {
    intercept: bundle.intercept,
    text: textPart,
    lexicon: lexiconPart,
    company: numeric.company_late_rate + numeric.company_log_volume,
    days_to_company: numeric.days_to_company,
    product: dropdown("product", input.product),
    issue: dropdown("issue", input.issue),
    held: cat.held_baseline,
  };
  let logit = 0;
  for (const v of Object.values(parts)) logit += v;
  return { probability: 1 / (1 + Math.exp(-logit)), logit, parts, tokens: toks, lexicon, terms };
}
