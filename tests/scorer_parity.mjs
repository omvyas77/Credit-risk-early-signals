// Scores parity inputs with docs/js/scorer.js; called by tests/test_scorer_parity.py.
// Usage: node tests/scorer_parity.mjs <model.json> <inputs.json>  (writes JSON to stdout)
import { readFileSync } from "node:fs";
import { prepare, score } from "../docs/js/scorer.js";

const [bundlePath, inputsPath] = process.argv.slice(2);
const model = prepare(JSON.parse(readFileSync(bundlePath, "utf8")));
const inputs = JSON.parse(readFileSync(inputsPath, "utf8"));
process.stdout.write(JSON.stringify(inputs.map((input) => score(model, input))));
