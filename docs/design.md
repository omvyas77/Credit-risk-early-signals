# Design: complaint-handling risk

The locked system for the site in `docs/`. Every page reads this first and shares it. Change it on purpose; the file is the rule.

## System
- Genre: modern-minimal
- Shape of the first page: a long report page, the answer as the first screen and the scorer second
- Theme: Console, paper colourway (an instrument kept by careful people, on light paper)
- Axes: light/grotesk-sans/chromatic-amber · posture: restrained

## Tokens
`tokens.css` is the source of truth; `base.css` sets the elements. The roles:
```css
:root {
  --color-paper: oklch(98% 0.006 85);  --color-paper-2: oklch(96.2% 0.007 85);  --color-paper-3: oklch(93.6% 0.008 85);
  --color-ink: oklch(19% 0.010 60);  --color-ink-2: oklch(31% 0.010 60);  --color-muted: oklch(45% 0.010 60);
  --color-rule: oklch(87.5% 0.010 80);  --color-rule-2: oklch(76% 0.012 80);
  --color-accent: oklch(54% 0.14 70);  --color-accent-ink: oklch(98% 0.01 85);  --color-focus: oklch(54% 0.14 70);
  --color-cool: oklch(52% 0.14 250);  --color-ok: oklch(48% 0.13 150);  --color-bad: oklch(52% 0.17 25);
  --font-display: "Host Grotesk", ui-sans-serif, system-ui, sans-serif;  --font-body: var(--font-display);
  --font-outlier: "Atkinson Hyperlegible Mono", ui-monospace, monospace;
  --radius-card: 10px;  --radius-pill: 6px;  --rule-hair: 1px;  --shadow-card: none;
}
```
The spacing scale, the type scale, the easings and the durations live in `tokens.css`. Fonts are self-hosted in `docs/fonts/` (Host Grotesk 400 and 500, Atkinson Hyperlegible Mono 400 and 500), loaded with `font-display: swap`.

## Meaning
- Amber is the reader: the one primary action, the threshold they set, their own inputs.
- Cool blue is the instrument: a model's reading, the lead series in a chart.
- Ok and bad tints appear only for direction (a word that lowers or raises the score) and for the confusion matrix.
- Everything else is ink, muted ink and rules. No shadows, glows or gradients on surfaces.

## Type
- Display: Host Grotesk 500, tracking -0.03em, sentence case, roman.
- Body: Host Grotesk 400 at 16px or more, line-height 1.55, in `--color-muted`; the lead in `--color-ink-2`.
- Labels, captions, table heads: the body face a step smaller, sentence case, untracked.
- Measured: Atkinson Hyperlegible Mono for numbers a model or the pipeline wrote (metrics, probabilities, counts, hashes), always tabular. Never a label, a head or a button.

## Surfaces
- Hairline panels: 1px `--color-rule`, `--color-paper-2` inside, 10px radius; panels that sit together are one surface split by 1px rules.
- Three steps: the paper, a panel one step up, the deepest step for code and the manifest.
- Charts are hand-written SVG on the paper: 1px `--color-rule` grid, 2px lines, one cool series that argues, the rest muted. Each chart has a table fallback in a `<details>` beneath it.

## Controls
- Primary: amber fill, `--color-accent-ink` text, 6px radius, 48px tall.
- Secondary: 1px `--color-rule-2` outline on `--color-paper-2`, same radius and height.
- Inputs and selects: `--color-paper` inside a 1px `--color-muted` edge, 6px radius, 48px tall; the focus ring is a 2px amber outline, instant.
- The slider: a native range input, thumb in amber, operable by arrow keys, Page Up/Down, Home and End.

## Motion
- Character: still. A value changes when its source does, over `--dur-short` on `--ease-out`; no reveals, no parallax, no hover lift. Under reduced motion everything ends at once.

## Content rules
- Every number is read from `docs/data/*.json` at load time; none is typed into the HTML.
- Framing is complaint-handling risk. Not for credit decisions.
- Narrative text never appears; example complaints are written by hand.

## Exports
`tokens.css` in this folder. Tailwind, DTCG or shadcn blocks are added here on request.
