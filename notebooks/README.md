# Deep models on Colab

`colab_deep.ipynb` runs `python -m cres deep --model minilm|distilbert` on a Colab GPU. It
holds no logic: everything it runs is in `src/cres/deep.py`. The same-data baseline (`lr_sub`,
TF-IDF LR on DistilBERT's training subsample) runs on the laptop with `make deep-baseline`.

## Steps

1. **On the laptop, write the input.**

   ```bash
   make deep-input
   ```

   This writes `data/deep_input/` (sizes are printed; about 345 MB for W5):
   `narratives.parquet` (complaint id and narrative for every W5 row), `splits.parquet` (the
   W5 window tags) and `W5/` (the structured features, ids and labels from
   `data/features/W5/`). It holds narrative text, so keep it in a private Drive folder and out
   of git.

2. **Upload it to Drive.** In Google Drive, make a folder `cres_deep` in My Drive and upload
   the `data/deep_input` folder into it (New > Folder upload), so the result is
   `My Drive/cres_deep/deep_input/`.

3. **Open the notebook in Colab.** File > Open notebook > GitHub, search
   `omvyas77/Credit-risk-early-signals`, pick branch `v2` and `notebooks/colab_deep.ipynb`.
   Set Runtime > Change runtime type > T4 GPU (or better).

4. **Run all cells** (Runtime > Run all) and allow Drive access when asked. The cells:
   check the GPU and Python 3.12, mount Drive, clone `v2` and `pip install ".[deep]"` (this
   replaces Colab's torch with the pinned version; pip's warnings about torchvision or
   torchaudio don't matter here), copy `deep_input` into `data/`, run MiniLM and copy its
   files to Drive, then run DistilBERT and copy its files to Drive. On a T4, expect about 15
   minutes of embedding plus up to 30 minutes of LR fits on the CPU for MiniLM, and 40 to 60
   minutes for DistilBERT (3 epochs). Both print progress. The standard runtime's RAM is
   enough: MiniLM's `both` matrix for a year of training rows is about 4 GB.
   If the runtime restarts but keeps its disk, rerun all cells: MiniLM reuses finished
   embedding chunks and DistilBERT resumes after its last finished epoch.

5. **Bring the outputs back.** In Drive, download `My Drive/cres_deep/output` (Drive zips
   it) and unzip it. It mirrors the repo, so from the repo root:

   ```bash
   cp -R ~/Downloads/output/. .
   ```

   That puts `minilm__{text,both}__W5.parquet` and `distilbert__{text,both}__W5.parquet` in
   `data/preds/`, and their run notes (`*.json`: chosen C, epochs, validation AP by epoch,
   GPU) in `models/`.

6. **Evaluate.**

   ```bash
   make evaluate
   ```

   The "Better text models" table in `reports/summary.md` then shows the deep rows next to
   LR and `lr_sub`. Until their files exist, those cells read "not run" and are listed under
   `not_run` in `reports/metrics.json`.

## Checks

- `models/distilbert__text__W5.json` and `models/lr_sub__text__W5.json` both carry
  `subsample.ids_sha256`. They must match: that is the evidence that DistilBERT and its
  baseline trained on the same rows.
- Other windows: rerun step 1 with `uv run python -m cres deep-input --windows W4 W5`, set
  `WINDOW` in the notebook's first cell, and add the window to `deep.windows` in
  `config/default.yaml` so the summary expects it.
