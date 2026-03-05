# Web Examples Page — Design

## Goal
Add a dedicated examples page (`examples.html`) to the AutoChecklist website that showcases precomputed pipeline comparison results using real benchmark data. Serves two purposes: help visitors understand what checklist evaluation looks like in practice, and demonstrate the value of comparing multiple pipelines side by side.

## Data Pipeline

**Location:** `../autocl-demo/` (sibling directory, outside website repo)

### Steps
1. **Sample selection** — Pull 2-3 samples from InFoBench and/or WildBench (via HuggingFace). Select diverse task types that produce interesting pipeline differences (e.g. multi-constraint instruction, open-ended generation).
2. **Pipeline execution** — Run 4 pipelines on each sample, organized by generator type × reference usage:
   - Direct (no reference) — e.g. `tick`
   - Direct + Reference — e.g. `rocketeval`
   - Contrastive (no reference) — e.g. `rlcf_candidate`
   - Contrastive + Reference — e.g. `rlcf_direct`
3. **Output** — JSON/JSONL with precomputed results per example: input, target, optional reference, and per-pipeline data (checklist items, pass/fail, reasoning, confidence, scores, weights).
4. **Export** — Transform raw output into a compact JSON blob for the web page (embedded `<script>` or loaded `.json` file).

Scripts save incrementally (JSONL) for resumability.

## Examples Page (`examples.html`)

### Layout
Mirrors the local UI's Compare Page closely:

- **Tab bar** at top to switch between 2-3 precomputed examples (labeled by task type/description)
- **Input section** (read-only): instruction/query, target response, optional reference
- **Method cards row** — 4 cards side by side (responsive: stack on mobile), styled to match the local UI's `MethodCard`:
  - Pipeline name + description header
  - Score display: fraction (e.g. "7/8") and normalized percentage
  - Checklist items list with pass/fail checkmarks
  - Reasoning text per item
  - Confidence labels
  - Weights if applicable

### Tech
- Static HTML + CSS + vanilla JS (no build step, consistent with homepage)
- Shares `styles.css` warm paper design system with homepage
- Tab switching swaps which precomputed JSON data renders
- Data embedded in `<script>` tag or loaded from a small `.json` file

## Homepage Integration

Minimal changes to `index.html`:
- Replace "Examples coming soon" placeholder with a teaser line + "View Examples" link
- Add "Examples" to the nav bar

## Decisions
- **Approach:** Tabbed example selector (vs. vertical scroll or comparison table)
- **Fidelity:** Close replica of Compare Page UI (method cards, scores, checklist items, reasoning, confidence)
- **Data source:** Real benchmark samples (InFoBench/WildBench)
- **Pipelines:** 4 pipelines covering the 2x2 of Direct/Contrastive × with/without Reference
- **Example count:** 2-3
- **Scripts location:** `../autocl-demo/` (separate from website repo)
