# Web Examples Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an examples page to the AutoChecklist website showing precomputed pipeline comparison results from InFoBench/WildBench, closely mirroring the local UI's Compare Page.

**Architecture:** Two workstreams — (1) Python scripts in `../autocl-demo/` to select benchmark samples and run 4 pipelines (tick, rocketeval, rlcf_candidates_only, rlcf_candidate), producing a web-ready JSON blob; (2) static HTML/CSS/JS in the website repo creating `examples.html` with tabbed example switching and method cards styled after the local UI. Homepage gets a nav link and teaser.

**Tech Stack:** Python + autochecklist library (data generation), vanilla HTML/CSS/JS (web page), HuggingFace datasets (sample source)

**Design doc:** `docs/plans/2026-03-04-web-examples-design.md`

**Security note:** The examples page renders precomputed JSON that we generate ourselves. The `escapeHtml()` function sanitizes all text before DOM insertion as an additional safety layer. No user-supplied or external data is rendered.

---

## Task 1: Set up autocl-demo project

**Files:**
- Create: `../autocl-demo/pyproject.toml`
- Create: `../autocl-demo/.python-version`

**Step 1: Create directory and init uv project**

```bash
mkdir -p ../autocl-demo
cd ../autocl-demo
uv init --no-readme
```

**Step 2: Add dependencies**

```bash
cd ../autocl-demo
uv add datasets  # HuggingFace datasets
uv add autochecklist --dev --editable ../AutoChecklist  # local dev install
```

**Step 3: Commit**

```bash
cd ../autocl-demo
git init
git add .
git commit -m "init autocl-demo project with dependencies"
```

---

## Task 2: Write sample selection script

**Files:**
- Create: `../autocl-demo/select_samples.py`
- Create: `../autocl-demo/samples.jsonl` (output)

Selects 2-3 diverse samples from InFoBench and/or WildBench. Criteria: tasks with multiple constraints (to produce interesting checklist differences), varied task types. Must include reference answers for pipelines that need them.

**Step 1: Write the script**

```python
"""Select benchmark samples for web examples.

Pulls from InFoBench (keirp/infobench or similar) and/or WildBench.
Saves selected samples as JSONL with fields: input, target, reference (optional), source, task_type.

Selection criteria:
- Multi-constraint instructions (interesting checklist differences)
- Diverse task types
- Include reference where available
"""
import json
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("samples.jsonl")

def select_infobench_samples():
    """Pull 2-3 samples from InFoBench.

    InFoBench has instruction-following tasks with decomposed requirements.
    Fields typically: instruction, input, output, decomposed_questions.

    Look for samples where:
    - The instruction has 3+ constraints/requirements
    - A reference answer exists
    - The task is interesting to showcase
    """
    # TODO: Load dataset, inspect fields, select samples
    # ds = load_dataset("keirp/infobench", split="test")
    # Inspect: print(ds.column_names), print(ds[0])
    # Select 2-3 samples manually after inspection
    pass

def select_wildbench_samples():
    """Pull samples from WildBench if InFoBench alone isn't sufficient.

    WildBench: allenai/WildBench — realistic LLM evaluation tasks.
    """
    pass

def main():
    # 1. Load and inspect datasets
    # 2. Select 2-3 samples with good diversity
    # 3. Format as {input, target, reference, source, task_type}
    # 4. Write to samples.jsonl (append mode for resumability)
    samples = []

    # After inspection, manually curate sample indices
    # samples = select_infobench_samples()

    with open(OUTPUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT}")

if __name__ == "__main__":
    main()
```

This script requires manual curation — the implementer should:
1. Load the dataset, print schema and a few rows
2. Browse for samples with 3+ constraints and reference answers
3. Hardcode the selected indices
4. Run to produce `samples.jsonl`

**Step 2: Run dataset exploration interactively**

```bash
cd ../autocl-demo
uv run python -c "
from datasets import load_dataset
ds = load_dataset('keirp/infobench', split='test')
print('Columns:', ds.column_names)
print('Num rows:', len(ds))
print('Sample 0:', ds[0])
"
```

Inspect output, identify good samples, update script with selected indices.

**Step 3: Run the script to produce samples.jsonl**

```bash
cd ../autocl-demo
uv run python select_samples.py
```

Verify: `cat samples.jsonl | wc -l` should show 2-3 lines.

**Step 4: Commit**

```bash
cd ../autocl-demo
git add select_samples.py samples.jsonl
git commit -m "add sample selection script with curated benchmark samples"
```

---

## Task 3: Write pipeline runner script

**Files:**
- Create: `../autocl-demo/run_pipelines.py`
- Create: `../autocl-demo/results/` (output directory)

Runs 4 pipelines on each sample. Saves results incrementally as JSONL per pipeline.

**Step 1: Write the script**

```python
"""Run 4 pipelines on selected samples.

Pipelines (2x2 matrix of generator type x reference usage):
1. tick          — Direct, no reference
2. rocketeval    — Direct + Reference
3. rlcf_candidates_only — Contrastive, no reference
4. rlcf_candidate — Contrastive + Reference

Each pipeline writes results to results/{pipeline_name}.jsonl incrementally.
Existing results are skipped on re-run (resume support).
"""
import json
from pathlib import Path
from autochecklist import pipeline

SAMPLES_PATH = Path("samples.jsonl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PIPELINES = ["tick", "rocketeval", "rlcf_candidates_only", "rlcf_candidate"]
GENERATOR_MODEL = "openai/gpt-4o-mini"
SCORER_MODEL = "openai/gpt-4o-mini"

def load_samples():
    with open(SAMPLES_PATH) as f:
        return [json.loads(line) for line in f]

def get_completed_indices(pipeline_name):
    """Check which sample indices already have results."""
    path = RESULTS_DIR / f"{pipeline_name}.jsonl"
    if not path.exists():
        return set()
    indices = set()
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            indices.add(data["index"])
    return indices

def run_pipeline(pipeline_name, samples):
    completed = get_completed_indices(pipeline_name)
    output_path = RESULTS_DIR / f"{pipeline_name}.jsonl"

    pipe = pipeline(
        task=pipeline_name,
        generator_model=GENERATOR_MODEL,
        scorer_model=SCORER_MODEL,
    )

    for i, sample in enumerate(samples):
        if i in completed:
            print(f"  Skipping {pipeline_name}[{i}] (already done)")
            continue

        print(f"  Running {pipeline_name}[{i}]...")

        kwargs = {"input": sample["input"], "target": sample["target"]}
        if sample.get("reference"):
            kwargs["reference"] = sample["reference"]

        result = pipe(**kwargs)

        # Serialize result
        record = {
            "index": i,
            "pipeline": pipeline_name,
            "checklist": {
                "items": [
                    {
                        "id": item.id,
                        "question": item.question,
                        "weight": item.weight,
                        "category": item.category,
                    }
                    for item in result.checklist.items
                ],
                "source_method": result.checklist.source_method,
            },
            "score": {
                "item_scores": [
                    {
                        "item_id": s.item_id,
                        "answer": s.answer,
                        "reasoning": s.reasoning,
                        "confidence": s.confidence,
                        "confidence_level": s.confidence_level,
                    }
                    for s in result.score.item_scores
                ],
                "total_score": result.score.total_score,
                "weighted_score": result.score.weighted_score,
                "normalized_score": result.score.normalized_score,
                "primary_metric": result.score.primary_metric,
                "primary_score": result.score.primary_score,
                "scoring_method": result.score.scoring_method,
            },
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} samples")

    for pname in PIPELINES:
        print(f"\nPipeline: {pname}")
        run_pipeline(pname, samples)

    print("\nDone! Results in results/")

if __name__ == "__main__":
    main()
```

**Step 2: Run the pipelines**

```bash
cd ../autocl-demo
uv run python run_pipelines.py
```

This takes a few minutes per pipeline (LLM API calls). If interrupted, re-run to resume.

**Step 3: Verify results**

```bash
for f in ../autocl-demo/results/*.jsonl; do echo "$f: $(wc -l < $f) results"; done
```

Each file should have 2-3 lines (one per sample).

**Step 4: Commit**

```bash
cd ../autocl-demo
git add run_pipelines.py results/
git commit -m "add pipeline runner script with results"
```

---

## Task 4: Write export script

**Files:**
- Create: `../autocl-demo/export_for_web.py`
- Create: `../autocl-demo/web_data.json` (output — also copied to website repo)

Transforms raw JSONL results into a single JSON blob structured for the web page.

**Step 1: Write the script**

Output structure:
```json
{
  "examples": [
    {
      "id": "example-0",
      "label": "Multi-constraint Instruction",
      "input": "...",
      "target": "...",
      "reference": "..." ,
      "source": "InFoBench",
      "pipelines": {
        "tick": {
          "name": "TICK",
          "description": "Few-shot checklist generation from input text",
          "generator_type": "Direct",
          "uses_reference": false,
          "score": { "passed": 7, "total": 8, "percentage": 87.5, "primary_metric": "pass" },
          "items": [
            {
              "question": "...",
              "passed": true,
              "reasoning": "...",
              "confidence": null,
              "confidence_level": null,
              "weight": 100.0
            }
          ]
        }
      }
    }
  ]
}
```

```python
"""Export pipeline results as a single JSON for the web examples page."""
import json
from pathlib import Path

SAMPLES_PATH = Path("samples.jsonl")
RESULTS_DIR = Path("results")
OUTPUT = Path("web_data.json")
WEBSITE_OUTPUT = Path("../autochecklist.github.io/examples-data.json")

PIPELINE_META = {
    "tick": {
        "name": "TICK",
        "description": "Few-shot checklist generation from input text",
        "generator_type": "Direct",
        "uses_reference": False,
    },
    "rocketeval": {
        "name": "RocketEval",
        "description": "Confidence-aware checklist from input + reference",
        "generator_type": "Direct",
        "uses_reference": True,
    },
    "rlcf_candidates_only": {
        "name": "RLCF Candidates",
        "description": "Contrastive generation from candidates only",
        "generator_type": "Contrastive",
        "uses_reference": False,
    },
    "rlcf_candidate": {
        "name": "RLCF Candidate",
        "description": "Contrastive generation with candidates + reference",
        "generator_type": "Contrastive",
        "uses_reference": True,
    },
}

def main():
    with open(SAMPLES_PATH) as f:
        samples = [json.loads(line) for line in f]

    results = {}
    for pipeline_name in PIPELINE_META:
        path = RESULTS_DIR / f"{pipeline_name}.jsonl"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                results[(pipeline_name, data["index"])] = data

    examples = []
    for i, sample in enumerate(samples):
        example = {
            "id": f"example-{i}",
            "label": sample.get("task_type", f"Example {i+1}"),
            "input": sample["input"],
            "target": sample["target"],
            "reference": sample.get("reference"),
            "source": sample.get("source", "Unknown"),
            "pipelines": {},
        }

        for pname, meta in PIPELINE_META.items():
            key = (pname, i)
            if key not in results:
                continue

            r = results[key]
            checklist = r["checklist"]
            score = r["score"]

            score_map = {s["item_id"]: s for s in score["item_scores"]}
            items = []
            for item in checklist["items"]:
                s = score_map.get(item["id"], {})
                items.append({
                    "question": item["question"],
                    "passed": s.get("answer") == "yes" if s.get("answer") else None,
                    "reasoning": s.get("reasoning"),
                    "confidence": s.get("confidence"),
                    "confidence_level": s.get("confidence_level"),
                    "weight": item["weight"],
                })

            passed = sum(1 for it in items if it["passed"])
            total = len(items)
            primary_score = score.get("primary_score")

            example["pipelines"][pname] = {
                **meta,
                "score": {
                    "passed": passed,
                    "total": total,
                    "percentage": round(primary_score * 100, 1) if primary_score else round(passed / total * 100, 1) if total > 0 else 0,
                    "primary_metric": score.get("primary_metric", "pass"),
                },
                "items": items,
            }

        examples.append(example)

    web_data = {"examples": examples}

    for path in [OUTPUT, WEBSITE_OUTPUT]:
        with open(path, "w") as f:
            json.dump(web_data, f, indent=2)
        print(f"Wrote {path}")

if __name__ == "__main__":
    main()
```

**Step 2: Run the export**

```bash
cd ../autocl-demo
uv run python export_for_web.py
```

Verify: `cat web_data.json | python -m json.tool | head -20`

**Step 3: Commit in both repos**

```bash
cd ../autocl-demo
git add export_for_web.py web_data.json
git commit -m "add web export script"

cd ../autochecklist.github.io
git add examples-data.json
git commit -m "add precomputed examples data"
```

---

## Task 5: Build examples page CSS

**Files:**
- Create: `examples.css` (in website repo root)

Styles for the examples page that recreate the Compare Page look. Extends the existing `styles.css` design system.

**Step 1: Write examples.css**

Add missing CSS variables from the local UI that the website doesn't have yet:

```css
:root {
  /* Additional tokens from Compare Page UI (supplement styles.css) */
  --success: #16A34A;
  --success-light: #F0FDF4;
  --error: #DC2626;
  --error-light: #FEF2F2;
  --warning: #D97706;
  --warning-light: #FFFBEB;
  --surface-elevated: #fafafa;
}
```

Key CSS classes to implement (reference local UI's Tailwind classes):

- `.example-tabs` — horizontal row of pill buttons, `gap: 0.5rem`, flex-wrap
- `.example-tab` — pill button: `font-family: var(--font-mono)`, `font-size: 0.75rem`, uppercase, `border-radius: 6px`, `padding: 0.375rem 0.75rem`, `border: 1px solid var(--border-strong)`, `background: white`, `color: var(--text-secondary)`. Active: `background: var(--accent)`, `color: white`, `border-color: var(--accent)`
- `.example-inputs` — vertical stack of input display blocks, `gap: 1rem`, `margin: 1.5rem 0`
- `.example-input` — `background: var(--surface-elevated)`, `border-radius: 8px`, `padding: 1rem`
- `.example-input-label` — `font-family: var(--font-mono)`, `font-size: 0.7rem`, uppercase, `color: var(--text-muted)`, `letter-spacing: 0.05em`, `margin-bottom: 0.5rem`
- `.example-input-text` — `font-family: var(--font-body)`, `font-size: 0.9rem`, `line-height: 1.6`, `color: var(--text)`, `white-space: pre-wrap`
- `.method-cards` — `display: flex`, `gap: 1rem`, `overflow-x: auto`, `padding-bottom: 1rem`, `scroll-snap-type: x mandatory`
- `.method-card` — `min-width: 280px`, `max-width: 340px`, `flex-shrink: 0`, `background: var(--surface)`, `border: 1px solid var(--border-strong)`, `border-radius: 12px`, `scroll-snap-align: start`
- `.method-card-header` — `padding: 1.25rem 1.25rem 1rem`, `border-bottom: 1px solid var(--border)`
- `.method-card-header-top` — `display: flex`, `justify-content: space-between`, `align-items: center`, `margin-bottom: 0.25rem`
- `.method-card-type` — `font-family: var(--font-mono)`, `font-size: 0.6875rem`, `color: var(--text-muted)`
- `.method-card-badge` — same as local UI Badge: `font-family: var(--font-mono)`, `font-size: 0.6875rem`, `font-weight: 500`, `text-transform: uppercase`, `letter-spacing: 0.05em`, `padding: 0.125rem 0.375rem`, `border-radius: 6px`
- `.badge-success` — `background: var(--success-light)`, `color: var(--success)`
- `.method-card-name` — `font-family: var(--font-mono)`, `font-size: 1rem`, `font-weight: 600`, `color: var(--text)`
- `.method-card-description` — `font-size: 0.75rem`, `color: var(--text-muted)`, `margin-top: 0.125rem`
- `.method-card-body` — `padding: 1rem`
- `.method-card-score` — `display: flex`, `justify-content: space-between`, `align-items: center`, `padding: 0.75rem`, `border-radius: 6px`, `background: var(--surface-elevated)`
- `.method-card-score-label` — `font-size: 0.875rem`, `color: var(--text-secondary)`
- `.method-card-score-right` — `display: flex`, `flex-direction: column`, `align-items: flex-end`
- `.method-card-score-value` — `font-family: var(--font-mono)`, `font-size: 1.25rem`, `font-weight: 600`, `color: var(--accent)`
- `.method-card-score-metric` — `font-family: var(--font-mono)`, `font-size: 0.75rem`, `color: var(--text-muted)`
- `.method-card-divider` — `border: none`, `border-top: 1px solid var(--border)`, `margin: 0.75rem 0`
- `.checklist-header` — `font-size: 0.6875rem`, `text-transform: uppercase`, `letter-spacing: 0.05em`, `font-weight: 500`, `color: var(--text-muted)`, `margin-bottom: 0.75rem`
- `.checklist-items` — `list-style: none`, `display: flex`, `flex-direction: column`, `gap: 0.5rem`
- `.checklist-item` — `display: flex`, `align-items: flex-start`, `gap: 0.5rem`
- `.checklist-indicator` — `flex-shrink: 0`, `width: 20px`, `height: 20px`, `display: flex`, `align-items: center`, `justify-content: center`, `border-radius: 4px`, `font-size: 0.75rem`, `font-family: var(--font-mono)`, `font-weight: 700`, `margin-top: 0.125rem`
- `.checklist-indicator.pass` — `background: var(--success-light)`, `color: var(--success)`
- `.checklist-indicator.fail` — `background: var(--error-light)`, `color: var(--error)`
- `.checklist-indicator.neutral` — `background: var(--surface-elevated)`, `color: var(--text-muted)`
- `.checklist-item-content` — `flex: 1`, `min-width: 0`
- `.checklist-question` — `font-size: 0.875rem`, `line-height: 1.5`, `color: var(--text)`
- `.checklist-reasoning` — `font-size: 0.75rem`, `color: var(--text-muted)`, `margin-top: 0.25rem`, `line-height: 1.5`
- `.checklist-weight` — `font-family: var(--font-mono)`, `font-size: 0.75rem`, `color: var(--text-muted)`
- `.checklist-confidence` — `font-size: 0.75rem`

Responsive (at `max-width: 768px`):
- `.method-cards` — `flex-direction: column`
- `.method-card` — `min-width: unset`, `max-width: unset`, `width: 100%`
- `.example-tabs` — allow wrapping

**Step 2: Commit**

```bash
git add examples.css
git commit -m "add examples page styles mirroring Compare Page UI"
```

---

## Task 6: Build examples.html

**Files:**
- Create: `examples.html` (in website repo root)

Static HTML page that loads `examples-data.json` and renders the Compare Page UI. All data is precomputed and self-generated (not user input), sanitized via `escapeHtml()` before DOM insertion.

**Step 1: Write examples.html**

Structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Same meta/fonts as index.html -->
  <title>Examples — AutoChecklist</title>
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="examples.css">
</head>
<body>
  <!-- Nav (same as index.html, with "Examples" link added) -->
  <nav class="nav">...</nav>

  <section class="examples-hero">
    <div class="container">
      <p class="section-label">Pipeline Comparison Examples</p>
      <p class="section-subtitle">See how different evaluation pipelines generate
        and score checklists on real benchmark tasks.</p>
    </div>
  </section>

  <hr class="divider">

  <section class="examples-content">
    <div class="container">
      <div class="example-tabs" id="example-tabs">
        <!-- JS populates tab buttons -->
      </div>
      <div id="example-display">
        <div class="example-inputs" id="example-inputs">
          <!-- JS populates input/target/reference blocks -->
        </div>
        <div class="method-cards" id="method-cards">
          <!-- JS populates method cards -->
        </div>
      </div>
    </div>
  </section>

  <footer class="footer">...</footer>
</body>
</html>
```

JS rendering logic (inline `<script>` at bottom):

1. `init()` — fetch `examples-data.json`, call `renderTabs()` and `renderExample(0)`
2. `renderTabs()` — create tab buttons from `examplesData.examples[].label`
3. `selectExample(index)` — toggle active tab class, call `renderExample()`
4. `renderExample(index)` — call `renderInputs()` and `renderMethodCards()`
5. `renderInputs(ex)` — render Input, Target, Reference blocks
6. `renderMethodCards(ex)` — render 4 cards in pipeline order: tick, rocketeval, rlcf_candidates_only, rlcf_candidate
7. `renderCard(pipeline)` — render header (type label, badge, name, description), score box (fraction + optional metric%), checklist items
8. `renderChecklistItem(item, pipeline)` — render pass/fail indicator, question text, reasoning, weight (for RLCF), confidence label with color
9. `getConfidenceLabel(confidence)` — 5-band mapping matching local UI: >0.8 "High confidence" green, >0.6 "Moderate confidence" green, >0.4 "Unsure" muted, >0.2 "Moderate confidence" red, else "High confidence" red
10. `escapeHtml(str)` — sanitize &, <, >, " characters

**Step 2: Test locally**

```bash
cd /data/karen/checklist_repos/autochecklist.github.io
python -m http.server 8080
```

Open `http://localhost:8080/examples.html` — verify tabs switch, cards render, pass/fail indicators show correctly.

**Step 3: Commit**

```bash
git add examples.html
git commit -m "add examples page with tabbed pipeline comparison UI"
```

---

## Task 7: Homepage integration

**Files:**
- Modify: `index.html:15-20` (nav links)
- Modify: `index.html:127-131` (examples placeholder section)

**Step 1: Add "Examples" to nav**

In `index.html`, find `<ul class="nav-links">` and add before "Docs":
```html
<li><a href="examples.html">Examples</a></li>
```

**Step 2: Replace examples placeholder**

Find the "Examples coming soon" `<div class="demo-slot">` and replace its content with a teaser + link:
- Title: "Pipeline Comparison Examples"
- Subtitle: "See how TICK, RocketEval, and RLCF evaluate the same tasks"
- Make it a link to `examples.html`
- Change `border-style` from dashed to solid

**Step 3: Commit**

```bash
git add index.html
git commit -m "add examples nav link and teaser on homepage"
```

---

## Task 8: Visual polish and responsive testing

**Files:**
- Modify: `examples.css` (adjustments)
- Modify: `examples.html` (tweaks)

**Step 1: Test responsive breakpoints**

Open `examples.html` at various widths:
- Desktop (1080px+): 4 cards in a row, horizontal scroll
- Tablet (768px): cards should scroll horizontally or wrap to 2x2
- Mobile (480px): cards stack vertically, full width

**Step 2: Adjust CSS as needed**

Fix any overflow, spacing, or readability issues. Ensure:
- Cards don't overflow viewport
- Tab buttons wrap on narrow screens
- Input/target text blocks are readable
- Pass/fail indicators are clear at small sizes

**Step 3: Verify visual match with local UI**

Compare side-by-side with the local UI's Compare Page. Adjust colors, spacing, font sizes to match closely.

**Step 4: Commit**

```bash
git add examples.css examples.html
git commit -m "polish examples page responsive layout and visual alignment"
```

---

## Dependencies

```
Task 1 (setup) → Task 2 (samples) → Task 3 (pipelines) → Task 4 (export)
                                                                ↓
Task 5 (CSS) ──────────────────────────────────────────→ Task 6 (HTML)
                                                                ↓
                                                         Task 7 (homepage)
                                                                ↓
                                                         Task 8 (polish)
```

- Tasks 1-4 are sequential (each depends on the previous)
- Task 5 can run in parallel with Tasks 1-4 (CSS doesn't need data)
- Task 6 depends on Tasks 4 and 5 (needs both data and CSS)
- Task 7 depends on Task 6
- Task 8 depends on Tasks 6 and 7
