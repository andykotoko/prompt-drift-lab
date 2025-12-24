# 05 Summary & Outlook

This folder is the **interpretive layer** of the repo: it summarizes what the saved artifacts show, clarifies **claim boundaries**, and records **future directions** without exceeding evidence.

> Note: This repo version **does NOT include** an automated perturbation generator / large-scale prompt scan system. Do not imply such tooling exists unless it is present in the repo as runnable artifacts.

---

## 0) 30-second start

- Want *evidence* (raw outputs, judged JSON, CSV tables)? → go to `04_results/`
- Want the judge contract & scoring rules? → `03_evaluation_rules/` (`EVAL_PROTOCOL.md`, `JUDGE_PROMPT.md`)
- Want generator prompts & variants? → `02_prompt_variants/` (`PROMPT_MANIFEST.md`)

---

## 1) Claim boundary (what you MAY / MAY NOT say)

### ✅ Allowed (must be traceable)
You may:
- Summarize **patterns observed in this repo’s saved outputs** (prompt variants × models × questions).
- Report aggregations that are **directly computed from saved tables** under `04_results/**/summary_tables/`.
- Describe **failure modes** using the repo’s recorded evidence (e.g., schema breaks, instruction deviation), grounded in judged outputs.
- State **limitations** of this experimental setup and what the artifacts do *not* cover.
- Propose **hypotheses / future work** explicitly labeled as *hypothesis* (not a result).

### ❌ Not allowed (non-claims)
Do NOT claim:
- General model capability or “model X is better than model Y” beyond this repo’s narrow setup.
- Safety/alignment improvements in the real world (unless explicitly measured here).
- Benchmark-level conclusions, cross-domain generalization, or external validity not supported by artifacts.
- New metrics, new judging dimensions, or new experiments that are not implemented and recorded.

---

## 2) Traceability rule (hard constraint)

Every result statement in this folder must include a **pointer to evidence**, e.g.:

- Raw outputs: `04_results/01_raw_model_outputs/...`
- Judged JSON: `04_results/02_cross_model_evaluation/...`
- Aggregated CSV: `04_results/**/summary_tables/*.csv`
- Protocol snapshot (if present): `04_results/**/used_evaluation_protocol*.md`

If a claim has **no file pointer**, rewrite it as a hypothesis or remove it.

---

## 3) What belongs here (and what does not)

### Belongs in `05_summary_and_outlook/`
- High-level summaries grounded in `04_results/`
- Methodological implications (what this setup is good/bad at)
- Explicit non-claims and limitations
- Future directions (clearly labeled, not promised)

### Does NOT belong here
- New scoring rules or judge logic → `03_evaluation_rules/`
- Prompt text / variants → `02_prompt_variants/`
- Primary result artifacts → `04_results/`

---

## 4) Recommended structure for documents in this folder

1. Evidence-backed summary (with file pointers)
2. Failure modes (with evidence snippets / pointers)
3. Limitations & non-claims
4. Implications for evaluation design
5. Outlook (extensions; clearly marked as future work)