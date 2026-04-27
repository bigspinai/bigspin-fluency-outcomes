"""
Task Complexity Tagger

Tags chat transcripts with an assessment of the complexity of the user's task,
based on the transcript of their interaction with an AI service.

Usage:
    python tag_task_complexity.py submit  input.csv --job-name complexity
    python tag_task_complexity.py status  complexity
    python tag_task_complexity.py results complexity -o task_complexity.csv
    python tag_task_complexity.py run     input.csv --sample 20

Note: This tagger works on the transcript_summary field if available
(from UX signal output), falling back to the full transcript.

Requires:
    pip install anthropic openai
    Set ANTHROPIC_API_KEY or OPENAI_API_KEY env var.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# Added by Chris to handle some looooooong transcripts:
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a data analyst assessing the complexity of a user's task based on a chat transcript (or its summary) between a user and an AI service.

## Your Task

Read the provided chat transcript (or its summary) and assess how complex the user's underlying task is. Focus on the *task the user is trying to accomplish*, not on how well the AI responded or how long the conversation is.

## What "Complexity" Means

Task complexity is a multi-dimensional property. Consider each of these dimensions:

1. **Cognitive complexity** — How much reasoning, inference, or synthesis is required? Does the task require chaining multiple steps of logic, weighing trade-offs, or integrating information from multiple sources?

2. **Domain expertise** — How specialized is the knowledge required? A question answerable from general knowledge is low; one requiring expert-level knowledge of a narrow field (e.g., tax law in a specific jurisdiction, a niche programming framework, clinical medicine) is high.

3. **Scope / size** — How large is the deliverable or the space of things to be considered? A one-line answer is small; a multi-file refactor, a long document, or a plan spanning many entities is large.

4. **Ambiguity / under-specification** — How much must the assistant infer, disambiguate, or ask clarifying questions? Well-specified tasks are simpler; vague or open-ended ones are more complex.

5. **Constraints / dependencies** — Are there many interacting constraints (style, format, audience, prior context, external systems, conflicting goals) that must all be satisfied simultaneously?

6. **Novelty / creativity** — Does the task require generating genuinely novel content or ideas, versus retrieving or restating known information?

## Overall Complexity Rating

Assign an overall complexity level on a 5-point scale:

- **`trivial`** — A quick factual lookup, a simple definition, a one-line answer, a greeting, or a basic command. Essentially no reasoning required. (e.g., "What's the capital of France?", "Convert 5 miles to km".)

- **`low`** — A single, well-specified task that a knowledgeable non-expert could handle in a minute or two. Minimal reasoning, narrow scope, common knowledge. (e.g., "Rewrite this sentence more formally", "What does this error message mean?".)

- **`moderate`** — A multi-step task or one requiring some domain knowledge, but still bounded and reasonably well-specified. Requires thought but no deep expertise or large synthesis. (e.g., "Debug this short function", "Draft a short cover letter for this role", "Summarize this article and pull out the three main arguments".)

- **`high`** — Substantial task requiring domain expertise, multi-step reasoning, synthesis across sources, or handling real ambiguity. A knowledgeable professional would need to think carefully. (e.g., "Design a database schema for X", "Analyze this contract for risks", "Write a research-quality literature review on topic Y".)

- **`expert`** — Genuinely hard, open-ended, or high-stakes work that demands deep expertise, creative synthesis, or careful handling of many interacting constraints. A specialist might spend significant effort on it. (e.g., "Architect a distributed system that meets these constraints", "Produce a novel mathematical proof", "Draft a full regulatory filing").

When in doubt between two adjacent levels, pick the lower one and note your uncertainty in `confidence`.

## Output Format

Return a JSON object:

```json
{
  "overall_complexity": "moderate",
  "complexity_score": 3,
  "cognitive_complexity": "moderate",
  "domain_expertise": "low",
  "scope": "small",
  "ambiguity": "low",
  "constraints": "low",
  "novelty": "low",
  "task_type": "code_debugging",
  "task_brief": "Debugging a short Python function that throws a TypeError on list input",
  "key_complexity_drivers": ["requires tracing control flow", "mild under-specification of expected behavior"],
  "would_challenge_nonexpert": "yes",
  "requires_specialized_knowledge": "no",
  "multi_step_reasoning": "yes",
  "confidence": "high",
  "notes": "Task is bounded and the user provided the code; complexity is mostly in the reasoning, not the scope."
}
```

**Field definitions:**
- `overall_complexity`: One of `trivial`, `low`, `moderate`, `high`, `expert`.
- `complexity_score`: Integer 1–5 mapping to overall_complexity (1=trivial, 2=low, 3=moderate, 4=high, 5=expert). Provided for easy numeric analysis.
- `cognitive_complexity`: `low` / `moderate` / `high` — reasoning and synthesis required.
- `domain_expertise`: `low` / `moderate` / `high` — specialized knowledge required.
- `scope`: `small` / `medium` / `large` — size of the deliverable or problem space.
- `ambiguity`: `low` / `moderate` / `high` — how under-specified the task is.
- `constraints`: `low` / `moderate` / `high` — density of interacting constraints.
- `novelty`: `low` / `moderate` / `high` — how much genuine novelty/creativity is required.
- `task_type`: A short snake_case tag for the kind of task (e.g., `code_debugging`, `code_writing`, `factual_qa`, `summarization`, `creative_writing`, `translation`, `data_analysis`, `planning`, `explanation`, `math_problem`, `drafting_document`, `editing`, `advice`, `brainstorming`, `research_synthesis`, `tutoring`, `roleplay`, `other`). Pick whatever best captures the task — do not force a match.
- `task_brief`: One sentence describing what the user is specifically trying to accomplish (not just the category — be specific).
- `key_complexity_drivers`: A short list (0–4 items) of concrete factors that make this task more or less complex. Use natural-language phrases.
- `would_challenge_nonexpert`: `yes` / `no` / `ambiguous` — would an intelligent non-expert struggle with this task?
- `requires_specialized_knowledge`: `yes` / `no` / `ambiguous` — does it require expertise in a specific domain?
- `multi_step_reasoning`: `yes` / `no` / `ambiguous` — does completing the task require chaining multiple reasoning steps?
- `confidence`: `high` / `medium` / `low` — your confidence in the overall_complexity rating.
- `notes`: One short sentence of free-form explanation justifying the overall rating, especially if it was a close call.

## Guidelines

1. **Rate the task, not the outcome.** A simple task poorly handled by the AI is still a simple task. A complex task the AI nailed is still complex.
2. **Rate what the user *asked for*, not everything the conversation touched.** If the user asked a simple follow-up after a long context, rate the actual task.
3. **Multi-turn conversations:** assess the dominant task. If the user is incrementally building one deliverable (e.g., iterating on a document), treat it as one task at its overall difficulty. If the turns are genuinely separate unrelated tasks, rate the most demanding one and mention this in `notes`.
4. **Be calibrated.** Most everyday questions are `low` or `moderate`. `expert` should be rare and reserved for genuinely demanding work.
5. **Ignore prompt-engineering artifacts.** Long system prompts or elaborate formatting don't by themselves make a task complex; the underlying task does.
6. **Output valid JSON only.** No commentary, no markdown, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
#DEFAULT_MODEL = "claude-opus-4-6"

OUTPUT_FIELDS = [
    "conversation_id",
    "model",
    "overall_complexity",
    "complexity_score",
    "cognitive_complexity",
    "domain_expertise",
    "scope",
    "ambiguity",
    "constraints",
    "novelty",
    "task_type",
    "task_brief",
    "key_complexity_drivers",
    "would_challenge_nonexpert",
    "requires_specialized_knowledge",
    "multi_step_reasoning",
    "confidence",
    "notes",
    "full_response_json",
    "input_text",
]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _load_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"Error: {p} not found.", file=sys.stderr)
        sys.exit(1)
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_input_text(row: dict) -> str:
    """Use transcript_summary if available (cheaper), else full transcript."""
    if row.get("transcript_summary"):
        return row["transcript_summary"]
    if row.get("transcript"):
        return str(row["transcript"])
    # Fallback: use whatever text fields exist
    return str(row)


def _get_conversation_id(row: dict) -> str:
    return str(row.get("conversation_id", row.get("id", "")))


def _sample_rows(rows, n, seed):
    if n is None:
        return rows
    n = min(n, len(rows))
    random.seed(seed)
    return random.sample(rows, n)


def _fmt_list(v):
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return str(v) if v is not None else ""


def _result_row(cid, input_text, model, tags):
    return {
        "conversation_id": cid,
        "model": model,
        "overall_complexity": tags.get("overall_complexity", ""),
        "complexity_score": tags.get("complexity_score", ""),
        "cognitive_complexity": tags.get("cognitive_complexity", ""),
        "domain_expertise": tags.get("domain_expertise", ""),
        "scope": tags.get("scope", ""),
        "ambiguity": tags.get("ambiguity", ""),
        "constraints": tags.get("constraints", ""),
        "novelty": tags.get("novelty", ""),
        "task_type": tags.get("task_type", ""),
        "task_brief": tags.get("task_brief", ""),
        "key_complexity_drivers": _fmt_list(tags.get("key_complexity_drivers", "")),
        "would_challenge_nonexpert": tags.get("would_challenge_nonexpert", ""),
        "requires_specialized_knowledge": tags.get("requires_specialized_knowledge", ""),
        "multi_step_reasoning": tags.get("multi_step_reasoning", ""),
        "confidence": tags.get("confidence", ""),
        "notes": tags.get("notes", ""),
        "full_response_json": json.dumps(tags),
        "input_text": input_text,
    }


def _write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(results):
    from collections import Counter

    levels = [r["overall_complexity"] for r in results if r["overall_complexity"]]
    if levels:
        order = ["trivial", "low", "moderate", "high", "expert"]
        counts = Counter(levels)
        print("\nOverall complexity distribution:")
        for v in order:
            c = counts.get(v, 0)
            if c == 0:
                continue
            pct = c / len(levels) * 100
            bar = "█" * int(pct / 2)
            print(f"  {v:10s} {c:5d} ({pct:5.1f}%) {bar}")
        # Anything unexpected
        for v, c in counts.items():
            if v not in order:
                pct = c / len(levels) * 100
                print(f"  {v:10s} {c:5d} ({pct:5.1f}%) [unexpected]")

    scores = []
    for r in results:
        try:
            s = int(r["complexity_score"])
            scores.append(s)
        except (ValueError, TypeError):
            pass
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nMean complexity score: {avg:.2f} (n={len(scores)})")

    task_types = [r["task_type"] for r in results if r["task_type"]]
    if task_types:
        tcounts = Counter(task_types)
        print("\nTop task types:")
        for t, c in tcounts.most_common(10):
            pct = c / len(task_types) * 100
            print(f"  {t:30s} {c:5d} ({pct:5.1f}%)")

    conf = [r["confidence"] for r in results if r["confidence"]]
    if conf:
        ccounts = Counter(conf)
        print(
            f"\nConfidence: "
            f"{ccounts.get('high',0)} high, "
            f"{ccounts.get('medium',0)} medium, "
            f"{ccounts.get('low',0)} low"
        )


# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------

def cmd_submit(args):
    from anthropic import Anthropic

    rows = _load_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows from {args.input_csv}")

    sample = _sample_rows(rows, args.sample, args.seed)

    if args.exclude_from:
        exclude_path = Path(args.exclude_from)
        if not exclude_path.exists():
            print(f"Error: {exclude_path} not found.", file=sys.stderr)
            sys.exit(1)
        with open(exclude_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            done_ids = {
                r["conversation_id"]
                for r in reader
                if r.get("overall_complexity")
            }
        before = len(sample)
        sample = [r for r in sample if _get_conversation_id(r) not in done_ids]
        print(f"Excluded {before - len(sample)} already-tagged (from {exclude_path})")
        print(f"Remaining: {len(sample)}")
        if not sample:
            print("Nothing to submit — all transcripts already tagged!")
            return

    model = args.model or DEFAULT_MODEL
    chunk_size = args.chunk_size

    print(f"Total: {len(sample)} | Chunk size: {chunk_size}")
    num_chunks = (len(sample) + chunk_size - 1) // chunk_size
    print(f"Will submit {num_chunks} batch(es) with {model}...")

    all_requests = []
    for row in sample:
        all_requests.append({
            "custom_id": _get_conversation_id(row),
            "params": {
                "model": model,
                "max_tokens": 800,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": _get_input_text(row)}],
            },
        })

    client = Anthropic()
    job_name = args.job_name or f"complexity_{int(time.time())}"
    meta_path = Path(f"job_{job_name}.json")

    if meta_path.exists() and not args.fresh:
        with open(meta_path) as f:
            meta = json.load(f)
        batch_ids = meta.get("batch_ids", [])
        start_chunk = len(batch_ids)
        print(f"\n  Resuming '{job_name}' — {start_chunk} chunk(s) already submitted.")
    else:
        batch_ids = []
        start_chunk = 0

    def _save_meta():
        meta = {
            "job_name": job_name,
            "model": model,
            "num_requests": len(all_requests),
            "num_batches": len(batch_ids),
            "batch_ids": batch_ids,
            "input_csv": args.input_csv,
            "chunk_size": chunk_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tagger": "task_complexity",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    max_retries = 3
    for i in range(start_chunk * chunk_size, len(all_requests), chunk_size):
        chunk = all_requests[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        print(f"\n  Chunk {chunk_num}/{num_chunks} ({len(chunk)} requests)...", end=" ", flush=True)

        for attempt in range(1, max_retries + 1):
            try:
                batch = client.messages.batches.create(requests=chunk)
                batch_ids.append(batch.id)
                print(f"✓ {batch.id}")
                _save_meta()
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * attempt
                    print(f"\n    ✗ Attempt {attempt}: {e}\n    Retrying in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"\n    ✗ All {max_retries} attempts failed: {e}")
                    _save_meta()
                    print(f"\n  Partial job saved. Resume with: python tag_task_complexity.py submit {args.input_csv} --job-name {job_name}")
                    sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Job: {job_name} | {len(batch_ids)} batches | {len(all_requests)} requests")
    print(f"{'='*60}")
    print(f"\n  Status:  python tag_task_complexity.py status  {job_name}")
    print(f"  Results: python tag_task_complexity.py results {job_name} -o task_complexity.csv")


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

def _load_job_meta(job_id):
    meta_path = Path(f"job_{job_id}.json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"batch_ids": [job_id], "job_name": job_id}


def cmd_status(args):
    from anthropic import Anthropic

    meta = _load_job_meta(args.job_id)
    batch_ids = meta["batch_ids"]
    client = Anthropic()

    total_all = done_all = succeeded_all = errored_all = 0
    all_ended = True

    print(f"Job: {meta.get('job_name', args.job_id)} ({len(batch_ids)} batch(es))\n")
    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        rc = batch.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
        done = rc.succeeded + rc.errored
        pct = (done / total * 100) if total > 0 else 0
        icon = "✓" if batch.processing_status == "ended" else "⏳"
        print(f"  {icon} {bid}  {done:>6}/{total:<6} ({pct:5.1f}%)  {batch.processing_status}")
        total_all += total
        done_all += done
        succeeded_all += rc.succeeded
        errored_all += rc.errored
        if batch.processing_status != "ended":
            all_ended = False

    pct_all = (done_all / total_all * 100) if total_all > 0 else 0
    print(f"\nOverall: {done_all}/{total_all} ({pct_all:.1f}%) — {succeeded_all} succeeded, {errored_all} errored")

    if all_ended:
        print(f"\nDone! python tag_task_complexity.py results {args.job_id} -o task_complexity.csv")

    if args.wait and not all_ended:
        print(f"\nPolling every {args.poll}s...")
        while not all_ended:
            time.sleep(args.poll)
            done_all = 0
            all_ended = True
            for bid in batch_ids:
                batch = client.messages.batches.retrieve(bid)
                rc = batch.request_counts
                done_all += rc.succeeded + rc.errored
                if batch.processing_status != "ended":
                    all_ended = False
            print(f"  [{time.strftime('%H:%M:%S')}] {done_all}/{total_all} ({done_all/total_all*100:.1f}%)")
        print("All batches complete!")


# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

def cmd_results(args):
    from anthropic import Anthropic

    meta = _load_job_meta(args.job_id)
    batch_ids = meta["batch_ids"]
    model = meta.get("model", "unknown")
    client = Anthropic()

    not_done = [bid for bid in batch_ids if client.messages.batches.retrieve(bid).processing_status != "ended"]
    if not_done:
        if not args.wait:
            print(f"{len(not_done)}/{len(batch_ids)} still processing. Use --wait or check later.")
            sys.exit(1)
        print(f"Waiting (every {args.poll}s)...")
        while not_done:
            time.sleep(args.poll)
            not_done = [bid for bid in not_done if client.messages.batches.retrieve(bid).processing_status != "ended"]
            print(f"  [{time.strftime('%H:%M:%S')}] {len(batch_ids)-len(not_done)}/{len(batch_ids)} complete")

    # Load input texts
    input_texts = {}
    if meta.get("input_csv"):
        try:
            for row in _load_csv(meta["input_csv"]):
                cid = _get_conversation_id(row)
                input_texts[cid] = _get_input_text(row)
        except Exception:
            pass

    results = []
    errors = 0
    for batch_num, bid in enumerate(batch_ids, 1):
        print(f"Downloading {batch_num}/{len(batch_ids)} ({bid})...", end=" ", flush=True)
        count = 0
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            input_text = input_texts.get(cid, "")
            if result.result.type == "succeeded":
                try:
                    tags = _parse_json(result.result.message.content[0].text)
                except Exception:
                    tags = {"error": "parse_error"}
                    errors += 1
            else:
                tags = {"error": result.result.type}
                errors += 1
            results.append(_result_row(cid, input_text, model, tags))
            count += 1
        print(f"✓ {count} rows")

    _write_csv(results, Path(args.output))
    print(f"\n{len(results)} results ({errors} errors) → {args.output}")
    _print_summary(results)


# ---------------------------------------------------------------------------
# RUN (synchronous)
# ---------------------------------------------------------------------------

def cmd_run(args):
    rows = _load_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows")

    sample = _sample_rows(rows, args.sample, args.seed)
    model = args.model

    if args.provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic()
        model = model or DEFAULT_MODEL

        def call(text):
            resp = client.messages.create(
                model=model, max_tokens=800, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            return _parse_json(resp.content[0].text)
    elif args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = model or "gpt-4o-mini"

        def call(text):
            resp = client.chat.completions.create(
                model=model, max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            return _parse_json(resp.choices[0].message.content)

    print(f"Tagging {len(sample)} with {model}...\n")
    results = []
    for i, row in enumerate(sample):
        cid = _get_conversation_id(row)
        input_text = _get_input_text(row)
        print(f"[{i+1}/{len(sample)}] {cid[:30]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            tags = call(input_text)
            print(f"✓ {tags.get('overall_complexity','')} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"✗ {e} ({time.time()-t0:.1f}s)")
            tags = {"error": str(e)}
        results.append(_result_row(cid, input_text, model, tags))

    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_complexity.csv")
    _write_csv(results, output_path)
    print(f"\nDone → {output_path}")
    _print_summary(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tag transcripts by the complexity of the user's task.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="Submit batch job")
    p.add_argument("input_csv")
    p.add_argument("--model", default=None)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--job-name", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--exclude-from", default=None)

    p = sub.add_parser("status", help="Check progress")
    p.add_argument("job_id")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=int, default=30)

    p = sub.add_parser("results", help="Download results")
    p.add_argument("job_id")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll", type=int, default=30)

    p = sub.add_parser("run", help="Tag synchronously (testing)")
    p.add_argument("input_csv")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default=None)
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", default=None)

    args = parser.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "results": cmd_results, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
