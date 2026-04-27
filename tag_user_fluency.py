"""
User AI Fluency Tagger — Batch API

Tags chat transcripts for user behaviors indicative of AI fluency,
based on Anthropic's 4D AI Fluency Framework.

Usage:
    python tag_user_fluency.py submit  cohort.csv                      # submit batch job
    python tag_user_fluency.py submit  cohort.csv --sample 500         # sample 500 rows
    python tag_user_fluency.py status  fluency_job                     # check progress
    python tag_user_fluency.py results fluency_job -o fluency.csv      # download results

Synchronous mode (testing):
    python tag_user_fluency.py run cohort.csv --sample 20

Requires:
    pip install anthropic openai python-dotenv
    Set ANTHROPIC_API_KEY in .env or as env var (or OPENAI_API_KEY for OpenAI mode).
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

# Load .env file if present (before any SDK imports)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on real env vars

# ---------------------------------------------------------------------------
# System prompt — AI Fluency Behavior Taxonomy
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert researcher analyzing chat transcripts to assess the user's AI fluency — their ability to collaborate safely and effectively with an AI assistant.

## Background

AI fluency is the set of skills that enable a person to get reliable, high-quality results from AI systems. It goes beyond basic prompting: fluent users treat AI as a thought partner, steer it actively, and evaluate its output critically. Non-fluent users tend to accept first responses passively, delegate without guardrails, or struggle to get what they need.

Your job is to identify **specific user behaviors** in the transcript that are evidence of fluency (or its absence). Focus exclusively on what the USER does, not on the AI's performance.

## Fluency Behavior Taxonomy

### Category 1: Description & Delegation — How the user frames and steers the task

**`goal_clarification`**
User clearly articulates what they want to accomplish, providing context about the purpose, audience, or constraints. Goes beyond a bare request. Look for: explaining why they need something, describing the intended use, setting scope.
- Strength 1: Minimal context beyond the bare ask, but at least some framing.
- Strength 2: Clear goal with useful context (audience, purpose, or constraints).
- Strength 3: Rich, well-structured request with explicit success criteria or detailed requirements.

**`format_specification`**
User specifies how they want the output structured or formatted. Look for: requesting bullet points vs prose, asking for a specific length, specifying code language, requesting tables, asking for a particular tone or style.
- Strength 1: Basic format hint ("make it short", "in Python").
- Strength 2: Clear formatting requirements ("500-word blog post with H2 headers").
- Strength 3: Detailed structural spec (template, schema, multi-part format requirements).

**`example_provision`**
User provides examples of what they want (or don't want) to guide the AI. Look for: sample inputs/outputs, reference texts, "like this but...", before/after examples, style references.
- Strength 1: Vague reference ("something like X").
- Strength 2: Concrete example with clear intent.
- Strength 3: Multiple examples or a detailed reference that substantially constrains the output.

**`role_or_persona_setting`**
User sets collaboration expectations by assigning the AI a role, expertise level, or communication style. Look for: "act as a...", "you are an expert in...", "explain like I'm a...", "be concise/detailed/formal".
- Strength 1: Basic tone or brevity request.
- Strength 2: Clear role assignment or expertise framing.
- Strength 3: Detailed persona with specific behavioral instructions.

**`constraint_setting`**
User proactively defines boundaries, exclusions, or rules for the AI to follow. Look for: "don't include...", "avoid...", "only consider...", "assume that...", "ignore...". Distinct from format_specification — this is about content constraints, not structure.
- Strength 1: One simple exclusion or assumption.
- Strength 2: Multiple meaningful constraints that shape the output.
- Strength 3: Comprehensive guardrails that demonstrate deep understanding of how to control AI output.

### Category 2: Iteration & Refinement — How the user steers toward better results

**`iterative_refinement`**
User builds on the AI's response to improve it, rather than accepting the first output. Look for: "now make it more...", "change X to Y", "keep the structure but...", "that's good but add...", multi-turn editing.
- Strength 1: One minor adjustment.
- Strength 2: Targeted, purposeful refinement that meaningfully improves the output.
- Strength 3: Multi-step iterative process showing sophisticated steering (3+ refinement turns).

**`decomposition`**
User breaks a complex task into smaller steps or asks the AI to work through a problem stage by stage. Look for: "first do X, then Y", "let's start with...", "step 1:", multi-message workflows, chain-of-thought requests.
- Strength 1: Implicit sequencing (one follow-up building on prior).
- Strength 2: Explicit multi-step plan or deliberate phased approach.
- Strength 3: Sophisticated orchestration — managing a complex multi-part workflow across many turns.

**`reference_injection`**
User provides external context, data, or source material mid-conversation to improve accuracy. Look for: pasting code/text/data, linking sources, providing documentation, sharing error messages, giving real examples from their domain.
- Strength 1: Brief snippet or data point.
- Strength 2: Substantial context (paragraph, code block, data set) that materially helps.
- Strength 3: Multiple injections across turns, or a rich corpus that transforms the quality of the interaction.

### Category 3: Evaluation & Discernment — How critically the user engages with output

**`reasoning_questioning`**
User asks the AI to explain its reasoning, justify a choice, or walk through its logic. Look for: "why did you...", "what's your reasoning?", "how did you arrive at...", "explain your approach", "what are the tradeoffs?".
- Strength 1: Casual "why?" or mild probing.
- Strength 2: Substantive request for justification or alternative approaches.
- Strength 3: Deep interrogation of reasoning, exploring edge cases or challenging assumptions.

**`fact_checking`**
User questions the accuracy of the AI's output or cross-references it. Look for: "is that actually true?", "I thought it was X not Y", correcting factual errors, pointing out inconsistencies, requesting sources or citations.
- Strength 1: Casual accuracy check ("are you sure?").
- Strength 2: Specific factual challenge with the user's own knowledge.
- Strength 3: Systematic verification — catching errors, demanding sources, or demonstrating that the user independently validated claims.

**`critical_output_evaluation`**
User evaluates the quality, completeness, or appropriateness of AI output without a factual dispute. Look for: "this is too vague", "you missed X", "this doesn't address my actual question", "the tone is wrong", pointing out logical gaps.
- Strength 1: Mild quality feedback ("not quite what I meant").
- Strength 2: Specific, actionable critique identifying what's missing or wrong.
- Strength 3: Comprehensive evaluation showing deep engagement (multiple quality dimensions addressed).

**`context_gap_identification`**
User recognizes that the AI is missing important context and proactively supplies it. Look for: "you're assuming X but actually...", "you don't know that I...", "important context:...", correcting the AI's framing rather than its facts.
- Strength 1: One clarification to fix an assumption.
- Strength 2: Proactive context supply that substantially redirects the conversation.
- Strength 3: Sophisticated awareness of the AI's knowledge boundaries — anticipating and correcting multiple gaps.

### Category 4: Anti-Fluency Signals — Behaviors suggesting low fluency

**`passive_acceptance`**
User accepts AI output without any evaluation, refinement, or follow-up, even when the output is mediocre, generic, or potentially wrong. Single-turn conversations that end with the AI's first response. Look for: no follow-up questions, no edits, no evaluation — the user takes whatever they get.
- Note: Only tag this for conversations where the output quality or task complexity warranted more engagement. A simple factual Q&A ("what year was X founded?") with a correct one-line answer does NOT count.

**`vague_delegation`**
User provides an extremely underspecified request and makes no effort to clarify when the AI's response doesn't quite land. Look for: "write me something about X", "help me with Y", with no context, constraints, or follow-up.
- Note: Only tag if the vagueness led to a visibly suboptimal outcome. A short prompt that happens to get a great answer is fine.

**`prompt_flailing`**
User repeatedly rephrases the same request in different ways without converging on what they actually want, suggesting they don't know how to steer the AI. Distinct from deliberate iteration: flailing is undirected, while refinement is purposeful.
- Note: Look for lack of convergence. Iteration that improves the output is `iterative_refinement`; requests that circle without progress are flailing.

**`over_trust`**
User treats AI output as authoritative without appropriate skepticism, especially for high-stakes content (medical, legal, financial, factual claims). Look for: immediately acting on unverified AI advice, using AI-generated content in contexts where accuracy matters without checking, explicitly stating high trust ("you're always right").
- Note: Absence of evaluation is `passive_acceptance`. Over-trust is active — the user signals reliance or takes consequential action.

## Output Format

Return a JSON object. **Include only behaviors that are clearly present.** Do not tag behaviors that are merely plausible — they must be evident in the transcript.

```json
{
  "transcript_summary": "One sentence describing what the user was trying to accomplish.",
  "interaction_style": "augmentative | delegative | mixed",
  "fluency_behaviors": {
    "behavior_name": {
      "strength": 2,
      "evidence": "Brief quote or paraphrase (under 30 words).",
      "turn": 3,
      "notes": "Optional context."
    }
  },
  "anti_fluency_behaviors": {
    "behavior_name": {
      "evidence": "Brief quote or paraphrase (under 30 words).",
      "notes": "Optional context."
    }
  },
  "behavior_count": 4,
  "fluency_assessment": "high | moderate | low | minimal",
  "assessment_rationale": "2-3 sentences explaining the overall fluency judgment, citing the most important behaviors observed."
}
```

**Field definitions:**
- `interaction_style`: "augmentative" if the user treats AI as a thought partner (asking questions, iterating, evaluating), "delegative" if they hand off a task and accept the result, "mixed" if both patterns are present.
- `fluency_behaviors`: Only positive fluency signals from Categories 1-3. Each with a strength score (1-3).
- `anti_fluency_behaviors`: Only signals from Category 4. No strength score — presence alone is meaningful.
- `behavior_count`: Total number of distinct behaviors tagged (both fluency and anti-fluency).
- `fluency_assessment`: Overall judgment based on the balance of fluency vs anti-fluency behaviors:
  - **high**: Multiple strong fluency behaviors across categories, especially evaluation/discernment. User actively steers and evaluates.
  - **moderate**: Some fluency behaviors present, possibly strong in one category but absent in others. User engages but doesn't fully leverage AI.
  - **low**: Few fluency behaviors, possibly with anti-fluency signals. User gets basic results but doesn't steer or evaluate.
  - **minimal**: Predominantly anti-fluency signals or a bare interaction with no fluency behaviors. User treats AI as a search engine or magic oracle.

## Calibration Guidelines

1. **Focus on the USER's behavior, not the AI's.** A great AI response doesn't make the user fluent. A bad AI response doesn't make the user non-fluent.
2. **Short transcripts are inherently limited.** A 1-2 turn conversation can demonstrate low fluency (vague_delegation + passive_acceptance) but rarely demonstrates high fluency. Don't over-infer from limited data — err toward "moderate" or "low" for short conversations unless clear signals are present.
3. **Context matters.** A simple factual question doesn't require iteration or evaluation — don't penalize simple tasks. Only tag passive_acceptance or vague_delegation when the task complexity warranted more engagement.
4. **Iteration is the strongest single signal.** Users who iterate tend to exhibit many other fluency behaviors. But iteration alone isn't sufficient for "high" — evaluation/discernment behaviors are what distinguish high fluency.
5. **Anti-fluency signals carry weight.** Even one strong anti-fluency signal (especially over_trust) should cap the assessment at "moderate" unless offset by strong positive signals.
6. **Precision over recall.** Only tag behaviors that are clearly evident. If you have to squint, don't tag it.
7. **Output valid JSON only.** No commentary, no markdown wrapping, no explanation before or after."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

OUTPUT_FIELDS = [
    "conversation_id",
    "model",
    "fluency_assessment",
    "interaction_style",
    "behavior_count",
    "assessment_rationale",
    "transcript_summary",
    "fluency_behaviors_json",
    "anti_fluency_behaviors_json",
    "full_response_json",
    "transcript",
]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _iter_csv_lines(path):
    """Yield lines from a CSV file, stripping NUL bytes."""
    with open(path, "rb") as f:
        for raw_line in f:
            yield raw_line.replace(b"\x00", b"").decode("utf-8", errors="replace")


def _load_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"Error: {p} not found.", file=sys.stderr)
        sys.exit(1)
    reader = csv.DictReader(_iter_csv_lines(p))
    if "transcript" not in reader.fieldnames or "conversation_id" not in reader.fieldnames:
        print("Error: CSV must have 'conversation_id' and 'transcript' columns.", file=sys.stderr)
        sys.exit(1)
    return list(reader)


def _sample_rows(rows, n, seed):
    if n is None:
        return rows
    n = min(n, len(rows))
    random.seed(seed)
    return random.sample(rows, n)


def _result_row(cid, transcript, model, tags):
    return {
        "conversation_id": cid,
        "model": model,
        "fluency_assessment": tags.get("fluency_assessment", ""),
        "interaction_style": tags.get("interaction_style", ""),
        "behavior_count": tags.get("behavior_count", ""),
        "assessment_rationale": tags.get("assessment_rationale", ""),
        "transcript_summary": tags.get("transcript_summary", ""),
        "fluency_behaviors_json": json.dumps(tags.get("fluency_behaviors", {})),
        "anti_fluency_behaviors_json": json.dumps(tags.get("anti_fluency_behaviors", {})),
        "full_response_json": json.dumps(tags),
        "transcript": transcript,
    }


def _write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(results):
    """Print fluency distribution and interaction style breakdown."""
    assessments = [r["fluency_assessment"] for r in results if r["fluency_assessment"]]
    if not assessments:
        return

    print("\nFluency distribution:")
    for level in ["high", "moderate", "low", "minimal"]:
        count = assessments.count(level)
        pct = count / len(assessments) * 100
        bar = "\u2588" * int(pct / 2)
        print(f"  {level:12s} {count:5d} ({pct:5.1f}%) {bar}")

    styles = [r["interaction_style"] for r in results if r["interaction_style"]]
    if styles:
        from collections import Counter
        style_counts = Counter(styles)
        print(f"\nInteraction style: {style_counts.get('augmentative',0)} augmentative, "
              f"{style_counts.get('delegative',0)} delegative, {style_counts.get('mixed',0)} mixed")

    # Behavior frequency
    all_behaviors = []
    for r in results:
        try:
            fb = json.loads(r.get("fluency_behaviors_json", "{}"))
            all_behaviors.extend(fb.keys())
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            ab = json.loads(r.get("anti_fluency_behaviors_json", "{}"))
            all_behaviors.extend(ab.keys())
        except (json.JSONDecodeError, TypeError):
            pass

    if all_behaviors:
        from collections import Counter
        behavior_counts = Counter(all_behaviors)
        print(f"\nTop behaviors (across {len(results)} transcripts):")
        for beh, count in behavior_counts.most_common(15):
            pct = count / len(results) * 100
            print(f"  {beh:35s} {count:5d} ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------

def cmd_submit(args):
    from anthropic import Anthropic

    rows = _load_csv(args.input_csv)
    print(f"Loaded {len(rows)} rows from {args.input_csv}")

    sample = _sample_rows(rows, args.sample, args.seed)

    # Auto-discover existing results to avoid re-tagging
    exclude_files = args.exclude_from
    if exclude_files is None and not args.no_exclude:
        input_dir = Path(args.input_csv).parent or Path(".")
        candidates = sorted(input_dir.glob("*_fluency*.csv"))
        if candidates:
            exclude_files = [str(c) for c in candidates]
            print(f"Auto-discovered {len(exclude_files)} existing results: {', '.join(str(c) for c in candidates)}")

    if exclude_files:
        done_ids = set()
        for ef in exclude_files:
            exclude_path = Path(ef)
            if not exclude_path.exists():
                print(f"Warning: {exclude_path} not found, skipping.", file=sys.stderr)
                continue
            reader = csv.DictReader(_iter_csv_lines(exclude_path))
            file_ids = {
                r["conversation_id"]
                for r in reader
                if r.get("fluency_assessment") in ("high", "moderate", "low", "minimal")
            }
            done_ids.update(file_ids)
            print(f"  Loaded {len(file_ids)} completed IDs from {exclude_path}")

        before = len(sample)
        sample = [r for r in sample if str(r["conversation_id"]) not in done_ids]
        print(f"Excluded {before - len(sample)} already-tagged ({len(done_ids)} unique IDs)")
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
            "custom_id": str(row["conversation_id"]),
            "params": {
                "model": model,
                "max_tokens": 1536,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": row["transcript"]}],
            },
        })

    client = Anthropic()
    job_name = args.job_name or f"fluency_{int(time.time())}"
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
            "tagger": "user_fluency",
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
                print(f"\u2713 {batch.id}")
                _save_meta()
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * attempt
                    print(f"\n    \u2717 Attempt {attempt}: {e}\n    Retrying in {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"\n    \u2717 All {max_retries} attempts failed: {e}")
                    _save_meta()
                    print(f"\n  Partial job saved. Resume: python tag_user_fluency.py submit {args.input_csv} --job-name {job_name}")
                    sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Job: {job_name} | {len(batch_ids)} batches | {len(all_requests)} requests")
    print(f"{'='*60}")
    print(f"\n  Status:  python tag_user_fluency.py status  {job_name}")
    print(f"  Results: python tag_user_fluency.py results {job_name} -o fluency.csv")


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
        icon = "\u2713" if batch.processing_status == "ended" else "\u23f3"
        print(f"  {icon} {bid}  {done:>6}/{total:<6} ({pct:5.1f}%)  {batch.processing_status}")
        total_all += total
        done_all += done
        succeeded_all += rc.succeeded
        errored_all += rc.errored
        if batch.processing_status != "ended":
            all_ended = False

    pct_all = (done_all / total_all * 100) if total_all > 0 else 0
    print(f"\nOverall: {done_all}/{total_all} ({pct_all:.1f}%) \u2014 {succeeded_all} succeeded, {errored_all} errored")

    if all_ended:
        print(f"\nDone! python tag_user_fluency.py results {args.job_id} -o fluency.csv")

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

    # Load original transcripts
    transcripts = {}
    if meta.get("input_csv"):
        try:
            for row in _load_csv(meta["input_csv"]):
                transcripts[str(row["conversation_id"])] = row["transcript"]
            print(f"Loaded transcripts from {meta['input_csv']}")
        except Exception:
            pass

    results = []
    errors = 0
    for batch_num, bid in enumerate(batch_ids, 1):
        print(f"Downloading {batch_num}/{len(batch_ids)} ({bid})...", end=" ", flush=True)
        count = 0
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            transcript = transcripts.get(cid, "")
            if result.result.type == "succeeded":
                try:
                    tags = _parse_json(result.result.message.content[0].text)
                except Exception as e:
                    tags = {"error": f"parse_error: {e}"}
                    errors += 1
            else:
                tags = {"error": result.result.type}
                errors += 1
            results.append(_result_row(cid, transcript, model, tags))
            count += 1
        print(f"\u2713 {count} rows")

    _write_csv(results, Path(args.output))
    print(f"\n{len(results)} results ({errors} errors) \u2192 {args.output}")
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

        def call(transcript):
            resp = client.messages.create(
                model=model, max_tokens=1536, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": transcript}],
            )
            return _parse_json(resp.content[0].text)
    elif args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = model or "gpt-4o-mini"

        def call(transcript):
            resp = client.chat.completions.create(
                model=model, max_tokens=1536,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            )
            return _parse_json(resp.choices[0].message.content)

    print(f"Tagging {len(sample)} with {model}...\n")
    results = []
    for i, row in enumerate(sample):
        cid = row["conversation_id"]
        transcript = row["transcript"]
        print(f"[{i+1}/{len(sample)}] {cid[:40]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            tags = call(transcript)
            elapsed = time.time() - t0
            print(f"\u2713 {tags.get('fluency_assessment', '?')} | {tags.get('behavior_count', 0)} behaviors | {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\u2717 ERROR ({elapsed:.1f}s): {e}")
            tags = {"error": str(e)}
        results.append(_result_row(cid, transcript, model, tags))

    input_path = Path(args.input_csv)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_fluency.csv")
    _write_csv(results, output_path)
    print(f"\nDone \u2192 {output_path}")
    _print_summary(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tag chat transcripts for user AI fluency behaviors."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- submit ---
    p = sub.add_parser("submit", help="Submit batch job to Anthropic Batch API")
    p.add_argument("input_csv", help="CSV with 'conversation_id' and 'transcript' columns")
    p.add_argument("--model", default=None, help=f"Model (default: {DEFAULT_MODEL})")
    p.add_argument("--sample", type=int, default=None, help="Sample N rows (default: all)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--chunk-size", type=int, default=5000, help="Requests per batch (default: 5000)")
    p.add_argument("--job-name", default=None, help="Job name for tracking")
    p.add_argument("--fresh", action="store_true", help="Ignore existing metadata, start fresh")
    p.add_argument("--exclude-from", nargs="+", default=None, metavar="CSV",
                   help="Existing results CSVs — skip already-tagged conversation_ids")
    p.add_argument("--no-exclude", action="store_true",
                   help="Disable auto-exclusion (force re-tagging)")

    # --- status ---
    p = sub.add_parser("status", help="Check job progress")
    p.add_argument("job_id", help="Job name or batch ID")
    p.add_argument("--wait", action="store_true", help="Block until complete")
    p.add_argument("--poll", type=int, default=30, help="Poll interval (seconds)")

    # --- results ---
    p = sub.add_parser("results", help="Download results to CSV")
    p.add_argument("job_id", help="Job name or batch ID")
    p.add_argument("-o", "--output", required=True, help="Output CSV path")
    p.add_argument("--wait", action="store_true", help="Wait for completion first")
    p.add_argument("--poll", type=int, default=30, help="Poll interval (seconds)")

    # --- run (synchronous) ---
    p = sub.add_parser("run", help="Tag synchronously (for testing)")
    p.add_argument("input_csv", help="CSV with 'conversation_id' and 'transcript' columns")
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default=None, help="Model override")
    p.add_argument("--sample", type=int, default=20, help="Rows to sample (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("-o", "--output", default=None, help="Output CSV path")

    args = parser.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "results": cmd_results, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
