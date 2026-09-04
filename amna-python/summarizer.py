"""
AI Meeting Notes Summarizer for a Logistics Startup
-----------------------------------------------------
Group 06 - Group Lead Piece (Python)
Project: AI Meeting Notes Summarizer for a Logistics Startup

This module is the CORE orchestrator script for the group project.
It takes a raw meeting transcript (plain text) and produces:
  1. A concise summary of the meeting
  2. A list of extracted action items (who needs to do what)
  3. Basic error handling for missing/invalid input

Design note for teammates:
This script uses a simple, dependency-free extractive summarization
approach (word-frequency scoring) so it runs standalone without any
API keys. It is built so that Member 1's OpenAI/Anthropic API piece
can later plug in as a drop-in replacement for `summarize_text()`
(e.g., swap it out to call an LLM instead of the rule-based scorer),
without changing the rest of the pipeline.
"""

import re
import sys
from collections import Counter


# ---------------------------------------------------------------------
# 1. INPUT HANDLING + BASIC ERROR HANDLING
# ---------------------------------------------------------------------

def load_transcript(path: str) -> str:
    """Reads a meeting transcript from a text file.
    Raises clear, user-friendly errors for common bad-input cases.
    """
    if not path.strip():
        raise ValueError("No file path was provided. Please pass a valid .txt file path.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: '{path}'. Check the path and try again.")
    except UnicodeDecodeError:
        raise ValueError("The file could not be read as text. Please provide a plain .txt transcript.")

    if not text.strip():
        raise ValueError("The transcript file is empty. Please provide a non-empty meeting transcript.")

    return text


# ---------------------------------------------------------------------
# 2. TEXT CLEANING / SENTENCE SPLITTING
# ---------------------------------------------------------------------

def clean_and_split_sentences(text: str) -> list[str]:
    """Splits transcript into sentences, stripping speaker labels like 'Ahmed:'."""
    # Remove speaker name prefixes e.g. "Sara: " at the start of a line
    text_no_speakers = re.sub(r"^[A-Za-z][A-Za-z .]{0,30}:\s*", "", text, flags=re.MULTILINE)

    # Basic sentence split on '.', '?', '!'
    raw_sentences = re.split(r"(?<=[.?!])\s+", text_no_speakers.replace("\n", " "))
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 0]
    return sentences


# ---------------------------------------------------------------------
# 3. EXTRACTIVE SUMMARIZATION (word-frequency scoring)
# ---------------------------------------------------------------------

STOPWORDS = set("""
a an the is are was were be been being to of in on for with at by from
and or but if then so as this that these those it its i we you they he she
will would can could should shall may might do does did not no yes ok okay
good morning everyone let start update thanks please
""".split())


def summarize_text(sentences: list[str], num_sentences: int = 4) -> list[str]:
    """Picks the top N most 'important' sentences using word-frequency scoring.
    This is the function a teammate could later replace with an LLM API call.
    """
    if not sentences:
        return []

    words = re.findall(r"[a-zA-Z']+", " ".join(sentences).lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    freq = Counter(words)

    scored = []
    for idx, sent in enumerate(sentences):
        sent_words = re.findall(r"[a-zA-Z']+", sent.lower())
        score = sum(freq.get(w, 0) for w in sent_words)
        # Normalize by length so long sentences don't win purely on word count
        norm_score = score / max(len(sent_words), 1)
        scored.append((norm_score, idx, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:num_sentences]
    # Keep original order for readability
    top_in_order = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return top_in_order


# ---------------------------------------------------------------------
# 4. ACTION ITEM EXTRACTION (simple rule-based)
# ---------------------------------------------------------------------

ACTION_PATTERNS = [
    r"\bI'll\b.+", r"\bI will\b.+", r"\bneed to\b.+", r"\bplease\b.+",
    r"\bwill (send|share|prepare|loop|update|approve|sign|reconvene)\b.+",
    r"\bassigned\b.+", r"\bapproved\b.+", r"\bexpect(ed)? (it )?to be resolved\b.+",
]


def extract_action_items(text: str) -> list[str]:
    """Finds lines likely to contain action items / commitments."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    action_items = []

    for line in lines:
        for pattern in ACTION_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                action_items.append(line)
                break

    # De-duplicate while preserving order
    seen = set()
    unique_items = []
    for item in action_items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return unique_items


# ---------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------

def run_pipeline(transcript_path: str, num_summary_sentences: int = 4) -> dict:
    """Runs the full summarizer pipeline and returns a structured result."""
    raw_text = load_transcript(transcript_path)
    sentences = clean_and_split_sentences(raw_text)
    summary_sentences = summarize_text(sentences, num_sentences=num_summary_sentences)
    action_items = extract_action_items(raw_text)

    return {
        "summary": summary_sentences,
        "action_items": action_items,
        "sentence_count": len(sentences),
    }


def print_report(result: dict):
    print("=" * 60)
    print("MEETING SUMMARY")
    print("=" * 60)
    for i, sent in enumerate(result["summary"], 1):
        print(f"{i}. {sent}")

    print("\n" + "=" * 60)
    print("ACTION ITEMS")
    print("=" * 60)
    if result["action_items"]:
        for i, item in enumerate(result["action_items"], 1):
            print(f"{i}. {item}")
    else:
        print("No clear action items detected.")

    print("\n" + "=" * 60)
    print(f"(Processed {result['sentence_count']} sentences total)")
    print("=" * 60)


# ---------------------------------------------------------------------
# 6. TEST CASES (run_tests) -- required: at least 10 test cases
# ---------------------------------------------------------------------

def run_tests():
    """At least 10 test cases covering normal + edge/bad-input scenarios."""
    results = []

    # 1. Normal file loads correctly
    try:
        text = load_transcript("sample_meeting_transcript.txt")
        results.append(("Load valid transcript", len(text) > 0))
    except Exception as e:
        results.append(("Load valid transcript", False, str(e)))

    # 2. Missing file raises FileNotFoundError
    try:
        load_transcript("does_not_exist.txt")
        results.append(("Missing file raises error", False))
    except FileNotFoundError:
        results.append(("Missing file raises error", True))

    # 3. Empty path raises ValueError
    try:
        load_transcript("   ")
        results.append(("Empty path raises error", False))
    except ValueError:
        results.append(("Empty path raises error", True))

    # 4. Sentence splitting produces sentences
    sample = "Ahmed: Hello team. Sara: We had a good week!"
    sents = clean_and_split_sentences(sample)
    results.append(("Sentence split removes speaker labels", "Ahmed" not in sents[0]))

    # 5. Sentence splitting handles empty text
    results.append(("Empty text -> empty sentence list", clean_and_split_sentences("") == []))

    # 6. Summarize returns fewer or equal sentences than requested
    sents2 = clean_and_split_sentences(open("sample_meeting_transcript.txt").read())
    summary = summarize_text(sents2, num_sentences=4)
    results.append(("Summary length <= requested", len(summary) <= 4))

    # 7. Summarize on empty list returns empty list
    results.append(("Summarize empty input", summarize_text([]) == []))

    # 8. Action items detected on sample transcript
    raw = open("sample_meeting_transcript.txt").read()
    items = extract_action_items(raw)
    results.append(("Action items detected (>0)", len(items) > 0))

    # 9. Action item extraction on text with no commitments
    no_action_text = "The weather was nice. We discussed general updates only."
    results.append(("No action items on neutral text", extract_action_items(no_action_text) == []))

    # 10. Full pipeline runs end-to-end without error
    try:
        result = run_pipeline("sample_meeting_transcript.txt")
        results.append(("Full pipeline runs successfully", "summary" in result and "action_items" in result))
    except Exception as e:
        results.append(("Full pipeline runs successfully", False, str(e)))

    # 11. Full pipeline on bad input fails gracefully
    try:
        run_pipeline("nonexistent_file.txt")
        results.append(("Pipeline handles bad file gracefully", False))
    except FileNotFoundError:
        results.append(("Pipeline handles bad file gracefully", True))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    passed = 0
    for r in results:
        name, ok = r[0], r[1]
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        extra = f" -> {r[2]}" if len(r) > 2 else ""
        print(f"[{status}] {name}{extra}")
    print(f"\n{passed}/{len(results)} tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        result = run_pipeline("sample_meeting_transcript.txt")
        print_report(result)
