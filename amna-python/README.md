# AI Meeting Notes Summarizer for a Logistics Startup
### Group 06 — Group Lead Piece (Amna Tariq, Python)

## What this does
Takes a raw meeting transcript (plain text with `Speaker: text` lines) and produces:
1. A short, extractive **summary** of the meeting (top-scoring sentences).
2. A list of **action items** — commitments like "I'll...", "please...", "approved...".
3. Graceful **error handling** for missing files, empty files, and unreadable input.

## Why this design
This is the *core orchestrator* piece of the group project. It's built with
zero external dependencies (pure Python) so it runs out of the box and can be
tested immediately. The `summarize_text()` function is intentionally
swappable — a teammate using the OpenAI/Anthropic API (Member 1) can later
replace the rule-based scorer with an actual LLM call, without touching the
rest of the pipeline (file loading, sentence splitting, action-item
extraction, reporting).

## How to run
```bash
# Run the demo on the sample transcript
python3 summarizer.py

# Run the test suite (11 test cases)
python3 summarizer.py --test
```

## Files
- `summarizer.py` — main script (pipeline + tests)
- `sample_meeting_transcript.txt` — sample input data (logistics team meeting)
- `README.md` — this file

## Sample output (summary)
```
1. We had a 12% improvement in average delivery time...
2. I need approval to rent two temporary vehicles for the next two weeks.
3. I'll prepare a customer communication template to proactively notify
   affected customers about delays.
4. Let's reconvene next Tuesday to review progress on all these items.
```

## Test results
11/11 test cases passing — covers valid input, missing files, empty input,
empty transcripts, summarization edge cases, and action-item detection.

## How this fits the rest of the group's project
| Member | Tool | How it plugs in |
|---|---|---|
| Amna (Lead) | Python | Core pipeline (this script) |
| Laiba | OpenAI/Anthropic API | Can replace `summarize_text()` with an LLM call for higher-quality summaries |
| Aman | LangChain | Could wrap the pipeline into a LangChain chain/agent |
| Uzma | Pandas | Could log meeting summaries + action items into a structured DataFrame/CSV for reporting |
| Mahnoor | Jupyter Notebook | Demo notebook walking through the pipeline with example outputs |
| Maleha | Hugging Face Transformers | Alternative summarization model (e.g., `facebook/bart-large-cnn`) |
| Afsah | Vector DB (FAISS/Chroma) | Store past meeting summaries for semantic search across meetings |

## Learned
- Extractive summarization via word-frequency scoring is a fast, dependency-free
  way to get a reasonable baseline summary without an API key.
- Regex-based action-item detection works decently for structured meeting
  transcripts but would need refinement (e.g., NLP-based intent detection)
  for messier real-world transcripts.
