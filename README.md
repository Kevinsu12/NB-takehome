# Market Context Generator

## How to Run

### 1. Install Dependencies

```bash
make install
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
USE_MOCK_DATA=true
SNAPSHOT_MAX_AGE=3600
DEBUG=false

# Rate Limiting (optional - defaults provided)
RATE_LIMIT_REQUESTS_PER_MINUTE=50
RATE_LIMIT_TOKENS_PER_MINUTE=40000
RATE_LIMIT_MAX_CONCURRENT=5
RATE_LIMIT_BURST=3
RATE_LIMIT_AVG_TOKENS=2000
RATE_LIMIT_MAX_RETRIES=3
RATE_LIMIT_BASE_BACKOFF=1.0

# OpenAI Configuration (optional - defaults provided)
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2000
OPENAI_TEMPERATURE=0.0
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Note**: Rate limiting is built-in with sensible defaults (50 requests/min, 40k tokens/min). The service will cache market data for 1 hour by default.

### 3. Build RAG Index

```bash
make rag
```

### 4. Start the Server

```bash
make run
```

The API will be available at `http://localhost:8000`

## Usage

### Generate Market Context Report

```bash
curl -X POST "http://localhost:8000/market-context?period=2024-Q3" | jq
```

**Response includes:**
- `formatted_context`: Human-readable market context report
- `draft_json`: Raw JSON data from the draft generation
- `retrieved_chunks`: Information about source documents used
  - `chunk_id`: Unique identifier for the document chunk
  - `source_file`: PDF file name (e.g., "quarterly_commentary_sma_all_cap_core.pdf")
  - `page_number`: Page number in the source document
  - `similarity_score`: How similar this chunk is to the query (0.0-1.0)

### Save Results to File

```bash
# Save formatted output to a file
curl -X POST "http://localhost:8000/market-context?period=2024-Q3" | jq > output_2024_Q3.json

# Save multiple periods to different files
curl -X POST "http://localhost:8000/market-context?period=2024-Q3" | jq > output_2024_Q3.json &
curl -X POST "http://localhost:8000/market-context?period=2024-Q4" | jq > output_2024_Q4.json &
curl -X POST "http://localhost:8000/market-context?period=2025-Q1" | jq > output_2025_Q1.json &
```

### Check Service Status

```bash
curl "http://localhost:8000/health" | jq
```

## Focused points
I focused on demonstrating the ability of shipping a complete, (aiming for production level), market-context service efficeintly using AI tool without compromising integrity. Moving fast only matters if the result is something you can actually run under load, monitor, and trust. That’s why I treated this as a real service instead of a notebook,focusing on async API you can hit concurrently, a retrieval layer that’s explainable, validation that constrains outputs, and visible provenance so reviewers can see where the facts came from. In short, I optimized for the way this would behave with multiple users hitting it at once, not just for a happy path demo.

## Architectural and design decisions
I organized the core logic as a small LangGraph DAG with single purpose nodes. The flow is retrieve to draft to validate to (revise if needed) to output and the flow is easy to reason about and extend. The service runs on FastAPI with an async client stack. I reuse HTTP/LLM clients and enforce safety with a sliding-window rate limiter (requests per minute and tokens per minute) plus a concurrency semaphore. That lets me take advantage of asyncio without starving other requests. Retrieval is a simple FAISS vector store over the NB PDFs with page level citations; the goal is transparency. The final narrative is rendered from a validated Pydantic object rather than from random model output, which keeps the response consistent and auditable. Configuration comes from environment variables that can be managed easily. There’s a health endpoint, lightweight tests, and defensive error handling so failures are visible and contained instead of silent.

## Key assumptions and trade-offs
I chose a straightforward FAISS index and sentence/overlap chunking because it’s predictable and easy to debug. The trade-off is that I give up some recall compared to heavier hybrid search or reranking. I favored determinism and safety over maximum creativity in generation. Token usage is estimated pre-call and corrected post-call. For the API, I kept one process and focused on non-blocking I/O, true parallel CPU work was out of scope. Finally, to keep the scope tight, I used mock market data where needed; wiring a real data feed would be the next step, not a blocker for demonstrating the architecture.

## Next step if I have more time
First, I’d replace the mock market data with a real provider using the same async and limit approach. Second, I’d upgrade the retrieval layer to a richer store, maybe more information to use outside of the past reports, a more complicated search system. Third, I further develop the validate/revise stages since both nodes right now are very minimal. I would maybe apply a stricter Pydantic constraints, soft fact checks against the retrieved pages, and a revise pass that classifies errors (missing field, unsupported claim) and applies targeted fixes instead of a generic rewrite. 


## AI tools and prompts I have used:
AI TOOLs I have used:
Cursor, Claude

Outside of the small debuggins prompts, here are some main prompts I used to construct this project:

“Draft a high level architecture for: FastAPI endpoint to LangGraph DAG nodes: retrieve to draft to validate to revise to output. Show what state flows between nodes and where I should inject tracing and logging.”

"“I’m building a FastAPI app to generate market commentary (only the Market Context section). Help me scaffold a clean repo structure with app/, clients/, nodes/, rag/, schemas/, and scripts/. Include a main.py entrypoint with a basic FastAPI server.”



system prompt for Claude:
"You are a senior Python engineer. Build a small, reproducible app that generates the “Market Context” section of quarterly commentaries.

Hard rules:
- Stack: Python 3.11+, FastAPI, LangChain+LangGraph, httpx (async), Pydantic v2, FAISS (local), pytest.
- Determinism: temperature=0, top_p=1, fixed RAG params (embedding model, chunk size, k), fixed data snapshots. Validate every numeric claim against the snapshot; fail fast on mismatch.
- Scope: Only “Market Context”—never attribution or outlook. Enforce with denylist and schema.
- Project hygiene: runnable API (`POST /market-context?period=YYYY-Q#`), `requirements.txt` or `pyproject.toml`, `README.md`, tests for clients/schema/retrieval determinism.
- No secrets in code. Read API keys from env. Provide a `.env.example`.
- Multi-file edits: print full file paths, then full file contents. Never elide code. Use fenced blocks per file.

Behavior:
- Prefer small, composable modules. Add docstrings and 1–2 unit tests per module.
- When asked to “scaffold,” output a complete minimal working repo; when asked to “iterate,” only output the changed files.
"



"In `app/clients/api_clients.py`, add async fetchers using httpx:
- `get_sp500_tr(period)`, `get_ust10y(period)`, `get_dxy(period)`, `get_vix_peak(period)`.
- create mock API and mock datas for all market datas

Use TaskGroup/gather, per-call timeout, jittered exponential backoff, and a `Cache-Control: max-age` hint to support simple on-disk caching.

In `app/nodes/ingest.py`, normalize to `data/snapshot/{period}.json`: {"sp500_tr": float, "ust10y_yield": float, "dxy_chg": float, "vix_peak": float, ...}. Add a flag to use mock data (for tests) and to pin snapshots.

Add unit tests that assert concurrency (timestamps) and that retries happen on 429/5xx."



"Fill `app/prompts/`:
- system.md: tone, banned phrases (no attribution/outlook), citation rule (“any number must come from key_stats”), target length.
- user.md: takes {period, retrieved_context, key_stats_json}, instructs: “Use only provided facts. If missing, say ‘data not available’.”
- fewshot.md: 1–2 brief exemplars based on sanitized lines from our PDFs.
- style.md: sentence length, topic order: (1) broad market move, (2) drivers, (3) rates/FX/credit, (4) earnings breadth.

Modify `app/nodes/draft.py` to call provider based on env PROVIDER; temperature=0; request JSON matching the Pydantic schema.
Modify `app/nodes/validate.py` to: (a) parse, (b) fuzzy-match numbers in narrative (tolerance=0.01), (c) enforce denylist (“overweight”, “underweight”, “we expect”, “outlook”), (d) raise on violation.
"

"Wire `app/main.py` with FastAPI to run the LangGraph for `period`, return JSON; error responses include the validation failure reason."

“I have quarterly PDF commentaries. Write a pdf_loader.py that chunks text into natural sentences with overlap. Each chunk should carry metadata (source file, page number, confidence, is_market_context flag). Use NLTK sent_tokenize.”

“Scaffold a FAISS-based vector store class in vector_store.py that can: build index from documents, save/load from disk, run similarity search with optional filters (e.g. only market context). Add deterministic mock embeddings first, later plug in OpenAI embeddings.”

“Define a MarketContext schema in schemas/market_context.py. Fields: period, headline, macro_drivers, key_stats, narrative, sources. Add strict validation.”

“Implement a token limiter for the LLMClient to avoid exceeding API quotas.
– Use an asyncio.Semaphore to cap the maximum number of concurrent requests.
– Add a per-minute rate limiter (e.g., 60 calls/minute) that delays or queues extra requests.
– Wrap both generate and get_embeddings methods with this limiter.
– Design it to be lightweight, async-friendly, and safe for FastAPI’s event loop.
– Keep the limiter reusable so it can be extended to other API clients if needed.”


“Generate a Makefile with tasks for: rag (build index), rag-test, run, test, lint, format, setup. Include scripts/build_rag.py to build/rebuild index from PDFs.”

“I want to generate a full suite of tests for my Market Context project. Please write pytest (with pytest-asyncio where needed) tests that cover:

– Schema validation: the MarketContext Pydantic schema should accept valid JSON, reject missing fields, and raise errors for invalid types.
– VectorStore: building an index from mock documents, saving and loading from disk, similarity search returning expected docs, and filtering by is_market_context and confidence_score.
– Retrieve node: should return docs from vectorstore when available, fall back to mock data if no index is loaded, and respect the top-k limit.
– Ingest node: should save a fresh snapshot if none exists, reuse an existing snapshot if valid, refresh if expired, and extract key themes correctly from documents.
– Draft node: with a stubbed LLMClient, confirm it builds prompts correctly, parses raw JSON output, and extracts fenced JSON blocks if needed.
– Validate node: confirm it passes when all key stats are present in the narrative, raises when a number is missing, and raises when prohibited phrases appear.
– Revise node: with a stubbed LLMClient, confirm it parses a revision into schema, and if parsing fails, falls back to the validated context.
– FastAPI app: using httpx.AsyncClient in test mode, confirm POST /market-context returns a valid MarketContext for a good period, invalid period format returns 400, /health reports unhealthy if pipeline not initialized, and /status includes vectorstore summary when index is built.

Write clean, deterministic tests that mock external dependencies (LLM, API clients) so tests run offline.”