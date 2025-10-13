# Chat CLI

The repository includes a simple terminal chat client at `cli/chat.py`.

Basic usage:

```bash
# start the chat (uses full orchestrator if deps installed, else falls back to a small builtin engine)
python cli/chat.py
```

Flags:
- `--no-fallback` : fail early if required dependencies are missing (useful for CI/integration tests)
- `--stream` : attempt to stream model output when using a streaming-capable local model (Ollama)
- `--add-doc "Some text"` : add a custom document to `data/custom_docs.json` and re-seed

Environment variables:
- `ENABLE_FAISS=1` : enable FAISS index use
- `OLLAMA_EMBED_MODEL` : Ollama embedding model name (default `qwen3-embedding:latest`)
- `VERIFIER_THRESHOLD` : set a static similarity threshold for EmbeddingVerifier (optional)

Examples:

```bash
# add a doc and run the chat using full orchestrator path
python cli/chat.py --add-doc "Christopher Nolan directed Inception." --no-fallback

# run with streaming enabled (requires Ollama with streaming support)
python cli/chat.py --stream
```
