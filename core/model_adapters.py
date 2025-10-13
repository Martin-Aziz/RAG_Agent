from typing import List, Dict, Any, Optional
import os
import asyncio
import subprocess
import time
import json
import re
from itertools import islice
try:
    from jsonschema import validate, ValidationError
except Exception:
    # lightweight fallback validator for CI environments without jsonschema
    class ValidationError(Exception):
        pass

    def validate(instance, schema):
        # very small structural validation for the planner schema used here
        if schema.get("type") == "array":
            if not isinstance(instance, list):
                raise ValidationError("expected array")
            item_schema = schema.get("items", {})
            required = item_schema.get("required", [])
            for it in instance:
                if not isinstance(it, dict):
                    raise ValidationError("items must be objects")
                for r in required:
                    if r not in it:
                        raise ValidationError(f"item missing required field: {r}")
from core.observability import LLM_CALLS


class SLMStub:
    """Deterministic small language model stub for planning and rewriting.

    Provides both sync and async methods so orchestrator can call either.
    """

    def generate_answer(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        query_lc = (query or "").lower()
        greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
        if any(greet in query_lc for greet in greetings):
            return "Hello! I'm here to help you explore the knowledge base. Feel free to ask me anything about your documents."  # noqa: E501

        friendly_checks = ["how are you", "how's it going", "how are u"]
        if any(phrase in query_lc for phrase in friendly_checks):
            return "I'm just code, but I'm ready to help! What would you like to know from the documents?"

        gratitude_phrases = {"thank you", "thanks", "thx", "i appreciate"}
        if any(token in query_lc for token in gratitude_phrases):
            return "You're very welcome! Let me know if there's anything else you want to explore."

        if not evidence:
            return "I couldn't find supporting information in the documents. Try rephrasing or adding more context."

        def _extract_text(item: Any) -> str:
            try:
                if isinstance(item, dict):
                    return item.get("text", "")
                return getattr(item, "text", "") if hasattr(item, "text") else str(item)
            except Exception:
                return ""

        def _normalize_sentence(sentence: str) -> str:
            cleaned = sentence.strip().lstrip("•-*→")
            cleaned = re.sub(r"\s+", " ", cleaned)
            return cleaned

        sentences: List[str] = []
        seen = set()
        for item in evidence:
            text = _extract_text(item)
            if not text:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                normalized = _normalize_sentence(sentence)
                if normalized and normalized.lower() not in seen:
                    seen.add(normalized.lower())
                    sentences.append(normalized)

        if not sentences:
            return "I found related documents, but they didn't contain readable text."

        summary_parts = list(islice(sentences, 2))
        summary = " ".join(summary_parts)
        if summary and summary[-1] not in {".", "!", "?"}:
            summary += "."

        supporting_bullets = list(islice(sentences, 6))
        bullet_section = "\n".join(f"• {s}" for s in supporting_bullets)

        return f"{summary}\n\nSupporting evidence:\n{bullet_section}"

    async def generate_answer_async(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        # mimic async call
        return self.generate_answer(query, evidence)

    def rewrite_query(self, query: str) -> str:
        return query.replace("Which", "What").replace("which", "what")

    async def rewrite_query_async(self, query: str) -> str:
        return self.rewrite_query(query)

    def plan(self, query: str) -> List[Dict[str, Any]]:
        return []

    async def plan_async(self, query: str) -> List[Dict[str, Any]]:
        return self.plan(query)


class OllamaAdapter:
    """Adapter for local Ollama model.

    This adapter shells out to the local `ollama` CLI (must be in PATH).
    Because the CLI is blocking, we run it in a thread via asyncio.to_thread.

    Usage: set env var USE_OLLAMA=1 or set OLLAMA_MODEL to select.
    """

    def __init__(self, model: Optional[str] = None, timeout: int = 30):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
        self.timeout = timeout

    def _run_cli(self, prompt: str) -> str:
        # Call: ollama run <model> --prompt "..."
        cmd = ["ollama", "run", self.model, "--prompt", prompt]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            text = res.stdout.strip()
            if not text:
                # sometimes ollama prints to stderr
                text = res.stderr.strip()
            return text
        except FileNotFoundError:
            raise RuntimeError("'ollama' CLI not found in PATH. Please install Ollama or adjust PATH.")

    def _stream_cli(self, prompt: str):
        """Attempt to stream output from ollama CLI using subprocess.Popen. Yields lines."""
        cmd = ["ollama", "run", self.model, "--prompt", prompt]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            raise RuntimeError("'ollama' CLI not found in PATH.")
        assert p.stdout is not None
        for line in p.stdout:
            yield line
        p.wait()

    async def generate_answer_stream(self, query: str, evidence: List[Dict[str, Any]]):
        """Async generator that yields streaming chunks from the Ollama CLI."""
        prompt = f"Answer the user query: {query}\nGiven evidence:\n"
        for e in evidence:
            if isinstance(e, dict):
                txt = e.get("text", "")
            else:
                txt = getattr(e, "text", "") if hasattr(e, "text") else str(e)
            prompt += f"- {txt}\n"
        LLM_CALLS.labels(model=self.model).inc()
        try:
            # run the blocking stream in a thread and yield lines as they come
            def runner():
                for ln in self._stream_cli(prompt):
                    yield ln

            # asyncio.to_thread doesn't support yielding; instead, run synchronously in thread and collect
            # We'll spawn a subprocess here directly and read its stdout asynchronously using asyncio
            proc = await asyncio.create_subprocess_exec(
                "ollama", "run", self.model, "--prompt", prompt,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.readline()
                if not chunk:
                    break
                yield chunk.decode(errors="replace")
            await proc.wait()
        except FileNotFoundError:
            raise RuntimeError("'ollama' CLI not found in PATH.")
        except Exception:
            # fallback: return full text via run_cli
            text = await asyncio.to_thread(self._run_cli, prompt)
            yield text

    async def generate_answer_async(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        prompt = f"Answer the user query: {query}\nGiven evidence:\n"
        for e in evidence:
            if isinstance(e, dict):
                txt = e.get("text", "")
            else:
                txt = getattr(e, "text", "") if hasattr(e, "text") else str(e)
            prompt += f"- {txt}\n"
        LLM_CALLS.labels(model=self.model).inc()
        # prefer streaming if available
        try:
            text = ""
            # try streaming via subprocess; if it fails, fall back to full run
            try:
                for chunk in self._stream_cli(prompt):
                    text += chunk
            except Exception:
                text = await asyncio.to_thread(self._run_cli, prompt)
            if not text:
                text = await asyncio.to_thread(self._run_cli, prompt)
            return text
        except Exception:
            return await asyncio.to_thread(self._run_cli, prompt)

    async def rewrite_query_async(self, query: str) -> str:
        prompt = f"Rewrite this query to be more precise: {query}"
        return await asyncio.to_thread(self._run_cli, prompt)

    async def plan_async(self, query: str) -> List[Dict[str, Any]]:
        prompt = f"Decompose the query into multiple subquestions (JSON array). Respond with ONLY a JSON array of objects with keys id,type,instruction,expected_tool. Query: {query}"
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "instruction", "expected_tool"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "instruction": {"type": "string"},
                    "expected_tool": {"type": "string"}
                }
            }
        }

        # retry/parse loop with backoff
        attempts = 3
        backoff = 0.5
        for i in range(attempts):
            LLM_CALLS.labels(model=self.model).inc()
            out = await asyncio.to_thread(self._run_cli, prompt)
            try:
                parsed = json.loads(out)
                validate(instance=parsed, schema=schema)
                return parsed
            except Exception as e:
                # try to salvage JSON in text by extracting first [...]
                import re
                m = re.search(r"(\[.*\])", out, re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        validate(instance=parsed, schema=schema)
                        return parsed
                    except Exception:
                        pass
                # backoff
                await asyncio.sleep(backoff)
                backoff *= 2

        # fallback simple plan
        return [
            {"id": "step1", "type": "retrieval", "instruction": f"Find entity related to: {query}", "expected_tool": "vector_or_bm25"},
            {"id": "step2", "type": "retrieval", "instruction": f"Find facts about the entity from step1", "expected_tool": "vector_or_bm25"}
        ]

