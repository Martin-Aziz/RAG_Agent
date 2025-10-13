from typing import List, Dict, Any, Optional
import os
import asyncio
import subprocess
import time
import json
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
        if not evidence:
            return "I don't know based on the provided documents."
        lines = [f"Based on {len(evidence)} pieces of evidence:"]
        for e in evidence[:3]:
            lines.append(e.get("text", ""))
        return "\n".join(lines)

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
            prompt += f"- {e.get('text','')}\n"
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
            prompt += f"- {e.get('text','')}\n"
        LLM_CALLS.labels(model=self.model).inc()
        # prefer streaming if available
        try:
            text = ""
            for chunk in await asyncio.to_thread(lambda: "".join(self._stream_cli(prompt))):
                text += chunk
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

