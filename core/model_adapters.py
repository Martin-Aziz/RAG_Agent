from typing import List, Dict, Any, Optional, Sequence, Tuple
import os
import asyncio
import subprocess
import time
import json
import re
from .observability import LLM_CALLS
from .prompts import build_generation_prompt, RETRIEVAL_PROMPT
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


class SLMStub:
    """Deterministic small language model stub for planning and rewriting."""

    _GREETINGS = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
    _FRIENDLY_CHECKS = {"how are you", "how's it going", "how are u"}
    _GRATITUDE = {"thank you", "thanks", "thx", "i appreciate"}
    _SYNONYM_MAP = {
        "movie": ["film", "motion picture"],
        "director": ["filmmaker", "movie director"],
        "ai": ["artificial intelligence"],
        "rag": ["retrieval augmented generation"],
        "document": ["record", "file"],
        "company": ["organization", "firm"],
        "research": ["analysis", "study"],
    }

    def _is_greeting(self, query_lc: str) -> bool:
        return any(greet in query_lc for greet in self._GREETINGS)

    def _is_friendly_check(self, query_lc: str) -> bool:
        return any(phrase in query_lc for phrase in self._FRIENDLY_CHECKS)

    def _is_gratitude(self, query_lc: str) -> bool:
        return any(token in query_lc for token in self._GRATITUDE)

    def _extract_text(self, item: Any) -> str:
        try:
            if isinstance(item, dict):
                return item.get("text", "")
            return getattr(item, "text", "") if hasattr(item, "text") else str(item)
        except Exception:
            return ""

    def _normalize_sentence(self, sentence: str) -> str:
        cleaned = sentence.strip().lstrip("•-*→")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _normalize_evidence(self, evidence: Sequence[Any]) -> List[Dict[str, Any]]:
        normalised: List[Dict[str, Any]] = []
        for idx, item in enumerate(evidence, start=1):
            text = self._extract_text(item)
            if not text:
                continue
            default_doc = f"Document {idx}"
            if isinstance(item, dict):
                doc_id = item.get("doc_id") or default_doc
                passage_id = item.get("passage_id")
                score = item.get("score")
            else:
                doc_id = getattr(item, "doc_id", default_doc)
                passage_id = getattr(item, "passage_id", None)
                score = getattr(item, "score", None)
            normalised.append({
                "doc_id": doc_id,
                "passage_id": passage_id,
                "score": score,
                "text": text,
            })
        return normalised

    def _summarize_sentences(self, normalised: Sequence[Dict[str, Any]]) -> Tuple[str, List[str], List[str]]:
        sentences: List[str] = []
        reasoning: List[str] = []
        supporting: List[str] = []
        seen = set()

        for item in normalised:
            doc_id = item["doc_id"] or "Document"
            for sentence in re.split(r"(?<=[.!?])\s+", item["text"]):
                normalized = self._normalize_sentence(sentence)
                if not normalized:
                    continue
                key = (doc_id, normalized.lower())
                if key in seen:
                    continue
                seen.add(key)
                if len(sentences) < 3:
                    sentences.append(normalized)
                short = normalized
                if len(short) > 160:
                    short = short[:157].rstrip() + "…"
                reasoning.append(f"- Refer to {doc_id} for: {short}")
                supporting.append(f"- {doc_id}: \"{short}\"")

        summary = " ".join(sentences[:2])
        if summary and summary[-1] not in {".", "!", "?"}:
            summary += "."

        if not summary:
            summary = "The referenced documents contain information relevant to the question."

        return summary, reasoning[:5], supporting[:5]

    def generate_answer(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        query_lc = (query or "").lower()

        if self._is_greeting(query_lc):
            return json.dumps({
                "summary": "Hello! I'm here to help you explore the knowledge base. Let me know what you'd like to learn from the provided documents.",
                "evidence": []
            })

        if self._is_friendly_check(query_lc):
            return json.dumps({
                "summary": "I'm just code, but I'm ready to help! What would you like to know from the documents?",
                "evidence": []
            })

        if self._is_gratitude(query_lc):
            return json.dumps({
                "summary": "You're very welcome! Let me know if there's anything else you want to explore in the documents.",
                "evidence": []
            })

        normalised = self._normalize_evidence(evidence)
        if not normalised:
            return json.dumps({
                "summary": "I'm sorry — the provided documents don't contain enough information to answer that.",
                "evidence": []
            })

        summary, _, _ = self._summarize_sentences(normalised)
        
        evidence_list = []
        for item in normalised:
            evidence_list.append({
                "document": item.get("doc_id", "doc"),
                "passage": item.get("text", "")
            })

        response = {
            "summary": summary,
            "evidence": evidence_list
        }
        return json.dumps(response, indent=2)

    async def generate_answer_async(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        return self.generate_answer(query, evidence)

    def rewrite_query(self, query: str) -> str:
        if not query:
            return ""
        query_lc = query.strip().lower()
        if self._is_greeting(query_lc):
            return "How can I help you explore the documents today?"

        expanded_terms: List[str] = []
        for word in re.findall(r"[A-Za-z]{4,}", query_lc):
            synonyms = self._SYNONYM_MAP.get(word)
            if synonyms:
                expanded_terms.extend(synonyms)

        expanded_str = ", ".join(dict.fromkeys(expanded_terms))
        rewritten = query.strip()
        if expanded_str:
            rewritten += f" (also consider: {expanded_str})"
        if "document" not in query_lc and "context" not in query_lc:
            rewritten += " — focus on relevant documents with citations"
        return rewritten

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

    def _prepare_context_pairs(self, evidence: Sequence[Any]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for idx, item in enumerate(evidence, start=1):
            text: str
            doc_id: Optional[str]
            if isinstance(item, dict):
                text = item.get("text", "")
                doc_id = item.get("doc_id") or item.get("id")
            else:
                text = getattr(item, "text", "") if hasattr(item, "text") else str(item)
                doc_id = getattr(item, "doc_id", None)
            text = (text or "").strip()
            if not text:
                continue
            label = doc_id or f"Document {idx}"
            pairs.append((label, text))
        return pairs

    def _build_generation_prompt(self, query: str, evidence: Sequence[Any]) -> Tuple[str, List[Tuple[str, str]]]:
        context_pairs = self._prepare_context_pairs(evidence)
        prompt = build_generation_prompt(query, context_pairs)
        return prompt, context_pairs

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
        prompt, context_pairs = self._build_generation_prompt(query, evidence)
        if not context_pairs:
            yield "I'm sorry — the provided documents don't contain enough information to answer that."
            return
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
        prompt, context_pairs = self._build_generation_prompt(query, evidence)
        if not context_pairs:
            return "I'm sorry — the provided documents don't contain enough information to answer that."
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
        template = RETRIEVAL_PROMPT.format(question=query)
        LLM_CALLS.labels(model=self.model).inc()
        try:
            return await asyncio.to_thread(self._run_cli, template)
        except Exception:
            return query

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

