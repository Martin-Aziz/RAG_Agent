from typing import List, Dict, Any, Optional, Sequence, Tuple
import os
import asyncio
import subprocess
import time
import json
import re
import logging
from .observability import LLM_CALLS
from .prompts import build_generation_prompt, RETRIEVAL_PROMPT
from .exceptions import ModelException, ResponseParsingException, ModelUnavailableException, ServiceTimeoutException
from .caching import cached, query_cache, embedding_cache

# Configure logger
logger = logging.getLogger(__name__)

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
        tokens = query_lc.replace("?", " ").replace("!", " ").split()
        lowered_tokens = set(tokens)
        for greet in self._GREETINGS:
            if " " in greet:
                if greet in query_lc:
                    return True
            elif greet in lowered_tokens:
                return True
        return False

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

    def _shorten(self, text: str, limit: int = 160) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _infer_follow_up(self, query: str) -> str:
        query_lc = query.lower()
        if any(keyword in query_lc for keyword in ["install", "setup", "set up", "configure", "deploy"]):
            return "Would you like a verification checklist or sample configuration?"
        if any(keyword in query_lc for keyword in ["learn", "understand", "concept", "theory", "explain"]):
            return "Should we dive deeper into the underlying theory?"
        return "Do you want me to suggest a follow-up task or example?"

    def _infer_expertise(self, query: str) -> str:
        query_lc = query.lower()
        if any(phrase in query_lc for phrase in ["i'm new", "beginner", "step by step", "explain like", "eli5", "walk me through"]):
            return "beginner"
        if any(phrase in query_lc for phrase in ["advanced", "production", "optimize", "deep dive", "architecture"]):
            return "advanced"
        return "intermediate"

    def _infer_prerequisites_flag(self, query: str, existing_state: Optional[Dict[str, Any]]) -> str:
        query_lc = query.lower()
        if any(keyword in query_lc for keyword in ["install", "setup", "set up", "configure", "deploy"]):
            return "needs_confirmation"
        if any(keyword in query_lc for keyword in ["already", "have", "installed", "configured"]):
            return "likely_met"
        if existing_state and existing_state.get("prerequisites_met"):
            return str(existing_state.get("prerequisites_met"))
        return "unknown"

    def _compose_state_update(
        self,
        query: str,
        teaching_state: Optional[Dict[str, Any]],
        follow_up: str,
        step_count: int,
        prerequisites_flag: str,
    ) -> Dict[str, Any]:
        state = dict(teaching_state or {})
        if not state.get("user_goal"):
            state["user_goal"] = query.strip() or "Clarify request"
        if not state.get("expertise"):
            state["expertise"] = self._infer_expertise(query)
        state["current_step"] = f"Providing guidance with {step_count} step(s) prepared"
        state["prerequisites_met"] = prerequisites_flag
        state["next_suggestion"] = follow_up
        return state

    def generate_answer(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        teaching_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        query_lc = (query or "").lower()

        if self._is_greeting(query_lc):
            return json.dumps({
                "summary": "Hello! I'm here to help you explore the knowledge base. Let me know what you'd like to learn from the provided documents.",
                "structured_response": "✅ Requirements\n- Let me know the topic you want to cover.\n\n🧩 Steps\n1. Share a document-backed question.\n\n💡 Tips or common pitfalls\n- Ask about topics present in the knowledge base so I can cite them.\n\n📘 Optional theory\n- None this time.\n\n🤖 Follow-up\n- Would you like suggestions on topics to explore?\n\n📍 State\n- user_goal: greeting\n- current_step: awaiting user question\n- prerequisites_met: unknown\n- next_suggestion: explore documents",
                "citations": [],
                "teaching_state": self._compose_state_update(
                    query,
                    teaching_state,
                    "Would you like suggestions on topics to explore?",
                    step_count=0,
                    prerequisites_flag=self._infer_prerequisites_flag(query, teaching_state),
                ),
                "follow_up": "Would you like suggestions on topics to explore?"
            })

        if self._is_friendly_check(query_lc):
            return json.dumps({
                "summary": "I'm just code, but I'm ready to help! What would you like to know from the documents?",
                "structured_response": "✅ Requirements\n- Tell me the document-backed task you want to tackle.\n\n🧩 Steps\n1. Ask your next question.\n\n💡 Tips or common pitfalls\n- Mention what you have already tried so we can tailor the guidance.\n\n📘 Optional theory\n- None this time.\n\n🤖 Follow-up\n- Do you want me to suggest a specific topic to review?\n\n📍 State\n- user_goal: friendly check-in\n- current_step: awaiting next query\n- prerequisites_met: unknown\n- next_suggestion: suggest topic",
                "citations": [],
                "teaching_state": self._compose_state_update(
                    query,
                    teaching_state,
                    "Do you want me to suggest a specific topic to review?",
                    step_count=0,
                    prerequisites_flag=self._infer_prerequisites_flag(query, teaching_state),
                ),
                "follow_up": "Do you want me to suggest a specific topic to review?"
            })

        if self._is_gratitude(query_lc):
            return json.dumps({
                "summary": "You're very welcome! Let me know if there's anything else you want to explore in the documents.",
                "structured_response": "✅ Requirements\n- Decide what you would like to learn next.\n\n🧩 Steps\n1. Share your follow-up question.\n\n💡 Tips or common pitfalls\n- Let me know if you need more depth or a recap.\n\n📘 Optional theory\n- None this time.\n\n🤖 Follow-up\n- Would you like me to recap the key documents again?\n\n📍 State\n- user_goal: continue learning\n- current_step: awaiting next topic\n- prerequisites_met: unknown\n- next_suggestion: recap documents",
                "citations": [],
                "teaching_state": self._compose_state_update(
                    query,
                    teaching_state,
                    "Would you like me to recap the key documents again?",
                    step_count=0,
                    prerequisites_flag=self._infer_prerequisites_flag(query, teaching_state),
                ),
                "follow_up": "Would you like me to recap the key documents again?"
            })

        normalised = self._normalize_evidence(evidence)
        prerequisites_flag = self._infer_prerequisites_flag(query, teaching_state)
        follow_up = self._infer_follow_up(query)

        if not normalised:
            structured = (
                "✅ Requirements\n"
                "- I could not locate supporting material in the knowledge base.\n\n"
                "🧩 Steps\n"
                "1. Provide more details or add documents covering this topic.\n\n"
                "💡 Tips or common pitfalls\n"
                "- Ensure the knowledge base contains the required references before retrying.\n\n"
                "📘 Optional theory\n"
                "- None available without supporting sources.\n\n"
                "🤖 Follow-up\n"
                f"- {follow_up}\n\n"
                "📍 State\n"
                f"- user_goal: {(teaching_state or {}).get('user_goal', query.strip() or 'clarify topic')}\n"
                "- current_step: awaiting additional context\n"
                f"- prerequisites_met: {prerequisites_flag}\n"
                f"- next_suggestion: {follow_up}"
            )
            return json.dumps({
                "summary": "I'm sorry — the provided documents don't contain enough information to answer that.",
                "structured_response": structured,
                "citations": [],
                "teaching_state": self._compose_state_update(
                    query,
                    teaching_state,
                    follow_up,
                    step_count=0,
                    prerequisites_flag=prerequisites_flag,
                ),
                "follow_up": follow_up,
            }, indent=2)

        summary, reasoning, supporting = self._summarize_sentences(normalised)

        def _build_section_lines(items: Sequence[Dict[str, Any]], formatter) -> List[str]:
            lines: List[str] = []
            for idx, item in enumerate(items, start=1):
                line = formatter(idx, item)
                if line:
                    lines.append(line)
            return lines

        requirements_lines = _build_section_lines(normalised[:2], lambda idx, item: (
            f"- Ensure you can access: {self._shorten(self._normalize_sentence(item['text']))} [{item.get('doc_id', 'doc')}]"
        ))
        if not requirements_lines:
            requirements_lines.append("- Confirm you have the necessary tools referenced in the cited documents.")

        steps_lines = _build_section_lines(normalised[:4], lambda idx, item: (
            f"{idx}. {self._shorten(self._normalize_sentence(item['text']))} [{item.get('doc_id', 'doc')}]"
        ))
        if not steps_lines:
            steps_lines.append("1. Review the cited documents for actionable steps.")

        tips_lines = [
            "- Cross-reference each step with the cited document to avoid misconfigurations."
        ]
        expertise_level = (teaching_state or {}).get("expertise") or self._infer_expertise(query)
        if expertise_level == "beginner":
            tips_lines.append("- Quick check: Which document explains the prerequisites mentioned above?")
        elif expertise_level == "advanced":
            tips_lines.append("- Consider automating these steps or integrating them into your pipeline if applicable.")

        optional_theory_lines = supporting[:2] if supporting else [
            "- No additional theory was found in the current context."
        ]

        structured_sections = [
            "✅ Requirements",
            "\n".join(requirements_lines),
            "",
            "🧩 Steps",
            "\n".join(steps_lines),
            "",
            "💡 Tips or common pitfalls",
            "\n".join(tips_lines),
            "",
            "📘 Optional theory",
            "\n".join(optional_theory_lines),
            "",
            "🤖 Follow-up",
            f"- {follow_up}",
            "",
            "📍 State",
            f"- user_goal: {(teaching_state or {}).get('user_goal', query.strip() or 'clarify topic')}",
            "- current_step: outlining guidance",
            f"- prerequisites_met: {prerequisites_flag}",
            f"- next_suggestion: {follow_up}",
        ]

        structured_response = "\n".join(section for section in structured_sections if section is not None)
        
        evidence_list = []
        for item in normalised:
            evidence_list.append({
                "document": item.get("doc_id", "doc"),
                "passage": item.get("text", "")
            })

        updated_state = self._compose_state_update(
            query,
            teaching_state,
            follow_up,
            step_count=len(steps_lines),
            prerequisites_flag=prerequisites_flag,
        )

        response = {
            "summary": summary,
            "structured_response": structured_response,
            "citations": evidence_list,
            "teaching_state": updated_state,
            "follow_up": follow_up,
        }
        return json.dumps(response, indent=2)

    async def generate_answer_async(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        teaching_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.generate_answer(query, evidence, teaching_state)

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

    def _build_generation_prompt(
        self,
        query: str,
        evidence: Sequence[Any],
        teaching_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        context_pairs = self._prepare_context_pairs(evidence)
        prompt = build_generation_prompt(query, context_pairs, teaching_state=teaching_state)
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

    async def generate_answer_stream(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        teaching_state: Optional[Dict[str, Any]] = None,
    ):
        """Async generator that yields streaming chunks from the Ollama CLI."""
        prompt, context_pairs = self._build_generation_prompt(query, evidence, teaching_state=teaching_state)
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

    async def generate_answer_async(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        teaching_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt, context_pairs = self._build_generation_prompt(query, evidence, teaching_state=teaching_state)
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

