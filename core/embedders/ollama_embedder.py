from typing import List, Optional
import os
import subprocess
import json
import shlex


class OllamaEmbedder:
    """Adapter to get embeddings from a local Ollama model via CLI.

    It supports a configurable command template via env var
    OLLAMA_EMBED_CMD_TEMPLATE with placeholders {model} and {text}.
    Defaults to trying common CLI patterns and attempts to parse numeric output.
    """

    def __init__(self, model: Optional[str] = None, timeout: int = 30):
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL")
        self.timeout = timeout
        # user can specify a custom command template, e.g.:
        # OLLAMA_EMBED_CMD_TEMPLATE="ollama embed {model} --text '{text}'"
        self.cmd_template = os.getenv("OLLAMA_EMBED_CMD_TEMPLATE", "")

    def _run_cmd(self, cmd: str) -> str:
        # Use shell to allow user templates; careful with quoting
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=self.timeout)
            out = res.stdout.strip() or res.stderr.strip()
            return out
        except FileNotFoundError:
            raise RuntimeError("'ollama' CLI not found in PATH or command template invalid.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("OLLAMA embed model not configured. Set OLLAMA_EMBED_MODEL")
        results = []
        for t in texts:
            # choose command
            if self.cmd_template:
                cmd = self.cmd_template.format(model=self.model, text=shlex.quote(t))
                out = self._run_cmd(cmd)
                try:
                    vec = self._parse_output(out)
                    results.append(vec)
                    continue
                except Exception:
                    # fallthrough to try run-based embedding
                    pass

            # try embed subcommand first
            tried_cmds = []
            embed_cmd = f"ollama embed {self.model} --text {shlex.quote(t)}"
            tried_cmds.append(embed_cmd)
            out = None
            try:
                out = self._run_cmd(embed_cmd)
                vec = self._parse_output(out)
                results.append(vec)
                continue
            except Exception:
                pass

            # try piping text to ollama run (handles versions that accept stdin)
            run_cmd_pipe = f"echo {shlex.quote(t)} | ollama run {self.model}"
            tried_cmds.append(run_cmd_pipe)
            try:
                out2 = self._run_cmd(run_cmd_pipe)
                vec = self._parse_output(out2)
                results.append(vec)
                continue
            except Exception:
                pass

            # try passing text as an argument to ollama run
            run_cmd_arg = f"ollama run {self.model} {shlex.quote(t)}"
            tried_cmds.append(run_cmd_arg)
            try:
                out3 = self._run_cmd(run_cmd_arg)
                vec = self._parse_output(out3)
                results.append(vec)
                continue
            except Exception:
                pass

            # If none worked, raise an informative error including what we tried and the last output
            raise RuntimeError(f"Failed to get embeddings from ollama for model {self.model}. Tried commands: {tried_cmds}. Last output: {out or out2 or out3}")
        return results

    def _parse_output(self, out: str) -> List[float]:
        # Try JSON parse
        try:
            parsed = json.loads(out)
            # parsed could be dict with 'embedding' key or a list
            if isinstance(parsed, dict) and "embedding" in parsed:
                return parsed["embedding"]
            if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                return parsed
        except Exception:
            pass
        # fallback: try to extract numbers from text
        parts = out.replace("[", " ").replace("]", " ").replace(",", " ").split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                continue
        if not nums:
            raise RuntimeError(f"Could not parse embedding output: {out[:200]}")
        return nums
