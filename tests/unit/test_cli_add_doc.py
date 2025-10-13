import json
import os
import subprocess
import sys


def test_cli_add_doc(tmp_path, monkeypatch):
    # run the CLI to add a custom doc using the project python
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    python = sys.executable
    custom_text = "Test document from CLI add-doc"
    cmd = [python, os.path.join(repo, "cli", "chat.py"), "--add-doc", custom_text]
    env = os.environ.copy()
    env["PYTHONPATH"] = repo + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    # run the CLI; it will write data/custom_docs.json
    subprocess.run(cmd, env=env, check=True)
    custom_path = os.path.join(repo, "data", "custom_docs.json")
    assert os.path.exists(custom_path)
    with open(custom_path) as f:
        docs = json.load(f)
    assert any(custom_text in d.get("text", "") for d in docs)
