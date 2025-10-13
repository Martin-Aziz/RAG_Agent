import pytest

from core.model_adapters import SLMStub


@pytest.fixture()
def stub():
    return SLMStub()


def test_generate_answer_greeting(stub):
    reply = stub.generate_answer("Hello there", evidence=[])
    assert "feel free to ask" in reply.lower()


def test_generate_answer_friendly_check(stub):
    reply = stub.generate_answer("how are you doing?", evidence=[{"text": "Sample evidence."}])
    assert "i'm just code" in reply.lower()


def test_generate_answer_gratitude(stub):
    reply = stub.generate_answer("Thanks for your help", evidence=[])
    assert "you're very welcome" in reply.lower()


def test_generate_answer_no_evidence(stub):
    reply = stub.generate_answer("What is the capital?", evidence=[])
    assert "couldn't find supporting information" in reply


def test_generate_answer_deduplicates_sentences(stub):
    evidence = [
        {"text": "Paris is the capital of France. Paris is known for the Eiffel Tower."},
        {"text": "Paris is the capital of France."},
    ]

    reply = stub.generate_answer("Tell me about Paris", evidence=evidence)

    lines = reply.splitlines()
    bullets = [line for line in lines if line.startswith("• ")]
    assert len(bullets) == 2
    assert any("capital of France" in line for line in bullets)
    assert any("Eiffel Tower" in line for line in bullets)


def test_rewrite_query_lowercases(stub):
    assert stub.rewrite_query("Which city?") == "What city?"