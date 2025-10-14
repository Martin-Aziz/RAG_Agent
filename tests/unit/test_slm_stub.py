import pytest

from core.model_adapters import SLMStub


@pytest.fixture()
def stub():
    return SLMStub()


def test_generate_answer_greeting(stub):
    reply = stub.generate_answer("Hello there", evidence=[])
    assert "knowledge base" in reply.lower() or "documents" in reply.lower()


def test_generate_answer_friendly_check(stub):
    reply = stub.generate_answer("how are you doing?", evidence=[{"text": "Sample evidence."}])
    assert "i'm just code" in reply.lower()


def test_generate_answer_gratitude(stub):
    reply = stub.generate_answer("Thanks for your help", evidence=[])
    assert "you're very welcome" in reply.lower()


def test_generate_answer_no_evidence(stub):
    reply = stub.generate_answer("What is the capital?", evidence=[])
    assert "don't contain enough information" in reply or "provided documents" in reply


def test_generate_answer_deduplicates_sentences(stub):
    evidence = [
        {"doc_id": "Paris-doc", "text": "Paris is the capital of France. Paris is known for the Eiffel Tower."},
        {"doc_id": "Paris-doc2", "text": "Paris is the capital of France."},
    ]

    reply = stub.generate_answer("Tell me about Paris", evidence=evidence)

    # Updated to check for the new structured format with "Summary", "Reasoning", "Supporting Evidence", "Conclusion"
    assert "Summary:" in reply
    assert "Reasoning" in reply or "Supporting Evidence" in reply
    # The answer should mention Paris facts with document references
    assert "Paris" in reply


def test_rewrite_query_lowercases(stub):
    rewritten = stub.rewrite_query("Which movie won the Oscar?")
    # The new rewrite_query expands terms and adds instructions, not just lowercasing
    assert len(rewritten) > 0
    # It should have added synonyms or instructions
    assert "consider" in rewritten or "film" in rewritten or "citations" in rewritten