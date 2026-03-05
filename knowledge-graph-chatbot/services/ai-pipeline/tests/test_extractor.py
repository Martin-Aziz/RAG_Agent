"""
tests/test_extractor.py — Unit tests for the knowledge extraction pipeline.
Tests NER entity extraction and REBEL relation parsing.
"""

import pytest
from src.nlp.extractor import KnowledgeExtractor


class TestCyberEntityExtraction:
    """Test cybersecurity-specific entity detection via regex patterns."""

    def setup_method(self):
        self.extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        self.extractor._nlp = None
        self.extractor._rebel = None
        self.extractor._spacy_model = "en_core_web_sm"

    def test_cve_extraction(self):
        """CVE identifiers should be detected with high confidence."""
        text = "The vulnerability CVE-2021-44228 affects Apache Log4j 2."
        entities = self.extractor._extract_cyber_entities(text)

        cve_entities = [e for e in entities if e.label == "CVE"]
        assert len(cve_entities) == 1
        assert cve_entities[0].text == "CVE-2021-44228"
        assert cve_entities[0].confidence >= 0.95

    def test_multiple_cves(self):
        """Multiple CVE IDs in one text should all be extracted."""
        text = "Both CVE-2021-44228 and CVE-2021-45046 affect Log4j."
        entities = self.extractor._extract_cyber_entities(text)

        cve_entities = [e for e in entities if e.label == "CVE"]
        assert len(cve_entities) == 2

    def test_cwe_extraction(self):
        """CWE identifiers should be detected as VULNERABILITY type."""
        text = "This is related to CWE-79 (Cross-site Scripting)."
        entities = self.extractor._extract_cyber_entities(text)

        vuln_entities = [e for e in entities if e.label == "VULNERABILITY"]
        assert len(vuln_entities) == 1
        assert "CWE-79" in vuln_entities[0].text

    def test_mitre_attack_extraction(self):
        """MITRE ATT&CK technique IDs should be detected."""
        text = "The attacker used technique T1059.001 for execution."
        entities = self.extractor._extract_cyber_entities(text)

        attack_entities = [e for e in entities if e.label == "ATTACK_PATTERN"]
        assert len(attack_entities) == 1
        assert "T1059" in attack_entities[0].text


class TestREBELParsing:
    """Test REBEL output parsing."""

    def setup_method(self):
        self.extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        self.extractor._nlp = None
        self.extractor._rebel = None
        self.extractor._spacy_model = "en_core_web_sm"

    def test_single_triplet(self):
        """Single REBEL triplet should be parsed correctly."""
        output = "<triplet> APT41 <subj> exploits <obj> CVE-2021-44228"
        triples = self.extractor._parse_rebel_output(output)

        assert len(triples) == 1
        assert triples[0] == ("APT41", "exploits", "CVE-2021-44228")

    def test_multiple_triplets(self):
        """Multiple REBEL triplets should all be parsed."""
        output = (
            "<triplet> APT41 <subj> exploits <obj> CVE-2021-44228 "
            "<triplet> Log4j <subj> affected by <obj> CVE-2021-44228"
        )
        triples = self.extractor._parse_rebel_output(output)
        assert len(triples) == 2

    def test_empty_output(self):
        """Empty REBEL output should return empty list."""
        triples = self.extractor._parse_rebel_output("")
        assert len(triples) == 0


class TestRelationMapping:
    """Test relation type mapping from REBEL to domain types."""

    def setup_method(self):
        self.extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)

    def test_exploit_mapping(self):
        assert self.extractor._map_relation("exploits") == "EXPLOITS"
        assert self.extractor._map_relation("attacked by") == "EXPLOITS"

    def test_mitigate_mapping(self):
        assert self.extractor._map_relation("mitigates") == "MITIGATES"
        assert self.extractor._map_relation("patched by") == "MITIGATES"

    def test_fallback_mapping(self):
        assert self.extractor._map_relation("unknown relation") == "RELATED_TO"


class TestDeduplication:
    """Test entity and relation deduplication."""

    def setup_method(self):
        self.extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)

    def test_entity_dedup(self):
        """Duplicate entities should be merged, keeping highest confidence."""
        from src.models.schemas import ExtractedEntity

        entities = [
            ExtractedEntity(text="CVE-2021-44228", label="CVE",
                          start_char=0, end_char=15, confidence=0.8),
            ExtractedEntity(text="CVE-2021-44228", label="CVE",
                          start_char=50, end_char=65, confidence=0.95),
        ]

        deduped = self.extractor._deduplicate_entities(entities)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.95
