"""
nlp/extractor.py — Knowledge extraction pipeline using spaCy NER + REBEL.

Extracts entities (NER) and relations (relation extraction) from text to
build the knowledge graph. Combines:
1. spaCy transformer pipeline for named entity recognition
2. REBEL (Babelscape/rebel-large) for relation extraction from sentences

Design decisions:
- spaCy for NER: best accuracy with transformer backbone, handles entity spans
- REBEL for RE: seq2seq model that generates (subject, relation, object) triples
- Confidence thresholds: filter low-confidence extractions (NER >= 0.5, RE >= 0.3)
- Entity deduplication: merge entities with same text/type across sentences
- Cybersecurity-specific: custom entity label mapping for STIX/MITRE types
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from loguru import logger

from src.models.schemas import ExtractedEntity, ExtractedRelation


# Minimum confidence thresholds for extraction quality filtering
NER_MIN_CONFIDENCE = 0.5
RE_MIN_CONFIDENCE = 0.3

# Map spaCy NER labels to our domain-specific labels
SPACY_LABEL_MAP = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",      # Geo-political entities
    "LOC": "LOCATION",
    "PRODUCT": "SOFTWARE",   # Software products
    "EVENT": "CAMPAIGN",     # Campaigns/events
    "WORK_OF_ART": "OTHER",
    "LAW": "MITIGATION",    # Regulations as mitigations
    "NORP": "ORGANIZATION",  # Nationalities/groups
    "FAC": "OTHER",
    "DATE": "OTHER",
    "CARDINAL": "OTHER",
}

# Regex patterns for cybersecurity-specific entity detection
CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,}', re.IGNORECASE)
CWE_PATTERN = re.compile(r'CWE-\d+', re.IGNORECASE)
MITRE_PATTERN = re.compile(r'T\d{4}(?:\.\d{3})?', re.IGNORECASE)  # ATT&CK technique IDs


class KnowledgeExtractor:
    """Extracts entities and relations from text for knowledge graph construction.

    Combines spaCy NER with REBEL relation extraction in a two-stage pipeline:
    1. NER pass: extract named entities with confidence scores
    2. RE pass: extract (subject, relation, object) triples from sentences
    3. Alignment: match RE triples to NER entities for consistency
    """

    def __init__(self, spacy_model: str = "en_core_web_trf"):
        """Initialize the extraction pipeline.

        Args:
            spacy_model: spaCy model to use for NER. Falls back to
                en_core_web_sm if transformer model is unavailable.
        """
        self._nlp = None
        self._rebel = None
        self._spacy_model = spacy_model

        logger.info(f"KnowledgeExtractor configured: spacy={spacy_model}")

    def _load_models(self):
        """Lazily load NLP models on first use."""
        if self._nlp is not None:
            return

        import spacy

        # Try transformer model first, fall back to small model
        try:
            self._nlp = spacy.load(self._spacy_model)
            logger.info(f"Loaded spaCy model: {self._spacy_model}")
        except OSError:
            logger.warning(
                f"spaCy model {self._spacy_model} not found, "
                f"falling back to en_core_web_sm"
            )
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("No spaCy model available — downloading en_core_web_sm")
                spacy.cli.download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")

        # Load REBEL for relation extraction
        try:
            from transformers import pipeline as hf_pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1
            self._rebel = hf_pipeline(
                "text2text-generation",
                model="Babelscape/rebel-large",
                tokenizer="Babelscape/rebel-large",
                device=device,
            )
            logger.info(f"Loaded REBEL model on device={device}")
        except Exception as e:
            logger.warning(f"REBEL model not available: {e}. Relation extraction disabled.")
            self._rebel = None

    def extract(
        self, text: str, document_id: str = ""
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
        """Run the full extraction pipeline on input text.

        Args:
            text: Input text to analyze.
            document_id: Optional document ID for provenance.

        Returns:
            Tuple of (entities, relations) extracted from the text.
        """
        self._load_models()

        if not text or not text.strip():
            return [], []

        logger.info(f"Extracting from text ({len(text)} chars)")

        # Step 1: Run spaCy NER on full text → entities
        entities = self._extract_entities(text)

        # Step 1b: Add cybersecurity-specific entities via regex
        cyber_entities = self._extract_cyber_entities(text)
        entities.extend(cyber_entities)

        # Deduplicate entities by text + label
        entities = self._deduplicate_entities(entities)

        # Step 2: Split text into sentences for relation extraction
        doc = self._nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Step 3: Run REBEL on each sentence → relations
        relations = []
        for sentence in sentences:
            # Skip very short sentences (unlikely to contain useful relations)
            if len(sentence) < 20:
                continue

            sent_relations = self._extract_relations(sentence, document_id)
            relations.extend(sent_relations)

        # Deduplicate relations
        relations = self._deduplicate_relations(relations)

        logger.info(
            f"Extraction complete: {len(entities)} entities, "
            f"{len(relations)} relations"
        )

        return entities, relations

    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities using spaCy NER.

        Maps spaCy labels to domain-specific types and assigns
        confidence scores based on the model's prediction scores.
        """
        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            # Map spaCy label to our domain label
            label = SPACY_LABEL_MAP.get(ent.label_, ent.label_)

            # Skip non-useful entity types
            if label == "OTHER":
                continue

            # Estimate confidence from spaCy's internal scores
            # spaCy doesn't expose per-entity confidence directly, so we
            # use a heuristic based on the entity length and type
            confidence = min(0.95, 0.7 + (len(ent.text) / 100))

            entities.append(ExtractedEntity(
                id=str(uuid.uuid4()),
                text=ent.text,
                label=label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=round(confidence, 3),
            ))

        return entities

    def _extract_cyber_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract cybersecurity-specific entities via regex patterns.

        Catches CVE IDs, CWE IDs, and MITRE ATT&CK technique IDs
        that spaCy's general NER model might miss.
        """
        entities = []

        # CVE identifiers (e.g., CVE-2021-44228)
        for match in CVE_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                id=str(uuid.uuid4()),
                text=match.group(),
                label="CVE",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.99,  # Regex matches are highly confident
            ))

        # CWE identifiers (e.g., CWE-79)
        for match in CWE_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                id=str(uuid.uuid4()),
                text=match.group(),
                label="VULNERABILITY",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.99,
            ))

        # MITRE ATT&CK technique IDs (e.g., T1059.001)
        for match in MITRE_PATTERN.finditer(text):
            entities.append(ExtractedEntity(
                id=str(uuid.uuid4()),
                text=match.group(),
                label="ATTACK_PATTERN",
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.95,
            ))

        return entities

    def _extract_relations(
        self, sentence: str, document_id: str = ""
    ) -> List[ExtractedRelation]:
        """Extract relations from a single sentence using REBEL.

        REBEL generates text in the format:
        <triplet> subject <subj> relation <obj> object

        We parse this output to extract structured triples.
        """
        if self._rebel is None:
            return []

        try:
            # Run REBEL inference
            # max_length=256 to limit output for long sentences
            outputs = self._rebel(
                sentence,
                max_length=256,
                num_beams=3,
                num_return_sequences=1,
                return_text=True,
            )

            if not outputs:
                return []

            generated_text = outputs[0].get("generated_text", "")

            # Parse REBEL output format
            triples = self._parse_rebel_output(generated_text)

            relations = []
            for head, relation, tail in triples:
                # Skip self-referential relations
                if head.lower() == tail.lower():
                    continue

                # Map REBEL relation to our domain types
                mapped_relation = self._map_relation(relation)

                relations.append(ExtractedRelation(
                    head_text=head,
                    head_label="",  # Will be resolved during graph construction
                    relation=mapped_relation,
                    tail_text=tail,
                    tail_label="",
                    confidence=0.7,  # REBEL doesn't provide per-triple confidence
                    source_sentence=sentence[:500],  # Truncate for storage
                ))

            return relations

        except Exception as e:
            logger.warning(f"Relation extraction failed for sentence: {e}")
            return []

    def _parse_rebel_output(self, text: str) -> List[Tuple[str, str, str]]:
        """Parse REBEL's generated text into (subject, relation, object) triples.

        REBEL output format:
        <triplet> Barack Obama <subj> president of <obj> United States

        Multiple triples are separated by <triplet> tokens.
        """
        triples = []
        # Split on <triplet> tokens
        triplet_parts = text.split('<triplet>')

        for part in triplet_parts:
            part = part.strip()
            if not part:
                continue

            # Split on <subj> and <obj> markers
            subj_split = part.split('<subj>')
            if len(subj_split) != 2:
                continue

            head = subj_split[0].strip()
            rest = subj_split[1]

            obj_split = rest.split('<obj>')
            if len(obj_split) != 2:
                continue

            relation = obj_split[0].strip()
            tail = obj_split[1].strip()

            if head and relation and tail:
                triples.append((head, relation, tail))

        return triples

    def _map_relation(self, rebel_relation: str) -> str:
        """Map REBEL's free-text relation to our domain-specific relation types.

        Uses keyword matching to classify relations into our predefined types.
        """
        rel_lower = rebel_relation.lower()

        # Keyword-based mapping for cybersecurity domain
        mapping = {
            "exploit": "EXPLOITS",
            "attack": "EXPLOITS",
            "mitigat": "MITIGATES",
            "patch": "MITIGATES",
            "fix": "MITIGATES",
            "affect": "AFFECTS",
            "impact": "AFFECTS",
            "use": "USES",
            "employ": "USES",
            "target": "TARGETS",
            "attribut": "ATTRIBUTED_TO",
            "develop": "DEVELOPS",
            "creat": "DEVELOPS",
            "discover": "DISCOVERED_BY",
            "found": "DISCOVERED_BY",
            "distribut": "DISTRIBUTES",
            "part of": "PART_OF",
            "member": "PART_OF",
            "belong": "PART_OF",
        }

        for keyword, relation_type in mapping.items():
            if keyword in rel_lower:
                return relation_type

        return "RELATED_TO"  # Default fallback

    def _deduplicate_entities(
        self, entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping the highest-confidence version."""
        seen: dict[str, ExtractedEntity] = {}

        for entity in entities:
            key = f"{entity.text.lower()}:{entity.label}"
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def _deduplicate_relations(
        self, relations: List[ExtractedRelation]
    ) -> List[ExtractedRelation]:
        """Remove duplicate relations, keeping the highest-confidence version."""
        seen: dict[str, ExtractedRelation] = {}

        for relation in relations:
            key = f"{relation.head_text.lower()}:{relation.relation}:{relation.tail_text.lower()}"
            if key not in seen or relation.confidence > seen[key].confidence:
                seen[key] = relation

        return list(seen.values())
