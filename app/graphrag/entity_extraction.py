"""Entity and relation extraction for GraphRAG.

Extracts structured knowledge (entities, relations, attributes) from text
for storage in a knowledge graph. Supports multiple extraction strategies:
- LLM-based extraction (GPT, Claude, Ollama)
- NER-based extraction (spaCy, Stanza)
- Hybrid extraction (combines both)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import re

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Common entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    DATE = "DATE"
    OTHER = "OTHER"


class RelationType(str, Enum):
    """Common relation types."""
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"
    FOUNDED = "FOUNDED"
    ACQUIRED = "ACQUIRED"
    CREATED = "CREATED"
    HAPPENED_AT = "HAPPENED_AT"
    OTHER = "OTHER"


@dataclass
class Entity:
    """Represents an entity extracted from text."""
    name: str
    entity_type: EntityType
    mentions: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "mentions": self.mentions,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "source_text": self.source_text,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            entity_type=EntityType(data["type"]),
            mentions=data.get("mentions", []),
            attributes=data.get("attributes", {}),
            confidence=data.get("confidence", 1.0),
            source_text=data.get("source_text", ""),
        )


@dataclass
class Relation:
    """Represents a relation between entities."""
    source: str
    target: str
    relation_type: RelationType
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=RelationType(data["type"]),
            attributes=data.get("attributes", {}),
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence", ""),
        )


@dataclass
class ExtractionResult:
    """Result of entity/relation extraction."""
    entities: List[Entity]
    relations: List[Relation]
    source_text: str
    extraction_method: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """Extract entities and relations from text using LLM."""
    
    def __init__(
        self,
        model_adapter,
        extraction_prompt: Optional[str] = None,
        min_confidence: float = 0.5,
        max_entities_per_doc: int = 50,
    ):
        """Initialize entity extractor.
        
        Args:
            model_adapter: LLM adapter for extraction
            extraction_prompt: Custom extraction prompt template
            min_confidence: Minimum confidence threshold
            max_entities_per_doc: Maximum entities to extract per document
        """
        self.model = model_adapter
        self.extraction_prompt = extraction_prompt or self._default_prompt()
        self.min_confidence = min_confidence
        self.max_entities_per_doc = max_entities_per_doc
    
    def _default_prompt(self) -> str:
        """Default extraction prompt."""
        return """Extract entities and relations from the following text.

For each entity, identify:
- name: The entity name
- type: One of [PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, CONCEPT, DATE, OTHER]
- attributes: Any relevant attributes (e.g., title, description)

For each relation, identify:
- source: Source entity name
- target: Target entity name
- type: One of [IS_A, PART_OF, RELATED_TO, LOCATED_IN, WORKS_FOR, FOUNDED, ACQUIRED, CREATED, HAPPENED_AT, OTHER]
- evidence: Supporting text from the document

Return as JSON:
{
  "entities": [
    {"name": "...", "type": "...", "attributes": {...}},
    ...
  ],
  "relations": [
    {"source": "...", "target": "...", "type": "...", "evidence": "..."},
    ...
  ]
}

Text:
{text}

JSON:"""
    
    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities and relations from text.
        
        Args:
            text: Input text to extract from
            context: Optional context for extraction
            
        Returns:
            ExtractionResult with entities and relations
        """
        try:
            # Prepare prompt
            prompt = self.extraction_prompt.format(text=text[:3000])  # Limit text length
            
            # Call LLM
            response = await self.model.generate(prompt, temperature=0.1, max_tokens=2000)
            
            # Parse response
            result = self._parse_extraction_response(response, text)
            
            # Filter by confidence
            result.entities = [e for e in result.entities if e.confidence >= self.min_confidence]
            result.relations = [r for r in result.relations if r.confidence >= self.min_confidence]
            
            # Limit entities
            if len(result.entities) > self.max_entities_per_doc:
                result.entities = sorted(
                    result.entities,
                    key=lambda e: e.confidence,
                    reverse=True
                )[:self.max_entities_per_doc]
            
            logger.info(
                f"Extracted {len(result.entities)} entities and {len(result.relations)} relations"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                extraction_method="llm",
                confidence=0.0,
                metadata={"error": str(e)},
            )
    
    def _parse_extraction_response(self, response: str, source_text: str) -> ExtractionResult:
        """Parse LLM extraction response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            
            # Parse entities
            entities = []
            for e_data in data.get("entities", []):
                entity = Entity(
                    name=e_data["name"],
                    entity_type=EntityType(e_data.get("type", "OTHER")),
                    mentions=[e_data["name"]],
                    attributes=e_data.get("attributes", {}),
                    confidence=e_data.get("confidence", 0.8),
                    source_text=source_text,
                )
                entities.append(entity)
            
            # Parse relations
            relations = []
            for r_data in data.get("relations", []):
                relation = Relation(
                    source=r_data["source"],
                    target=r_data["target"],
                    relation_type=RelationType(r_data.get("type", "OTHER")),
                    attributes=r_data.get("attributes", {}),
                    confidence=r_data.get("confidence", 0.8),
                    evidence=r_data.get("evidence", ""),
                )
                relations.append(relation)
            
            return ExtractionResult(
                entities=entities,
                relations=relations,
                source_text=source_text,
                extraction_method="llm",
                confidence=0.8,
            )
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            logger.debug(f"Response: {response}")
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=source_text,
                extraction_method="llm",
                confidence=0.0,
                metadata={"parse_error": str(e)},
            )
    
    async def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 5,
    ) -> List[ExtractionResult]:
        """Extract from multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process in parallel
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.extract(text) for text in batch],
                return_exceptions=True,
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction error: {result}")
                    results.append(ExtractionResult(
                        entities=[],
                        relations=[],
                        source_text="",
                        extraction_method="llm",
                        confidence=0.0,
                    ))
                else:
                    results.append(result)
        
        return results


class NERExtractor:
    """Extract entities using NER models (spaCy, Stanza)."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize NER extractor.
        
        Args:
            model_name: spaCy model name
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model."""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except ImportError:
            logger.warning("spaCy not installed, NER extraction unavailable")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using NER.
        
        Args:
            text: Input text
            
        Returns:
            ExtractionResult with entities
        """
        if not self.nlp:
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                extraction_method="ner",
                confidence=0.0,
            )
        
        try:
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                entity = Entity(
                    name=ent.text,
                    entity_type=entity_type,
                    mentions=[ent.text],
                    attributes={"label": ent.label_},
                    confidence=0.9,
                    source_text=text,
                )
                entities.append(entity)
            
            return ExtractionResult(
                entities=entities,
                relations=[],
                source_text=text,
                extraction_method="ner",
                confidence=0.9,
            )
            
        except Exception as e:
            logger.error(f"NER extraction error: {e}")
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                extraction_method="ner",
                confidence=0.0,
            )
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy label to EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "DATE": EntityType.DATE,
        }
        return mapping.get(label, EntityType.OTHER)


class HybridExtractor:
    """Combine LLM and NER extraction for best results."""
    
    def __init__(
        self,
        llm_extractor: EntityExtractor,
        ner_extractor: Optional[NERExtractor] = None,
    ):
        """Initialize hybrid extractor.
        
        Args:
            llm_extractor: LLM-based extractor
            ner_extractor: Optional NER extractor
        """
        self.llm_extractor = llm_extractor
        self.ner_extractor = ner_extractor or NERExtractor()
    
    async def extract(self, text: str) -> ExtractionResult:
        """Extract using both methods and merge results.
        
        Args:
            text: Input text
            
        Returns:
            Merged extraction result
        """
        # Run both extractors
        llm_result = await self.llm_extractor.extract(text)
        ner_result = self.ner_extractor.extract(text)
        
        # Merge entities (deduplicate by name)
        entity_map: Dict[str, Entity] = {}
        
        for entity in llm_result.entities + ner_result.entities:
            name_key = entity.name.lower()
            if name_key not in entity_map:
                entity_map[name_key] = entity
            else:
                # Keep higher confidence entity
                if entity.confidence > entity_map[name_key].confidence:
                    entity_map[name_key] = entity
        
        merged_entities = list(entity_map.values())
        
        return ExtractionResult(
            entities=merged_entities,
            relations=llm_result.relations,  # Only LLM extracts relations
            source_text=text,
            extraction_method="hybrid",
            confidence=(llm_result.confidence + ner_result.confidence) / 2,
            metadata={
                "llm_entities": len(llm_result.entities),
                "ner_entities": len(ner_result.entities),
                "merged_entities": len(merged_entities),
            },
        )
