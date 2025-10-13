"""Generate synthetic document corpus for testing."""
import random
import json
from pathlib import Path
from typing import List, Dict, Any
from faker import Faker

fake = Faker()
Faker.seed(42)  # Deterministic generation
random.seed(42)


def generate_synthetic_corpus(output_dir: Path, num_docs: int = 200) -> List[Dict[str, Any]]:
    """Generate synthetic document corpus with diverse characteristics.
    
    Generates documents with:
    - Various lengths (short, medium, long, very long)
    - Overlapping facts for reranking tests
    - Near-duplicates for de-duplication tests
    - Simulated PII for redaction tests
    - Different topics and domains
    
    Args:
        output_dir: Directory to save generated documents
        num_docs: Number of documents to generate
        
    Returns:
        List of generated documents with metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    
    # Document templates by category
    categories = {
        "programming": [
            "Python is a high-level programming language known for its simplicity.",
            "JavaScript is the language of the web, running in browsers worldwide.",
            "Machine learning involves training models on data to make predictions.",
            "APIs allow different software systems to communicate with each other.",
        ],
        "legal": [
            "Section 42 defines fundamental freedoms and individual rights.",
            "Contract law governs agreements between parties and their enforcement.",
            "Intellectual property protects creative works and inventions.",
            "Privacy regulations like GDPR protect personal data.",
        ],
        "medical": [
            "Disclaimer: This is simulated medical information for testing only.",
            "Regular exercise and healthy diet contribute to overall wellness.",
            "Preventive care includes regular checkups and screenings.",
            "Mental health is as important as physical health.",
        ],
        "business": [
            "Quarterly earnings reports provide financial performance insights.",
            "Market analysis helps companies understand competitive landscapes.",
            "Customer relationship management improves client satisfaction.",
            "Supply chain optimization reduces costs and improves efficiency.",
        ],
        "science": [
            "The scientific method involves hypothesis, experimentation, and analysis.",
            "Climate change is driven by greenhouse gas emissions.",
            "Quantum mechanics describes behavior at atomic scales.",
            "Biodiversity is essential for ecosystem stability.",
        ],
    }
    
    # PII templates for redaction tests
    pii_templates = [
        "Contact: {name} at {email} or {phone}.",
        "SSN: {ssn}, DOB: {dob}",
        "Credit Card: {cc_number}, Exp: {cc_exp}",
        "Address: {address}",
    ]
    
    doc_id = 0
    
    # Generate short documents (100-300 tokens)
    for _ in range(num_docs // 5):
        category = random.choice(list(categories.keys()))
        text = ". ".join(random.sample(categories[category], k=min(2, len(categories[category]))))
        text += f". {fake.paragraph(nb_sentences=2)}"
        
        doc = create_document(doc_id, text, category, "short")
        documents.append(doc)
        doc_id += 1
        
    # Generate medium documents (500-1500 tokens)
    for _ in range(num_docs // 5):
        category = random.choice(list(categories.keys()))
        text = ". ".join(categories[category])
        text += f"\n\n{fake.paragraph(nb_sentences=10)}"
        text += f"\n\n{fake.paragraph(nb_sentences=10)}"
        
        doc = create_document(doc_id, text, category, "medium")
        documents.append(doc)
        doc_id += 1
        
    # Generate long documents (2000-5000 tokens)
    for _ in range(num_docs // 5):
        category = random.choice(list(categories.keys()))
        text = f"# {fake.catch_phrase()}\n\n"
        text += ". ".join(categories[category]) + "\n\n"
        
        for _ in range(5):
            text += f"## {fake.bs()}\n\n"
            text += fake.paragraph(nb_sentences=20) + "\n\n"
            
        doc = create_document(doc_id, text, category, "long")
        documents.append(doc)
        doc_id += 1
        
    # Generate very long documents (10000+ tokens)
    for _ in range(num_docs // 10):
        category = random.choice(list(categories.keys()))
        text = f"# {fake.catch_phrase()} - Complete Guide\n\n"
        
        for section in range(10):
            text += f"## Section {section + 1}: {fake.bs()}\n\n"
            for _ in range(10):
                text += fake.paragraph(nb_sentences=15) + "\n\n"
                
        doc = create_document(doc_id, text, category, "very_long")
        documents.append(doc)
        doc_id += 1
        
    # Generate documents with overlapping facts (for reranking tests)
    overlap_fact = "The answer to the ultimate question is 42."
    for i in range(5):
        text = f"{fake.paragraph(nb_sentences=5)} {overlap_fact} {fake.paragraph(nb_sentences=5)}"
        doc = create_document(doc_id, text, "science", "medium")
        doc["metadata"]["has_overlap"] = True
        documents.append(doc)
        doc_id += 1
        
    # Generate near-duplicate documents (for de-duplication tests)
    base_text = fake.paragraph(nb_sentences=20)
    for i in range(3):
        # Add small variations
        text = base_text + f" Additional note {i}: {fake.sentence()}"
        doc = create_document(doc_id, text, "business", "medium")
        doc["metadata"]["duplicate_group"] = "dup1"
        documents.append(doc)
        doc_id += 1
        
    # Generate documents with PII (for redaction tests)
    for _ in range(10):
        pii_template = random.choice(pii_templates)
        pii_data = {
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "ssn": fake.ssn(),
            "dob": fake.date_of_birth().isoformat(),
            "cc_number": fake.credit_card_number(),
            "cc_exp": fake.credit_card_expire(),
            "address": fake.address().replace("\n", ", "),
        }
        text = pii_template.format(**pii_data)
        text += f"\n\n{fake.paragraph(nb_sentences=5)}"
        
        doc = create_document(doc_id, text, "sensitive", "short")
        doc["metadata"]["contains_pii"] = True
        doc["metadata"]["pii_types"] = list(pii_data.keys())
        documents.append(doc)
        doc_id += 1
        
    # Fill remaining with random documents
    while len(documents) < num_docs:
        category = random.choice(list(categories.keys()))
        text = fake.paragraph(nb_sentences=random.randint(5, 30))
        doc = create_document(doc_id, text, category, "random")
        documents.append(doc)
        doc_id += 1
        
    # Save documents to files
    for doc in documents:
        doc_file = output_dir / f"{doc['id']}.json"
        with open(doc_file, 'w') as f:
            json.dump(doc, f, indent=2)
            
    # Save corpus metadata
    metadata_file = output_dir / "corpus_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "num_documents": len(documents),
            "categories": list(categories.keys()),
            "length_distribution": {
                "short": sum(1 for d in documents if d["metadata"]["length_category"] == "short"),
                "medium": sum(1 for d in documents if d["metadata"]["length_category"] == "medium"),
                "long": sum(1 for d in documents if d["metadata"]["length_category"] == "long"),
                "very_long": sum(1 for d in documents if d["metadata"]["length_category"] == "very_long"),
            },
            "special_docs": {
                "with_pii": sum(1 for d in documents if d["metadata"].get("contains_pii", False)),
                "with_overlap": sum(1 for d in documents if d["metadata"].get("has_overlap", False)),
                "duplicates": sum(1 for d in documents if "duplicate_group" in d["metadata"]),
            }
        }, f, indent=2)
        
    print(f"Generated {len(documents)} synthetic documents in {output_dir}")
    return documents


def create_document(doc_id: int, text: str, category: str, length_category: str) -> Dict[str, Any]:
    """Create document with metadata."""
    word_count = len(text.split())
    
    return {
        "id": f"doc{doc_id:04d}",
        "text": text,
        "metadata": {
            "source": f"{category}_{length_category}_{doc_id}.pdf",
            "category": category,
            "length_category": length_category,
            "word_count": word_count,
            "author": fake.name(),
            "created_date": fake.date_time_this_year().isoformat(),
            "page_count": max(1, word_count // 250),
        }
    }


if __name__ == "__main__":
    # Generate corpus when run directly
    output_dir = Path("tests/fixtures/corpus")
    generate_synthetic_corpus(output_dir, num_docs=200)
