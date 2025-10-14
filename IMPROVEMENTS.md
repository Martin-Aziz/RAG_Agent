# RAG System Improvements - Out-of-Scope Query Handling

## Problem Statement
The system was attempting to answer questions that were outside the scope of the available documents, leading to irrelevant or misleading responses. For example:
- "who is messi" → returned information about Python/FAISS
- "which model do you use" → returned greetings with unrelated evidence
- "who are you" → returned information about Inception

## Root Cause
The orchestrator was performing retrieval for all queries and attempting to synthesize answers from whatever documents it found, regardless of relevance to the user's question.

## Solutions Implemented

### 1. **Meta Question Detection** (`core/orchestrator.py`)
Added detection for questions about the system itself:
```python
META_QUESTIONS = ["who are you", "what are you", "what is your name", 
                  "what can you do", "which model", "what model", 
                  "what llm", "which llm"]
```

When a meta question is detected, the system now responds:
> "I'm a document-grounded assistant that can only answer questions based on the provided documents in the knowledge base. I don't have information about myself or the models I use - I can only help you explore the documents you've uploaded."

### 2. **Relevance Scoring Function**
Implemented `_calculate_relevance_score()` that:
- Extracts meaningful terms from the user query
- Removes common stop words
- Calculates keyword overlap between query and retrieved evidence
- Returns a relevance score (0.0 to 1.0)

### 3. **Relevance Threshold Enforcement**
After retrieval, the system now:
1. Calculates relevance score of evidence to the query
2. If score < 0.3 (low relevance), refuses to answer with:
   > "I'm sorry — the provided documents don't contain information about that topic. I can only answer questions based on the documents in the knowledge base."
3. Returns empty evidence list (no misleading citations)

### 4. **Enhanced Confidence Scoring**
Adjusted confidence scores based on relevance:
- High relevance (>0.5): confidence 0.7
- Low relevance (0.3-0.5): confidence 0.4
- Very low relevance (<0.3): confidence 0.1 + refusal

## Expected Behavior After Fix

### ✅ Greetings
**Query:** "hi"
**Response:** "Hello! I'm here to help you explore the knowledge base. Feel free to ask me anything about your documents."
**Evidence:** None

### ✅ Meta Questions
**Query:** "who are you" / "which model do you use"
**Response:** Explanation that system only works with documents
**Evidence:** None

### ✅ Out-of-Scope Questions
**Query:** "who is messi"
**Response:** "I'm sorry — the provided documents don't contain information about that topic. I can only answer questions based on the documents in the knowledge base."
**Evidence:** None

### ✅ In-Scope Questions
**Query:** "who directed inception"
**Response:** "Inception was directed by Christopher Nolan."
**Evidence:** Relevant passages from documents with proper citations

## Technical Details

### Changes Made:
1. **`core/orchestrator.py`**:
   - Added `META_QUESTIONS` list
   - Added `_is_meta_question()` function
   - Added `_calculate_relevance_score()` function
   - Modified `handle_query()` to check for meta questions
   - Modified `_handle_traditional_parrag()` to check relevance
   - Modified `_handle_advanced_parrag()` to check relevance

### Algorithm:
```
1. Check if query is a greeting → return greeting
2. Check if query is a meta question → return explanation
3. Perform retrieval
4. Calculate relevance_score(query, evidence)
5. If relevance_score < 0.3:
   → Return refusal message with no evidence
6. Else:
   → Generate answer from relevant evidence
```

## Testing Recommendations

Test with:
1. ✅ Greetings: "hi", "hello", "hey"
2. ✅ Meta questions: "who are you", "which model", "what can you do"
3. ✅ Out-of-scope: "who is messi", "what is the weather", "tell me about cars"
4. ✅ In-scope: "who directed inception", "what is Python", "what is FAISS"
5. ✅ Ambiguous: Questions that might partially match document content

## Additional Improvements Possible

1. **Semantic relevance scoring** using embeddings instead of keyword matching
2. **User feedback loop** to improve relevance thresholds
3. **Query classification** before retrieval to filter out obvious mismatches
4. **Domain detection** to inform users about available document topics
5. **Confidence calibration** based on retrieval scores and model uncertainty

## Conclusion

The system now properly refuses to answer questions outside its document scope, providing clear, honest responses that align with the principle of document-grounded assistance. This eliminates hallucinations and improves user trust in the system.
