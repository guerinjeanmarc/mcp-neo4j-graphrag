# Production-Ready Features

This document describes the production-ready output control features implemented in `mcp-neo4j-graphrag`.

## 🎯 Implementation Summary

Following [Neo4j's production-proofing best practices](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/), we implemented **Layers 3 & 4** to prevent overwhelming LLM context windows.

---

## 📊 Layer 3: Size-Based Filtering

**Constants:** 
- `MAX_LIST_SIZE = 128`
- `MAX_STRING_SIZE = 10000`

**What it does:**
- Recursively scans all properties in results
- **Lists:** Replaces lists with ≥128 items with descriptive placeholders
  - Example: `"embedding": "<list with 1536 items (truncated, limit: 128)>"`
- **Strings:** Truncates strings with ≥10K chars with descriptive suffix
  - Example: `"extractedText": "Lorem ipsum... <truncated at 10000 chars, total: 50000>"`

**Why it matters:**
- ✅ Automatically blocks embeddings (typically 384-1536 floats)
- ✅ Truncates large text properties (OCR output, long descriptions)
- ✅ Truncates binary data stored as strings (base64 images, blobs)
- ✅ Blocks any large array property
- ✅ Preserves small, useful values (tags, short text, small lists)

**Applied to:**
- `vector_search()` - sanitizes node properties
- `fulltext_search()` - sanitizes node/relationship properties  
- `read_neo4j_cypher()` - sanitizes all query results
- `search_cypher_query()` - sanitizes all query results

---

## 🎯 Layer 4: Token-Aware Truncation

**Constant:** `RESPONSE_TOKEN_LIMIT = 8000`

**What it does:**
- Measures total response size using OpenAI's `tiktoken`
- If over limit, drops results from the end (least relevant first)
- Adds warning to response: `"warning": "Results truncated from 100 to 45 items (token limit: 8000)"`

**Why it matters:**
- ✅ **Guarantees** responses never overflow LLM context
- ✅ Prevents context window exhaustion
- ✅ Maintains response structure integrity
- ✅ Smart truncation (drops least relevant results first)

**Applied to:**
- `vector_search()` - after sanitization, before return
- `fulltext_search()` - after sanitization, before return
- `read_neo4j_cypher()` - after sanitization, before return
- `search_cypher_query()` - after sanitization, before return

---

## 🔄 Processing Flow

```
1. Execute Neo4j query
   ↓
2. Get results from database
   ↓
3. Apply Layer 3: Size-based filtering
   - Replace large lists with placeholders
   ↓
4. Apply Layer 4: Token-aware truncation
   - Count tokens
   - Drop results if over limit
   ↓
5. Return sanitized, truncated results
```

---

## 📝 Code Locations

### `utils.py`
```python
MAX_LIST_SIZE = 128                    # Layer 3 config (lists)
MAX_STRING_SIZE = 10000                # Layer 3 config (strings)
RESPONSE_TOKEN_LIMIT = 8000            # Layer 4 config

_value_sanitize(d, list_limit, string_limit)  # Layer 3 implementation
_count_tokens(text, model)                     # Layer 4 helper
_truncate_results_to_token_limit(...)          # Layer 4 implementation
```

### `server.py`
Applied in all 4 tools:
- `vector_search()`
- `fulltext_search()`
- `read_neo4j_cypher()`
- `search_cypher_query()`

---

## 🧪 Example Outputs

### Without Protection (Old Behavior)
```json
{
  "results": [
    {
      "nodeId": "123",
      "properties": {
        "title": "Document",
        "embedding": [0.023, 0.156, ...1536 floats...],           // ❌ 6KB+ of noise
        "extractedText": "Lorem ipsum dolor sit amet... 50000 chars", // ❌ 200KB+ of text
        "extractedImage": "iVBORw0KGgoAAAANSUhEUgAA... 100000 chars" // ❌ 400KB+ base64
      }
    },
    // ... 99 more results ...
  ]
}
// Total: 60MB+, overwhelming LLM context
```

### With Protection (New Behavior)
```json
{
  "results": [
    {
      "nodeId": "123",
      "properties": {
        "title": "Document",
        "embedding": "<list with 1536 items (truncated, limit: 128)>",     // ✅ Clean
        "extractedText": "Lorem ipsum... <truncated at 10000 chars, total: 50000>",  // ✅ Clean
        "extractedImage": "iVBORw0K... <truncated at 10000 chars, total: 100000>"    // ✅ Clean
      }
    },
    // ... 4 more results (truncated from 100) ...
  ],
  "warning": "Results truncated from 100 to 5 items (token limit: 8000)"
}
// Total: 7.8KB, LLM-friendly
```

---

## ⚙️ Configuration

Currently **hardcoded** for simplicity:
```python
MAX_LIST_SIZE = 128           # Production-proven value from Neo4j
MAX_STRING_SIZE = 10000       # 10K chars ≈ 2500 tokens (conservative)
RESPONSE_TOKEN_LIMIT = 8000   # Conservative default for full response
```

**Why these values:**
- `128` items: Catches embeddings (384-1536 dims) and large arrays
- `10000` chars: Allows useful text (~2500 tokens) but blocks huge OCR/base64
- `8000` tokens: Conservative limit for full responses

Future enhancement: Could be made configurable via environment variables if needed.

---

## 📚 References

- [Production-Proofing Your Neo4j Cypher MCP Server](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/)
- [Implementing Neo4j GraphRAG Retrievers as MCP Server](https://medium.com/neo4j/implementing-neo4j-graphrag-retrievers-as-mcp-server-77162e1d2b40)

---

**Status:** ✅ Implemented in v0.2.0, enhanced in v0.3.0

