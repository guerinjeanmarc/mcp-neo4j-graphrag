# Advanced Topics

This document covers advanced features and comparisons for `mcp-neo4j-graphrag`.

## Comparison with Neo4j Labs MCP Server

This server extends the functionality of the [Neo4j Labs `mcp-neo4j-cypher`](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher) server.

### Feature Comparison

| Feature | `mcp-neo4j-cypher` (Labs) | `mcp-neo4j-graphrag` (This) |
|---------|---------------------------|----------------------------|
| Schema discovery | `get_neo4j_schema` | `get_neo4j_schema_and_indexes` ✨ |
| Read Cypher | ✅ | ✅ |
| Write Cypher | ✅ | ❌ (read-only) |
| Vector search | ❌ | ✅ `vector_search` |
| Fulltext search | ❌ | ✅ `fulltext_search` |
| Search + Cypher | ❌ | ✅ `search_cypher_query` |
| Multi-provider embeddings | ❌ | ✅ (via LiteLLM) |
| Property size warnings | ❌ | ✅ |

### Key Additions

#### 1. Enhanced Schema Discovery

`get_neo4j_schema_and_indexes` extends the Labs server by:
- Listing vector and fulltext indexes
- Warning about large properties (e.g., "avg ~705KB")
- Helping LLMs avoid requesting token-heavy fields

#### 2. Vector Search

```python
vector_search(
    text_query="science fiction about AI",
    vector_index="moviePlotsEmbedding",
    top_k=10,
    return_properties="title,year,plot"
)
```

#### 3. Search-Augmented Cypher

The `search_cypher_query` tool lets LLMs combine search with graph traversal:

```python
search_cypher_query(
    vector_query="romantic comedy",
    cypher_query="""
        CALL db.index.vector.queryNodes('moviePlotsEmbedding', 100, $vector_embedding)
        YIELD node, score
        WHERE score > 0.8
        MATCH (node)-[:IN_GENRE]->(g:Genre)
        MATCH (node)<-[:ACTED_IN]-(a:Actor)
        RETURN node.title, collect(DISTINCT g.name) as genres, 
               collect(DISTINCT a.name)[0..3] as actors, score
        ORDER BY score DESC
        LIMIT 10
    """
)
```

---

## Production Features

Following [Neo4j's production-proofing best practices](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/), this server implements output control to prevent overwhelming LLM context windows.

### Layer 3: Size-Based Filtering

| Type | Limit | Behavior |
|------|-------|----------|
| Lists | ≥128 items | Replaced with `<list with 1536 items (truncated)>` |
| Strings | ≥10,000 chars | Truncated with `...<truncated at 10000 chars>` |

**Why this matters:**
- ✅ Blocks embedding arrays (typically 384-1536 floats)
- ✅ Truncates large text (OCR output, descriptions)
- ✅ Truncates base64 data (images stored as strings)
- ✅ Preserves small, useful values

### Layer 4: Token-Aware Truncation

- Measures response size using `tiktoken`
- Limit: 8,000 tokens
- Drops results from the end if over limit
- Adds warning: `"Results truncated from 100 to 45 items"`

### Example

**Before (no protection):**
```json
{
  "embedding": [0.023, 0.156, ...1536 floats...],
  "extractedText": "...50000 chars...",
  "extractedImage": "...base64 100000 chars..."
}
```

**After (with protection):**
```json
{
  "embedding": "<list with 1536 items (truncated, limit: 128)>",
  "extractedText": "Lorem ipsum... <truncated at 10000 chars, total: 50000>",
  "extractedImage": "iVBORw0K... <truncated at 10000 chars, total: 100000>"
}
```

---

## Detailed Tool Reference

### `get_neo4j_schema_and_indexes`

Returns:
- Vector indexes (name, dimensions, properties)
- Fulltext indexes (name, properties)
- Node/relationship schema with property types
- Size warnings for large properties

### `vector_search`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `text_query` | Yes | - | Text to embed and search |
| `vector_index` | Yes | - | Name of vector index |
| `top_k` | No | 5 | Number of results |
| `return_properties` | No | all | Comma-separated property names |

**Performance:** Fetches `max(top_k × 2, 100)` results internally to avoid kANN local maximum issues.

### `fulltext_search`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `text_query` | Yes | - | Lucene query (supports AND, OR, wildcards) |
| `fulltext_index` | Yes | - | Name of fulltext index |
| `top_k` | No | 5 | Number of results |
| `return_properties` | No | all | Comma-separated property names |

**Lucene syntax examples:**
- `"Tom Hanks"` - exact phrase
- `Tom AND Hanks` - both terms
- `Tom*` - wildcard
- `Hanks~` - fuzzy match

### `read_neo4j_cypher`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `query` | Yes | - | Cypher query (read-only) |
| `params` | No | {} | Query parameters |

### `search_cypher_query`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `cypher_query` | Yes | - | Cypher with `$vector_embedding` or `$fulltext_text` |
| `vector_query` | No | - | Text to embed (use `$vector_embedding` in Cypher) |
| `fulltext_query` | No | - | Text for fulltext (use `$fulltext_text` in Cypher) |
| `params` | No | {} | Additional parameters |

---

## CLI Options

```bash
mcp-neo4j-graphrag --help
```

| Option | Env Variable | Default | Description |
|--------|--------------|---------|-------------|
| `--db-url` | `NEO4J_URI` | `bolt://localhost:7687` | Neo4j URI |
| `--username` | `NEO4J_USERNAME` | `neo4j` | Username |
| `--password` | `NEO4J_PASSWORD` | - | Password |
| `--database` | `NEO4J_DATABASE` | `neo4j` | Database |
| `--embedding-model` | `EMBEDDING_MODEL` | `text-embedding-3-small` | Model |
| `--transport` | `NEO4J_TRANSPORT` | `stdio` | `stdio`, `http`, `sse` |
| `--namespace` | `NEO4J_NAMESPACE` | - | Tool prefix |
| `--read-timeout` | `NEO4J_READ_TIMEOUT` | 30 | Query timeout (seconds) |

---

## References

- [Neo4j MCP Documentation](https://neo4j.com/developer/genai-ecosystem/model-context-protocol-mcp/)
- [Production-Proofing Your Neo4j Cypher MCP Server](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/)
- [LiteLLM Embedding Providers](https://docs.litellm.ai/docs/embedding/supported_embedding)

