# 🔍🕸️ Neo4j GraphRAG MCP Server

A unified Model Context Protocol (MCP) server combining **vector search**, **fulltext search**, **graph schema**, and **search-augmented Cypher queries** for powerful GraphRAG applications on Neo4j.

## 🌟 Overview

This MCP server provides everything needed for GraphRAG workflows:

1. **🔍 Discovery** - List vector/fulltext indexes and graph schema
2. **🎯 Simple Search** - Vector and fulltext search tools
3. **🕸️ Graph Queries** - Execute Cypher queries
4. **⚡ Search + Graph** - Combine search indexes with Cypher queries

## 🎯 The Key Innovation: `search_cypher_query`

This tool solves a critical limitation: **LLMs can't directly use Neo4j search indexes in Cypher** because:
- ❌ They don't have OpenAI API keys to generate embeddings
- ❌ They can't fetch 100-1000 results and post-filter
- ❌ They can't combine vector/fulltext search with graph traversal in one query

**Now they can!** ✅

### **Example Usage:**

```python
search_cypher_query(
    vector_query="legal requirements for students",
    cypher_query="""
        CALL db.index.vector.queryNodes('chunk_embedding_vector', 500, $vector_embedding)
        YIELD node, score
        WHERE score > 0.75
        MATCH (node)-[:BELONGS_TO]->(d:Document)-[:HAS_TOPIC]->(t:Topic)
        WHERE d.year >= 2020
        RETURN node.chunkId, d.title, collect(t.name) as topics, score
        ORDER BY score DESC
        LIMIT 20
    """
)
```

**What happens:**
1. Server generates embedding from "legal requirements for students"
2. Injects it as `$vector_embedding` in the Cypher query
3. Query searches 500 results, filters, traverses relationships, aggregates
4. Returns structured results

## 🧩 All Tools

### 1️⃣ **Discovery Tool**

#### `get_neo4j_schema_and_indexes(sample_size)`
**⚠️ IMPORTANT: Call this tool BEFORE using any search tools.**

Returns Neo4j graph schema with search indexes and property size warnings.

**Parameters:**
- `sample_size` - Number of nodes to sample for schema and property sizes (default: 1000)

**Returns (compact JSON):**
- Vector & fulltext indexes (for search)
- Node/relationship schemas with property types  
- **Property size warnings** (helps choose efficient `return_properties`)

**Example output:**
```json
{"indexes":{"vector":[{"name":"page_text_embeddings","properties":["embedding"]}],"fulltext":[]},"schema":{"Page":{"type":"node","properties":{"id":{"type":"String","indexed":true},"pageNumber":{"type":"Long"},"extractedText":{"type":"String","warning":"medium (avg ~12KB)"},"extractedImage":{"type":"String","warning":"very large (avg ~150KB, max ~200KB)"},"embedding":{"type":"List","warning":"large list (avg ~1536 items)"}}}}}
```

**How to use warnings:**
- `"very large (avg ~150KB)"` → ⚠️ **Avoid** returning this property (causes token truncation)
- `"large list (avg ~1536 items)"` → Likely embedding array (auto-excluded in `vector_search` only)

### 2️⃣ **Simple Search Tools**

#### `vector_search(text_query, vector_index, top_k, return_properties)`
Semantic similarity search using OpenAI embeddings.

**Parameters:**
- `text_query` - Query to embed and search (required)
- `vector_index` - Name of vector index (required)
- `top_k` - Number of results (default: 5)
- `return_properties` - Comma-separated property names (optional, e.g., "pageNumber,id")

**Automatic Sanitization (always applied):**
- Large lists (≥128 items) → placeholders
- Large strings (≥10K chars) → truncated
- Response limited to 8000 tokens

**Returns:**
- Node IDs, labels, properties (sanitized), similarity scores

**Examples:**
```python
# Returns all properties (sanitized)
vector_search("compliance requirements", "chunk_embedding_vector", top_k=10)

# Returns only specific properties (more efficient)
vector_search("compliance", "chunk_embedding_vector", top_k=10, 
              return_properties="pageNumber,id")
```

#### `fulltext_search(text_query, fulltext_index, top_k, return_properties)`
Text search with Lucene query syntax (AND, OR, wildcards, fuzzy, etc.).

**Parameters:**
- `text_query` - Query with Lucene syntax (required)
- `fulltext_index` - Name of fulltext index (required)
- `top_k` - Number of results (default: 5)
- `return_properties` - Comma-separated property names (optional, e.g., "title,category")

**Automatic Sanitization (always applied):**
- Large lists (≥128 items) → placeholders
- Large strings (≥10K chars) → truncated
- Response limited to 8000 tokens

**Returns:**
- Node/relationship IDs, labels/types, properties (sanitized), relevance scores

**Examples:**
```python
# Returns all properties (sanitized)
fulltext_search("legal AND compliance", "chunk_content_fulltext", top_k=10)

# Returns only specific properties (more efficient)
fulltext_search("legal", "chunk_content_fulltext", top_k=10,
                return_properties="title,category")
```

### 3️⃣ **Cypher Query Tools**

#### `read_neo4j_cypher(query, params)`
Execute standard read Cypher queries.

#### `search_cypher_query(vector_query, fulltext_query, cypher_query, params)` ⭐
**The game-changer!** Combine search indexes with Cypher queries.

**Placeholders:**
- `$vector_embedding` - Replaced with generated embedding vector
- `$fulltext_text` - Replaced with text string for fulltext search

## 🚀 Installation

### Local Development

```bash
cd mcp-neo4j-graphrag
uv venv
source .venv/bin/activate  # On Unix/macOS
uv pip install -e .
```

### Configuration with Cursor/Claude Desktop

Add to your `mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neo4j-graphrag": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-neo4j-graphrag",
        "run",
        "mcp-neo4j-graphrag",
        "--transport",
        "stdio"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "sk-your-api-key",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
      }
    }
  }
}
```

## 💡 Use Cases

### 1. **Post-filtering Large Result Sets**
```cypher
-- Get 1000 results, filter to 10
CALL db.index.vector.queryNodes('chunk_embedding', 1000, $vector_embedding)
YIELD node, score
WHERE node.category = 'legal' AND node.year = 2024 AND score > 0.8
RETURN node
LIMIT 10
```

### 2. **Search + Graph Traversal**
```cypher
-- Find relevant chunks, expand to documents and topics
CALL db.index.fulltext.queryNodes('content_index', $fulltext_text)
YIELD node, score
WHERE score > 5.0
MATCH (node)-[:BELONGS_TO]->(d:Document)-[:HAS_TOPIC]->(t:Topic)
RETURN node, d, collect(t) as topics, score
ORDER BY score DESC
LIMIT 20
```

### 3. **Aggregation Over Search Results**
```cypher
-- Search and aggregate by year
CALL db.index.vector.queryNodes('chunk_embedding', 500, $vector_embedding)
YIELD node
MATCH (node)-[:BELONGS_TO]->(d:Document)
RETURN d.year, count(*) as count, avg(node.score) as avgScore
ORDER BY count DESC
```

### 4. **Multi-Index Queries**
```cypher
-- Combine vector AND fulltext in single query
CALL db.index.vector.queryNodes('chunk_embedding', 200, $vector_embedding)
YIELD node as vNode, score as vScore
WITH vNode, vScore
CALL db.index.fulltext.queryNodes('chunk_content', $fulltext_text)
YIELD node as fNode, score as fScore
WHERE elementId(vNode) = elementId(fNode)
RETURN vNode, vScore, fScore, (vScore + fScore) as combined
ORDER BY combined DESC
```

## 🛡️ Production-Ready Output Control

This server implements [Neo4j's production-proofing best practices](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/) to prevent overwhelming LLM context windows:

### **Layer 3: Size-Based Filtering** 
Large values are automatically sanitized:

**Lists (>128 items)** replaced with placeholders:
```
"embedding": "<list with 1536 items (truncated, limit: 128)>"
```

**Strings (>10,000 chars)** truncated with suffix:
```
"extractedText": "Lorem ipsum dolor sit amet... <truncated at 10000 chars, total: 50000>"
"extractedImage": "iVBORw0KGgoAAAANSUhEUgAAA... <truncated at 10000 chars, total: 100000>"
```

**Benefits:**
- ✅ Blocks embeddings (typically 384-1536 floats)
- ✅ Truncates large text (OCR output, descriptions)
- ✅ Truncates binary data strings (base64 images, blobs)
- ✅ Blocks any large arrays
- ✅ Preserves small, useful values

**Configuration:** 
- `MAX_LIST_SIZE = 128` (production-proven value)
- `MAX_STRING_SIZE = 10000` (10K chars ~2500 tokens)

### **Layer 4: Token-Aware Truncation**
Results are measured using OpenAI's tiktoken and truncated if they exceed the limit:
```json
{
  "warning": "Results truncated from 100 to 45 items (token limit: 8000)",
  "results": [...]
}
```

**Benefits:**
- ✅ Guarantees responses never overflow LLM context
- ✅ Drops results from the end (least relevant first)
- ✅ Maintains response structure integrity
- ✅ Logs warnings for visibility

**Configuration:** Hardcoded to `RESPONSE_TOKEN_LIMIT = 8000` (conservative default)

### **Applied to All Tools**
Both layers protect:
- `vector_search`
- `fulltext_search`
- `read_neo4j_cypher`
- `search_cypher_query`

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `OPENAI_API_KEY` | **(required)** | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `NEO4J_NAMESPACE` | _(empty)_ | Tool namespace prefix |
| `NEO4J_TRANSPORT` | `stdio` | Transport protocol |
| `NEO4J_READ_TIMEOUT` | `30` | Query timeout (seconds) |
| `NEO4J_SCHEMA_SAMPLE_SIZE` | `1000` | Schema sampling size |

### Command Line

```bash
mcp-neo4j-graphrag \
  --db-url bolt://localhost:7687 \
  --username neo4j \
  --password your-password \
  --database neo4j \
  --openai-api-key sk-... \
  --embedding-model text-embedding-3-small \
  --transport stdio
```

## ✨ Key Features

- ✅ **Unified server** - All GraphRAG tools in one place
- ✅ **Search-augmented Cypher** - Use indexes directly in queries
- ✅ **Full property return** - Search tools return all node/relationship properties by default
- ✅ **Flexible property selection** - Optional `return_properties` for token efficiency
- ✅ **Production-ready output control** - Multi-layer protection against overwhelming LLM context:
  - **Layer 3: Size-based filtering** - Lists >128 items replaced with descriptive placeholders
  - **Layer 4: Token-aware truncation** - Results automatically truncated to 8000 tokens
- ✅ **OpenAI embeddings** - Automatic embedding generation
- ✅ **Post-filtering** - Fetch 100-1000 results, filter with WHERE
- ✅ **Graph traversal** - Combine search with MATCH patterns
- ✅ **Lucene syntax** - Full Lucene query support for fulltext
- ✅ **Query validation** - Read-only enforcement for safety
- ✅ **Performance optimized** - Fetches 2x results to avoid kANN local maximum

## 🎯 Typical Workflow

```
1. Agent: get_neo4j_schema_and_indexes()
   → Discovers indexes (chunk_embedding_vector, chunk_content_fulltext)
   → Learns graph schema: Chunk-[:BELONGS_TO]->Document-[:HAS_TOPIC]->Topic
   → Gets property size warnings to avoid large fields

2. Agent: search_cypher_query(
     vector_query="student requirements",
     cypher_query="... uses $vector_embedding with MATCH patterns ..."
   )
   → Gets semantically relevant chunks with full graph context

3. Agent: Synthesizes answer from rich graph results
```

## 📋 Prerequisites

1. **Neo4j database** with vector/fulltext indexes
2. **OpenAI API key** for embeddings
3. **Python 3.10+**
4. **APOC plugin** (for schema tool)

### Creating Indexes in Neo4j

```cypher
// Vector index
CREATE VECTOR INDEX chunk_embedding_vector IF NOT EXISTS
FOR (c:Chunk)
ON c.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}

// Fulltext index
CREATE FULLTEXT INDEX chunk_content_fulltext IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.content]
```

## 🔒 Security

- ✅ **Read-only by default** - `search_cypher_query` only allows read (MATCH) queries
- ✅ **Query validation** - Prevents write operations (CREATE, DELETE, etc.)
- ✅ **Parameter injection** - All queries use parameterized execution

## 📄 License

This MCP server is licensed under the MIT License.

## 🙏 Acknowledgments

This project is inspired by and extends the labs [Neo4j MCP servers](https://github.com/neo4j-contrib/mcp-neo4j), particularly the `mcp-neo4j-cypher` server. It adds specialized GraphRAG capabilities including vector search, fulltext search, and the innovative `search_cypher_query` tool.

## 🤝 Related Projects

- [Neo4j MCP Servers](https://github.com/neo4j-contrib/mcp-neo4j) - Labs Neo4j MCP servers (Cypher, Memory, Data Modeling, Aura)
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [Neo4j GraphRAG Python](https://github.com/neo4j/neo4j-graphrag-python) - GraphRAG library
- [Production-Proofing Your Neo4j Cypher MCP Server](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/) - Best practices followed in this implementation

---

**Built with ❤️ for powerful GraphRAG applications**

