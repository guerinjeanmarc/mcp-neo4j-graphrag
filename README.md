# Neo4j GraphRAG MCP Server

An MCP server that extends Neo4j with **vector search**, **fulltext search**, and **search-augmented Cypher queries** for GraphRAG applications.

> **Inspired by** the [Neo4j Labs `mcp-neo4j-cypher`](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher) server. This server adds vector search, fulltext search, and the innovative `search_cypher_query` tool for combining search with graph traversal.

## Overview

This server enables LLMs to:
- 🔍 Search Neo4j vector indexes using semantic similarity
- 📝 Search fulltext indexes with Lucene syntax
- ⚡ Combine search with Cypher queries via `search_cypher_query`
- 🕸️ Execute read-only Cypher queries

Built on [LiteLLM](https://docs.litellm.ai/) for multi-provider embedding support (OpenAI, Azure, Bedrock, Cohere, etc.).

> **Related:** For the official Neo4j MCP Server, see [neo4j/mcp](https://github.com/neo4j/mcp). For Neo4j Labs MCP Servers (Cypher, Memory, Data Modeling), see [neo4j-contrib/mcp-neo4j](https://github.com/neo4j-contrib/mcp-neo4j).

## Installation

### Step 1: Download the Repository

```bash
git clone https://github.com/guerinjeanmarc/mcp-neo4j-graphrag.git
cd mcp-neo4j-graphrag
```

### Step 2: Configure Your MCP Client

#### Claude Desktop

Edit the configuration file:
- **macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add this server configuration (update the path to where you cloned the repo):

```json
{
  "mcpServers": {
    "neo4j-graphrag": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-neo4j-graphrag",
        "run",
        "mcp-neo4j-graphrag"
      ],
      "env": {
        "NEO4J_URI": "neo4j+s://demo.neo4jlabs.com",
        "NEO4J_USERNAME": "recommendations",
        "NEO4J_PASSWORD": "recommendations",
        "NEO4J_DATABASE": "recommendations",
        "OPENAI_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-ada-002"
      }
    }
  }
}
```

#### Cursor

Edit `.cursor/mcp.json` in your project or global settings. Use the same configuration as above.

### Step 3: Reload Configuration

- **Claude Desktop:** Quit and restart the application
- **Cursor:** Reload the window (Cmd/Ctrl + Shift + P → "Reload Window")

## Tools

### `get_neo4j_schema_and_indexes`

Discover the graph schema, vector indexes, and fulltext indexes.

💡 The agent should automatically call this tool first before using other tools to understand the schema and indexes of the database.

**Example prompt:**
> "What indexes and node types are available in the database?"

### `vector_search`

Semantic similarity search using embeddings.

**Parameters:** `text_query`, `vector_index`, `top_k`, `return_properties`

**Example prompt:**
> "Search for movies about artificial intelligence using the moviePlotsEmbedding index"

### `fulltext_search`

Keyword search with Lucene syntax (AND, OR, wildcards, fuzzy).

**Parameters:** `text_query`, `fulltext_index`, `top_k`, `return_properties`

**Example prompt:**
> "Find people named 'Tom' using the personFulltext index"

### `read_neo4j_cypher`

Execute read-only Cypher queries.

**Parameters:** `query`, `params`

**Example prompt:**
> "Show me all genres and how many movies are in each"

### `search_cypher_query`

Combine vector/fulltext search with Cypher queries. Use `$vector_embedding` and `$fulltext_text` placeholders.

**Parameters:** `cypher_query`, `vector_query`, `fulltext_query`, `params`

**Example prompt:**
> "Find movies similar to 'time travel adventure', then show me their directors and genres"

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | `password` | Neo4j password |
| `NEO4J_DATABASE` | No | `neo4j` | Database name |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model (see below) |

### Embedding Providers

Set `EMBEDDING_MODEL` and the corresponding API key:

| Provider | Model Format | API Key Variable |
|----------|-------------|------------------|
| OpenAI | `text-embedding-ada-002` | `OPENAI_API_KEY` |
| Azure | `azure/deployment-name` | `AZURE_API_KEY`, `AZURE_API_BASE` |
| Bedrock | `bedrock/amazon.titan-embed-text-v1` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Cohere | `cohere/embed-english-v3.0` | `COHERE_API_KEY` |
| Ollama | `ollama/nomic-embed-text` | *(none - local)* |

## Advanced Topics

See [docs/ADVANCED.md](docs/ADVANCED.md) for:
- Comparison with Neo4j Labs `mcp-neo4j-cypher` server
- Production features (output sanitization, token limits)
- Detailed tool documentation

## License

MIT License
