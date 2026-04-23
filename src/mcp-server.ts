import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';
import http from 'http';
import { query } from './db';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
    input: text,
  });
  return res.data[0].embedding;
}

const server = new Server(
  { name: 'notion-memory', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'memory.get_context',
      description:
        'Retrieve relevant memory chunks from Notion based on a query.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The query to search for' },
          limit: { type: 'number', description: 'Max chunks to return (default 5)' },
        },
        required: ['query'],
      },
    },
    {
      name: 'memory.search',
      description: 'Search Notion memory using semantic similarity.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query string' },
          limit: { type: 'number', description: 'Max results (default 10)' },
          page_title_filter: { type: 'string', description: 'Filter by page title' },
        },
        required: ['query'],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === 'memory.get_context' || name === 'memory.search') {
    const userQuery = (args as any).query as string;
    const topK = parseInt(process.env.TOP_K || '5');
    const limit = (args as any).limit ?? (name === 'memory.search' ? topK * 2 : topK);
    const pageTitleFilter = (args as any).page_title_filter as string | undefined;

    const embedding = await getEmbedding(userQuery);

    let sql = `
      SELECT page_title, chunk_index, content,
             1 - (embedding <=> $1::vector) AS similarity
      FROM memory_chunks
    `;
    const params: any[] = [JSON.stringify(embedding)];

    if (pageTitleFilter) {
      sql += ` WHERE page_title ILIKE $2`;
      params.push(`%${pageTitleFilter}%`);
    }

    sql += ` ORDER BY embedding <=> $1::vector LIMIT $${params.length + 1}`;
    params.push(limit);

    const result = await query(sql, params);
    const chunks = result.rows.map((row: any) => ({
      page_title: row.page_title,
      chunk_index: row.chunk_index,
      content: row.content,
      similarity: parseFloat(row.similarity).toFixed(4),
    }));

    return {
      content: [{ type: 'text', text: JSON.stringify(chunks, null, 2) }],
    };
  }

  throw new Error(`Unknown tool: ${name}`);
});

// HTTP server: health check + /context endpoint for gateway
function startHealthServer() {
  const port = parseInt(process.env.PORT || '3000');
  const httpServer = http.createServer(async (req, res) => {
    if (req.url === '/health' || req.url === '/') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', service: 'notion-memory-mcp' }));
    } else if (req.url === '/context' && req.method === 'POST') {
      // Gateway calls this to get memory context
      let body = '';
      req.on('data', (chunk) => { body += chunk; });
      req.on('end', async () => {
        try {
          const { query: q, topK = 5 } = JSON.parse(body) as { query: string; topK?: number };
          const embedding = await getEmbedding(q);
          const result = await query(
            `SELECT page_title, content,
                    1 - (embedding <=> $1::vector) AS similarity
             FROM memory_chunks
             ORDER BY embedding <=> $1::vector
             LIMIT $2`,
            [JSON.stringify(embedding), topK]
          );
          const contextText = result.rows
            .map((r: any) => `[${r.page_title}]\n${r.content}`)
            .join('\n\n---\n\n');
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ context: contextText, chunks: result.rows.length }));
        } catch (err) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: String(err) }));
        }
      });
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  httpServer.listen(port, () => {
    console.error(`MCP HTTP server listening on port ${port}`);
  });
}

async function main() {
  startHealthServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Notion Memory MCP server running on stdio');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
