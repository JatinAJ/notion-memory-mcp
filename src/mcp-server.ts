import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';
import { query } from './db';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: 'text-embedding-3-small',
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
        'Retrieve relevant memory chunks from Notion based on a query. Use this to recall facts, notes, or context stored in Notion.',
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The query or topic to search for in Notion memory',
          },
          limit: {
            type: 'number',
            description: 'Max number of chunks to return (default 5)',
          },
        },
        required: ['query'],
      },
    },
    {
      name: 'memory.search',
      description:
        'Search Notion memory for specific terms or keywords using semantic similarity.',
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query string',
          },
          limit: {
            type: 'number',
            description: 'Max results to return (default 10)',
          },
          page_title_filter: {
            type: 'string',
            description: 'Optional: filter results to a specific Notion page title',
          },
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
    const limit = (args as any).limit ?? (name === 'memory.search' ? 10 : 5);
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
      content: [
        {
          type: 'text',
          text: JSON.stringify(chunks, null, 2),
        },
      ],
    };
  }

  throw new Error(`Unknown tool: ${name}`);
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Notion Memory MCP server running on stdio');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
