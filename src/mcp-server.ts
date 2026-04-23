import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';
import http from 'http';
import { query } from './db';
import crypto from 'crypto';
import { URL } from 'url';

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
      description: 'Retrieve relevant memory chunks from Notion based on a query.',
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

// --- OAuth 2.0 PKCE state store (in-memory, sufficient for single instance) ---
const authCodes = new Map<string, { redirectUri: string; expiresAt: number }>();
const accessTokens = new Set<string>();

function generateToken(length = 32): string {
  return crypto.randomBytes(length).toString('base64url');
}

// --- HTTP server: health + OAuth + /context endpoint ---
function startHealthServer() {
  const port = parseInt(process.env.PORT || '3000');
  const BASE_URL = process.env.BASE_URL || `https://notion-memory-mcp.onrender.com`;

  const httpServer = http.createServer(async (req, res) => {
    const urlObj = new URL(req.url || '/', BASE_URL);
    const path = urlObj.pathname;

    // CORS headers for Claude.ai
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    // OAuth metadata discovery
    if (path === '/.well-known/oauth-authorization-server') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        issuer: BASE_URL,
        authorization_endpoint: `${BASE_URL}/authorize`,
        token_endpoint: `${BASE_URL}/token`,
        response_types_supported: ['code'],
        grant_types_supported: ['authorization_code'],
        code_challenge_methods_supported: ['S256'],
        token_endpoint_auth_methods_supported: ['none'],
      }));
      return;
    }

    // Authorization endpoint — immediately issue a code and redirect back
    if (path === '/authorize' && req.method === 'GET') {
      const redirectUri = urlObj.searchParams.get('redirect_uri') || '';
      const state = urlObj.searchParams.get('state') || '';
      const code = generateToken(24);
      authCodes.set(code, { redirectUri, expiresAt: Date.now() + 5 * 60 * 1000 });
      const redirectUrl = new URL(redirectUri);
      redirectUrl.searchParams.set('code', code);
      if (state) redirectUrl.searchParams.set('state', state);
      res.writeHead(302, { Location: redirectUrl.toString() });
      res.end();
      return;
    }

    // Token endpoint — exchange code for access token
    if (path === '/token' && req.method === 'POST') {
      let body = '';
      req.on('data', (chunk) => { body += chunk; });
      req.on('end', () => {
        const params = new URLSearchParams(body);
        const code = params.get('code') || '';
        const entry = authCodes.get(code);
        if (!entry || Date.now() > entry.expiresAt) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'invalid_grant' }));
          return;
        }
        authCodes.delete(code);
        const accessToken = generateToken(32);
        accessTokens.add(accessToken);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          access_token: accessToken,
          token_type: 'bearer',
          expires_in: 86400,
        }));
      });
      return;
    }

    // Health check
    if (path === '/health' || path === '/') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', service: 'notion-memory-mcp' }));
      return;
    }

    // /context endpoint for gateway
    if (path === '/context' && req.method === 'POST') {
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
      return;
    }

    res.writeHead(404);
    res.end('Not found');
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
