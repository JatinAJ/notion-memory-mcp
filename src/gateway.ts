import express, { Request, Response } from 'express';
import OpenAI from 'openai';
import { query } from './db';
import { getEmbedding } from './mcp-server';

const app = express();
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MCP_URL = process.env.MCP_URL || 'https://notion-memory-mcp.onrender.com';

// Health check
app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

// Main chat endpoint
// POST /chat  { message: string, model?: string }
app.post('/chat', async (req: Request, res: Response) => {
  try {
    const { message, model = 'gpt-4o' } = req.body as { message: string; model?: string };
    if (!message) {
      res.status(400).json({ error: 'message is required' });
      return;
    }

    // 1) Fetch memory context from MCP server
    let context = '';
    try {
      const mcpRes = await fetch(`${MCP_URL}/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message, topK: 5 }),
      });
      if (mcpRes.ok) {
        const data = await mcpRes.json() as { context?: string };
        context = data.context || '';
      }
    } catch {
      // MCP unavailable — continue without memory
    }

    // 2) Build messages with injected context
    const systemPrompt = context
      ? `You are a helpful assistant with access to the user's Notion memory.

Relevant context from memory:
${context}

Use this context to inform your response when relevant.`
      : 'You are a helpful assistant.';

    // 3) Call OpenAI
    const completion = await openai.chat.completions.create({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message },
      ],
    });

    const reply = completion.choices[0]?.message?.content || '';
    res.json({ reply, context_used: !!context });
  } catch (err) {
    console.error('Gateway error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Direct vector search endpoint (bypasses OpenAI — returns raw matches)
// POST /search  { query: string, topK?: number }
app.post('/search', async (req: Request, res: Response) => {
  try {
    const { query: q, topK = 5 } = req.body as { query: string; topK?: number };
    if (!q) {
      res.status(400).json({ error: 'query is required' });
      return;
    }
    const embedding = await getEmbedding(q);
    const result = await query(
      `SELECT content, metadata,
              1 - (embedding <=> $1::vector) AS similarity
       FROM memory_chunks
       ORDER BY embedding <=> $1::vector
       LIMIT $2`,
      [JSON.stringify(embedding), topK]
    );
    res.json({ results: result.rows });
  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const PORT = parseInt(process.env.GATEWAY_PORT || process.env.PORT || '4000', 10);
app.listen(PORT, () => {
  console.log(`Gateway listening on port ${PORT}`);
});
