import express, { Request, Response } from 'express';
import OpenAI from 'openai';
import { query } from './db';

const app = express();
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MCP_URL = process.env.MCP_URL || 'https://notion-memory-mcp.onrender.com';

async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
    input: text,
  });
  return res.data[0].embedding;
}

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
    let memoryContext = '';
    try {
      const mcpRes = await fetch(`${MCP_URL}/context`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message, topK: 5 }),
      });
      if (mcpRes.ok) {
        const data = await mcpRes.json() as { context?: string };
        memoryContext = data.context || '';
      }
    } catch (e) {
      console.error('Failed to fetch memory context:', e);
    }

    // 2) Build system prompt with injected memory
    const systemPrompt = memoryContext
      ? `You have access to the following relevant memory context from Notion:\n\n${memoryContext}\n\nUse this context to inform your response.`
      : 'You are a helpful assistant.';

    // 3) Call OpenAI
    const completion = await openai.chat.completions.create({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message },
      ],
    });

    const reply = completion.choices[0].message.content;
    res.json({ reply, memoryUsed: !!memoryContext });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const PORT = parseInt(process.env.PORT || '4000', 10);
app.listen(PORT, () => {
  console.log(`Gateway listening on port ${PORT}`);
});
