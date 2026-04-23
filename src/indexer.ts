import { Client } from '@notionhq/client';
import OpenAI from 'openai';
import { query } from './db';

const notion = new Client({ auth: process.env.NOTION_TOKEN });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const CHUNK_SIZE = 500; // characters

function chunkText(text: string, size = CHUNK_SIZE): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return res.data[0].embedding;
}

function extractPlainText(blocks: any[]): string {
  return blocks
    .map((block: any) => {
      const type = block.type;
      const content = block[type];
      if (!content) return '';
      if (content.rich_text) {
        return content.rich_text.map((t: any) => t.plain_text).join('');
      }
      return '';
    })
    .join('\n')
    .trim();
}

async function getPageBlocks(pageId: string): Promise<any[]> {
  const blocks: any[] = [];
  let cursor: string | undefined;
  do {
    const res = await notion.blocks.children.list({
      block_id: pageId,
      start_cursor: cursor,
    });
    blocks.push(...res.results);
    cursor = res.has_more ? (res.next_cursor ?? undefined) : undefined;
  } while (cursor);
  return blocks;
}

async function ensureTable() {
  await query(`
    CREATE TABLE IF NOT EXISTS memory_chunks (
      id TEXT PRIMARY KEY,
      page_id TEXT NOT NULL,
      page_title TEXT,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      embedding vector(1536),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );
  `);
  await query(`
    CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx
    ON memory_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
  `);
}

async function indexDatabase(databaseId: string) {
  console.log(`Indexing database: ${databaseId}`);
  let cursor: string | undefined;
  do {
    const res = await notion.databases.query({
      database_id: databaseId,
      start_cursor: cursor,
    });
    for (const page of res.results) {
      await indexPage(page as any);
    }
    cursor = res.has_more ? (res.next_cursor ?? undefined) : undefined;
  } while (cursor);
}

async function indexPage(page: any) {
  const pageId = page.id;
  const titleProp = Object.values(page.properties || {}).find(
    (p: any) => p.type === 'title'
  ) as any;
  const title = titleProp?.title?.map((t: any) => t.plain_text).join('') || 'Untitled';

  console.log(`  Indexing page: ${title}`);
  const blocks = await getPageBlocks(pageId);
  const text = extractPlainText(blocks);
  if (!text) return;

  const chunks = chunkText(text);
  for (let i = 0; i < chunks.length; i++) {
    const chunkId = `${pageId}_${i}`;
    const embedding = await getEmbedding(chunks[i]);
    await query(
      `INSERT INTO memory_chunks (id, page_id, page_title, chunk_index, content, embedding, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6::vector, NOW())
       ON CONFLICT (id) DO UPDATE
       SET content = EXCLUDED.content,
           embedding = EXCLUDED.embedding,
           updated_at = NOW()`,
      [chunkId, pageId, title, i, chunks[i], JSON.stringify(embedding)]
    );
    console.log(`    Upserted chunk ${i + 1}/${chunks.length}`);
  }
}

async function main() {
  await ensureTable();
  const databaseId = process.env.NOTION_DATABASE_ID;
  if (!databaseId) throw new Error('NOTION_DATABASE_ID env var is required');
  await indexDatabase(databaseId);
  console.log('Indexing complete.');
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
