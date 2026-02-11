# Excel RAG with Pinecone

Natural language query system for Excel/CSV data using Pinecone vector database.

## Features

- Row-based chunking for structured data
- Pinecone serverless deployment
- Advanced metadata filtering
- OpenAI embeddings

## Technical Details

- Chunking Strategy: Row-based (1 row = 1 chunk)
- Vector DB: Pinecone (AWS us-east-1, serverless)
- Embeddings: OpenAI text-embedding-ada-002
- Metadata: customer_id, name, city, age, product, purchase_amount, satisfaction

## Results

Query 1: "Who bought laptops in Istanbul?"
- Found customers with Laptop purchases
- Semantic search working correctly

Query 2: "high satisfaction customers" + Filter: City=Istanbul AND Satisfaction>=4
- Combined semantic search with metadata filtering
- Only returned Istanbul customers with satisfaction >= 4
- Precise multi-condition filtering

## Data

20 customer records with fields:
- customer_id, name, city, age, product, purchase_amount, satisfaction
