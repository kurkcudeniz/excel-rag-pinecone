import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import os
import time

print("📊 Excel RAG with Pinecone Starting...")

# 1. CSV dosyasını oku
print("📂 Loading customer data...")
df = pd.read_csv('customers.csv')
print(f"✅ Loaded {len(df)} customer records")
print(df.head())

# 2. API Keys
openai_key = input("\nEnter OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = openai_key

pinecone_key = input("Enter Pinecone API Key: ")

# 3. Pinecone initialize
print("\n🔧 Initializing Pinecone...")
pc = Pinecone(api_key=pinecone_key)

# 4. Index oluştur
index_name = "customer-rag"

# Varsa sil, yoksa oluştur
if index_name in pc.list_indexes().names():
    print(f"🗑️  Deleting existing index: {index_name}")
    pc.delete_index(index_name)
    time.sleep(5)

print(f"🆕 Creating new index: {index_name}")
pc.create_index(
    name=index_name,
    dimension=1536,  # OpenAI embedding boyutu
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Index'e bağlan
index = pc.Index(index_name)
print("✅ Pinecone index ready!")

# 5. Embeddings oluştur
print("\n🔢 Creating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 6. Her satırı embed edip Pinecone'a kaydet
vectors_to_upsert = []

for idx, row in df.iterrows():
    # Row'u text'e çevir
    text = f"Customer: {row['name']}, City: {row['city']}, Age: {row['age']}, Product: {row['product']}, Amount: {row['purchase_amount']} TL, Satisfaction: {row['satisfaction']}/5"
    
    # Embed
    embedding = embeddings.embed_query(text)
    
    # Metadata
    metadata = {
        'customer_id': int(row['customer_id']),
        'name': row['name'],
        'city': row['city'],
        'age': int(row['age']),
        'product': row['product'],
        'purchase_amount': int(row['purchase_amount']),
        'satisfaction': int(row['satisfaction'])
    }
    
    # Vector hazırla
    vectors_to_upsert.append({
        'id': f"customer_{row['customer_id']}",
        'values': embedding,
        'metadata': metadata
    })
    
    print(f"✓ Processed: {row['name']}")

# 7. Pinecone'a batch insert
print(f"\n💾 Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
index.upsert(vectors=vectors_to_upsert)

print("✅ All data uploaded!")

# 8. Test query
print("\n" + "="*60)
print("🔍 TEST QUERY 1: Semantic Search")
print("="*60)

query1 = "Who bought laptops in Istanbul?"
print(f"Query: {query1}\n")

# Query'yi embed et
query_embedding = embeddings.embed_query(query1)

# Pinecone'da ara
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

for i, match in enumerate(results['matches'], 1):
    meta = match['metadata']
    print(f"\n📄 Result {i} (Score: {match['score']:.4f}):")
    print(f"Name: {meta['name']}")
    print(f"City: {meta['city']}")
    print(f"Product: {meta['product']}")
    print(f"Amount: {meta['purchase_amount']} TL")

# 9. Metadata filtering test
print("\n" + "="*60)
print("🔍 TEST QUERY 2: Metadata Filtering")
print("="*60)

query2 = "high satisfaction customers"
print(f"Query: {query2}")
print(f"Filter: City=Istanbul AND Satisfaction>=4\n")

query_embedding2 = embeddings.embed_query(query2)

# Metadata filter ile ara
results_filtered = index.query(
    vector=query_embedding2,
    top_k=5,
    include_metadata=True,
    filter={
        "city": {"$eq": "İstanbul"},
        "satisfaction": {"$gte": 4}
    }
)

for i, match in enumerate(results_filtered['matches'], 1):
    meta = match['metadata']
    print(f"\n📄 Result {i}:")
    print(f"Name: {meta['name']}")
    print(f"City: {meta['city']}")
    print(f"Product: {meta['product']}")
    print(f"Satisfaction: {meta['satisfaction']}/5")
    print(f"Amount: {meta['purchase_amount']} TL")

print("\n✅ Excel RAG Demo Complete!")
print(f"📊 Index Name: {index_name}")
print("🗑️  Don't forget to delete the index from Pinecone dashboard when done!")

