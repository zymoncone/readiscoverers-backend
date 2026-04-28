# Read it long ago? <span style="color: #6366f1;">Readiscover</span> it in seconds.

[![App Status](https://img.shields.io/badge/Live-readiscover.app-6366f1)](https://readiscover.app)
[![Pylint](https://github.com/zymoncone/readiscoverers-backend/actions/workflows/pylint.yml/badge.svg)](https://github.com/zymoncone/readiscoverers-backend/actions/workflows/pylint.yml)

![App Demo](./media/app_demo_v1.gif)

## 🌐 Visit the Live Website

Check out the live app at **[readiscover.app](https://readiscover.app)**!

> **Note:** We're currently in active development. To access the app, please email **szymons@umich.edu** for the password.

## 📖 What Does Readiscover Do?

Once a user uploads their book links, we parse and chunk the texts using chapter and paragraph structure along with our tuned splitting and overlap parameters. These chunks are embedded into vector representations and stored permanently in Google Cloud Storage.

When a user submits a query in natural language, a large language model reformulates it to extract context and keywords, producing a structured, search-optimized query. This enhanced query is also embedded, and we perform cosine similarity search against cached book embeddings (loaded from Redis cache or GCS as needed). The top three passages with the highest similarity scores are returned as results.

### Caching Architecture
- **First Request:** Load embeddings from GCS → Cache in Redis for 24 hours
- **Subsequent Requests:** Load directly from Redis cache (much faster)
- **Cache Miss:** Automatic refresh from GCS when cache expires

## 🛠️ Tech Stack

- **LLM:** Gemini Flash 2.0 (`gemini-2.0-flash-001`)
- **Embeddings:** Google Text Embedding (`text-embedding-005`)
- **API Framework:** FastAPI
- **HTML Parsing:** BeautifulSoup4 (bs4)
- **Deployment:** Docker + Google Cloud Run
- **Storage:** Google Cloud Storage (GCS) for persistent book data
- **Caching:** Redis Cloud for fast access to embeddings and metadata
- **Vector Storage:** Pandas DataFrames with cosine similarity search

## 🚀 Performance Optimizations

### Multi-Layer Caching Strategy
- **GCS (Persistent Storage):** Processed book embeddings and metadata stored permanently in Google Cloud Storage
- **Redis (Fast Cache):** In-memory cache layer using Redis Cloud to avoid repeated GCS downloads
- **Cache Keys:** `gcs_pickle:{bucket}:{embeddings_path}` and `gcs_json:{bucket}:{metadata_path}`
- **TTL:** 24 hours cache expiration with automatic refresh

### Benefits
- ⚡ **Fast Queries:** Sub-second response times after initial cache load
- 💰 **Cost Effective:** Redis Cloud Free tier ($0/month) for small datasets
- 🔄 **Scalable:** Easy to upgrade Redis tier as usage grows
- 🛡️ **Reliable:** Automatic failover and persistence in Redis Cloud

## 🔍 Architectural Diagram for Semantic Retriever

```mermaid
flowchart TD
    B[Compute embeddings<br/>and split chunks]
    C[User document upload] --> D{Already have<br/>embeddings for<br/>this book?}
    D -->|No| B
    D -->|Yes| E[(GCS Bucket<br/>embeddings & metadata)]
    B --> E

    F[User query<br/><i>'When does he realize<br/>the Wizard isn't real?'</i>] --> G[Query reformulation<br/>via LLM]
    G --> H[semantic_score<br/>reformed_query]

    E --> I{Redis Cache<br/>Hit?}
    I -->|Yes| J[Load from Redis<br/>Cache]
    I -->|No| K[Load from GCS<br/>+ Cache in Redis]

    J --> L[Retrieve raw<br/>document chunk]
    K --> L
    H --> L
    L --> M[Display response<br/>to user]

    G -.->|Example output| N[<i>'Dorothy discovers the<br/>Wizard is an ordinary man<br/>pretending to have powers.'</i>]

    style B fill:#fff9e6,stroke:#fbbf24,color:#000
    style D fill:#f3e5f5,stroke:#9c27b0,color:#000
    style G fill:#fee,stroke:#f43f5e,color:#000
    style I fill:#e0f2fe,stroke:#0284c7,color:#000
    style E fill:#f0f0f0,stroke:#666,color:#000
    style J fill:#f0f0f0,stroke:#666,stroke-dasharray: 5 5,color:#000

    note1[💡 This is a pre-trained NN that has vectorized most common words<br/>in the english language. Tuning this would be unnecessary unless<br/>we deem there is vernacular which is not commonly found in most<br/>english text i.e. med- or law terms.]
    B -.-> note1
    style note1 fill:#fff9e6,stroke:#fbbf24,color:#000
```

## 📡 API Documentation

The backend exposes the following REST API endpoints:

### `POST /v1/book-data`
Process and index a book from a URL (Project Gutenberg supported).

**Request Body:**
```json
{
  "url": "https://www.gutenberg.org/cache/epub/55/pg55-images.html",
  "target_chunk_size": 1200,
  "sentence_overlap": 3,
  "small_paragraph_length": 300,
  "small_paragraph_overlap": 3
}
```

**Response:**
```json
{
  "status": "success",
  "filename": "uuid-generated-filename",
  "book_title": "The Wonderful Wizard of Oz",
  "book_author": "L. Frank Baum",
  "message": "Book data processed and saved."
}
```

### `POST /v1/model-response`
Convert a natural language query into an optimized search query using LLM.

**Request Body:**
```json
{
  "user_query": "When does he realize the Wizard isn't real?"
}
```

**Response:**
```json
{
  "status": "success",
  "search_query": "Dorothy discovers the Wizard is an ordinary man pretending to have powers.",
  "keywords": ["Wizard", "discover", "ordinary man", "pretend", "powers"]
}
```

### `POST /v1/search-response`
Search for relevant passages across indexed books using semantic similarity.

**Request Body:**
```json
{
  "query": "Dorothy discovers the Wizard is an ordinary man",
  "filenames": ["uuid-generated-filename"],
  "top_k": 3,
  "query_id": "unique-query-id",
  "enhanced_query": true,
  "keywords": ["Wizard", "discover", "ordinary man"]
}
```

**Response:**
```json
{
  "status": "success",
  "search_results": [
    {
      "text": "Passage text...",
      "similarity_score": 0.67,
      "book_title": "The Wonderful Wizard of Oz",
      "book_author": "L. Frank Baum",
      "chapter": 9,
      "chunk_index": 101
    }
  ]
}
```

## 🚀 Running Locally

### Prerequisites

1. **Install Docker**
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. **Install Google Cloud CLI**
   - Download and install the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
   - After installation, initialize gcloud:
     ```bash
     gcloud init
     ```
   - Authenticate with your Google account:
     ```bash
     gcloud auth login
     gcloud auth application-default login
     ```

### Required Accounts & Services
- **Google Cloud Platform (GCP) Account** - For Cloud Storage and Cloud Run deployment
- **Redis Cloud Account & Database** - For caching book embeddings and metadata
  - Sign up at [Redis Cloud](https://redis.com/try-free/)
  - Create a free database (30MB limit, perfect for development)
  - Get your connection URL (format: `redis://username:password@host:port`)

### Build locally with docker

Build a new docker image and spin up container in local environment.

```bash
docker compose up --build
```