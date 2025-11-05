import os
import vertexai
from google import genai
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
from vertexai import rag

PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
LOCATION = str(os.environ.get("GOOGLE_CLOUD_LOCATION"))
EXISTING_CORPUS_ID = str(os.environ.get("CORPUS_ID", None))
paths = [
    "https://drive.google.com/drive/folders/1G-_s_17G-d4BQcPAGjV1TL9nhXDP4G-g"
]  # Supports Google Cloud Storage and Google Drive Links

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# Create a RAG Corpus, Import Files, and Generate a response
def get_rag_model_response(
    user_query: str = None, new_rag_corpus_name: str = None
) -> str:
    if user_query is None:
        return "no user query provided"

    if new_rag_corpus_name is None:
        if EXISTING_CORPUS_ID is None:
            raise ValueError(
                "No corpus ID provided. Please set the CORPUS_ID environment variable."
            )
        try:
            corpus_resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{EXISTING_CORPUS_ID}"

            # Get the existing corpus using its full resource name
            rag_corpus = rag.get_corpus(name=corpus_resource_name)
            print(
                f"Successfully referenced existing RAG corpus by ID: {rag_corpus.display_name}"
            )

        except Exception as e:
            print(f"Could not find or retrieve corpus '{display_name}': {e}")
            return f"Error: Could not find or retrieve corpus '{display_name}': {e}"
    else:
        print("Creating a new RAG corpus...")

        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )

        rag_corpus = rag.create_corpus(
            display_name=new_rag_corpus_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )

        # Import Files to the RagCorpus ONLY when creating a new corpus
        print(f"Importing files to new corpus: {rag_corpus.name}")
        rag.import_files(
            rag_corpus.name,
            paths,
            # Optional
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=512,
                    chunk_overlap=100,
                ),
            ),
            max_embedding_requests_per_min=1000,  # Optional
        )

    # Create a RAG retrieval tool
    rag_retrieval_tool = Tool(
        retrieval=Retrieval(
            vertex_rag_store=VertexRagStore(
                rag_corpora=[rag_corpus.name],
                similarity_top_k=10,
                vector_distance_threshold=0.5,
            )
        )
    )

    MODEL_ID = "gemini-2.0-flash-001"

    try:
        # Create a Gemini model instance
        print(f"Calling Gemini API with model: {MODEL_ID}")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_query,
            config=GenerateContentConfig(tools=[rag_retrieval_tool]),
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        print(f"Error calling Gemini API: {error_msg}")

        # Check for specific error types
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            return "Error: API quota exceeded. You've hit the rate limit for Vertex AI/Gemini API. Please wait a few minutes and try again, or check your quota at https://console.cloud.google.com/iam-admin/quotas"
        elif "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            return "Error: Permission denied. Please check that your service account has the necessary permissions for Vertex AI."
        else:
            return f"Error calling AI model: {error_msg}"
