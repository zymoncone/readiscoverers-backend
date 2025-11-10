from google.genai.types import (
    GenerateContentConfig,
    FunctionDeclaration,
    Tool,
)
from typing import Union
import os


def call_model_with_structured_output(user_query: str, client) -> Union[dict, None]:
    """
    Calls a Gemini model to convert natural language into a structured semantic search query.

    Args:
        user_query: The natural language query from the user.
        location: The region where your Gemini model is available (e.g., "us-central1").
        user_query: The natural language query from the user.
    """

    # Define the desired output schema as a FunctionDeclaration
    # Gemini will try to call this "function" with the structured data
    semantic_search_query_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="generate_book_search_query",
                description="Generates a semantic search query to find relevant passages in a book.",
                parameters={
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The refined search query text to find relevant book passages. This should capture the core semantic meaning of what the user is looking for.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context about what type of information the user wants (e.g., 'character description', 'plot event', 'dialogue', 'setting description').",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Important keywords or phrases that should appear in the results (e.g., character names, locations, themes).",
                        },
                    },
                    "required": ["search_query"],
                },
            )
        ]
    )

    # Construct the prompt with clear instructions for the model
    # and reference the desired output format implicitly via the tool
    prompt = f"""
    You are an expert at converting natural language questions into optimized semantic search queries for finding relevant passages in books.
    Your goal is to extract the core search intent and transform it into an effective query for finding book passages.

    User's question: "{user_query}"

    Please provide the book search query using the 'generate_book_search_query' tool.
    - 'search_query': A refined version of the user's question optimized for semantic search
    - 'context': What type of information is being sought (optional)
    - 'keywords': Important names, places, or concepts that should be found (optional)

    Examples:
    - User asks: "When does Dorothy meet the Scarecrow?"
      → search_query: "Dorothy first encounters and meets the Scarecrow"
      → keywords: ["Dorothy", "Scarecrow"]

    - User asks: "What happens in the Emerald City?"
      → search_query: "events and scenes taking place in the Emerald City"
      → keywords: ["Emerald City"]
    """

    # Generate content with the model, including the tool definition
    try:
        MODEL_ID = "gemini-2.0-flash-001"

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0,  # Aim for deterministic output for query generation
                tools=[semantic_search_query_tool],
            ),
        )

        # Process the response to extract the structured output
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "generate_book_search_query"
                ):
                    # Extract the arguments passed to the function, which is our structured data
                    query_data = part.function_call.args
                    if os.environ.get("ENV") == "dev":
                        print("Generated Book Search Query (JSON):")
                        print(query_data)
                    return query_data

        print("No structured output found in the response.")
        print(f"Response: {response}")
        return None

    except Exception as e:
        print(f"An error occurred in call_model_with_structured_output: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- FOR TESTING ---
if __name__ == "__main__":
    import vertexai
    from google import genai

    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    LOCATION = str(os.environ.get("GOOGLE_CLOUD_LOCATION"))

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    user_question_1 = "When does Dorothy meet the Scarecrow for the first time?"
    user_question_2 = "What happens in the Emerald City?"
    user_question_3 = "Tell me about the Wicked Witch of the West"

    print(f"\nProcessing query: '{user_question_1}'")
    call_model_with_structured_output(user_question_1, client)

    print(f"\nProcessing query: '{user_question_2}'")
    call_model_with_structured_output(user_question_2, client)

    print(f"\nProcessing query: '{user_question_3}'")
    call_model_with_structured_output(user_question_3, client)
