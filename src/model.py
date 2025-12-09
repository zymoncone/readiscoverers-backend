"""Module for converting natural language queries into structured book search queries."""

import os
from typing import Union

from google.genai.types import (
    GenerateContentConfig,
    FunctionDeclaration,
    Tool,
)


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
                description=(
                    "Reformulates a natural-language question into a concise, "
                    "semantic-search-optimized query for locating relevant book passages. "
                    "The model must NOT answer the question—only rewrite it."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": (
                                "A concise refined search query capturing the user's "
                                "true intent. Must not include an answer or invented facts. "
                                "Should explicitly name characters/items instead of pronouns."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional classification of what type of information is being sought "
                                "(e.g., 'character description', 'plot event', 'dialogue', "
                                "'setting', 'instructions/how-to')."
                            ),
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional important keywords: character names, locations, items, or concepts "
                                "extracted directly from the user's query without adding new facts."
                            ),
                        },
                    },
                    "required": ["search_query"],
                },
            )
        ]
    )  # Construct the prompt with clear instructions for the model
    # and reference the desired output format implicitly via the tool
    prompt = f"""
    You are an assistant that converts user questions into optimized semantic search queries
    for retrieving passages from books. Do NOT answer the question. Do NOT invent new facts.

    Your responsibilities:
    1. Extract the core intent of the user's question.
    2. Rewrite it into a concise search query (typically 4-12 words).
    3. Make entities explicit (expand pronouns like "she" -> "Dorothy" only if mentioned).
    4. Do not guess or add information beyond what the user provided.
    5. Optionally include:
    - "context": the general type of information requested
                    (e.g., "character description", "plot event", "instructions").
    - "keywords": explicit names, places, items, or major nouns from the query.

    User's question: "{user_query}"

    Produce your response by calling the 'generate_book_search_query' tool.

    Examples of correct behavior:

    Example 1
    User asks: "When does Dorothy meet the Scarecrow?"
    - search_query: "Dorothy's first meeting with the Scarecrow"
    - context: "plot event"
    - keywords: ["Dorothy", "Scarecrow"]

    Example 2
    User asks: "Why is Ojo arrested?"
    - search_query: "reasons for Ojo being arrested"
    - context: "plot event"
    - keywords: ["Ojo"]

    Example 3
    User asks: "How is Glinda described?"
    - search_query: "description of Glinda's appearance"
    - context: "character description"
    - keywords: ["Glinda"]

    Example 4
    User asks: "How does one use the Powder of Life?"
    - search_query: "instructions for using the Powder of Life"
    - context: "instructions"
    - keywords: ["Powder of Life"]

    Now reformulate the user's question accordingly.
    """

    # Generate content with the model, including the tool definition
    try:
        model_id = "gemini-2.0-flash-001"

        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0,  # Aim for deterministic output
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

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"An error occurred in call_model_with_structured_output: {e}")
        import traceback  # pylint: disable=import-outside-toplevel

        traceback.print_exc()
        return None
