import os
import aiohttp
import asyncio
import uuid

import pandas as pd

from .config import BASE_URL
from .chunk_matching import precompute_text_locations_in_chunks


async def process_book(
    session,
    url,
    target_chunk_size=800,
    sentence_overlap=2,
    small_paragraph_length=200,
    small_paragraph_overlap=2,
):
    """Process a single book and return its filename"""
    book_data_payload = {
        "url": url,
        "target_chunk_size": target_chunk_size,
        "sentence_overlap": sentence_overlap,
        "small_paragraph_length": small_paragraph_length,
        "small_paragraph_overlap": small_paragraph_overlap,
    }

    async with session.post(
        f"{BASE_URL}/v1/book-data", json=book_data_payload
    ) as response:
        result = await response.json()
        if result.get("status") == "error":
            raise Exception(f"Error uploading {url}: {result['message']}")
        return result.get("filename")


async def process_all_books(book_urls, **chunking_params):
    """
    Process all books in parallel and wait for all to complete

    This mimics Promise.all in JavaScript (as done in the frontend)
    """
    async with aiohttp.ClientSession() as session:
        # Create tasks for all uploads
        tasks = [process_book(session, url, **chunking_params) for url in book_urls]

        filenames = []
        # Process tasks as they complete
        for i, coroutine in enumerate(asyncio.as_completed(tasks), 1):
            filename = await coroutine
            print(f"[{i}/{len(tasks)}] Completed chunking and embedding: {filename}")
            filenames.append(filename)

        print(f"Successfully processed {len(filenames)} books")
        return filenames


async def run_test_async(
    session,
    test_query=None,
    filenames=None,
    book_urls=None,
    target_chunk_size=None,
    sentence_overlap=None,
    small_paragraph_length=None,
    small_paragraph_overlap=None,
    skip_book_upload=True,
):

    if not test_query:
        raise ValueError("test_query must be provided")

    if not skip_book_upload:
        if any(
            param is None
            for param in [
                target_chunk_size,
                sentence_overlap,
                small_paragraph_length,
                small_paragraph_overlap,
                book_urls,
            ]
        ):
            raise ValueError(
                "Chunking parameters must be provided when skip_book_upload is False"
            )

        filenames = await process_all_books(
            book_urls=book_urls,
            target_chunk_size=target_chunk_size,
            sentence_overlap=sentence_overlap,
            small_paragraph_length=small_paragraph_length,
            small_paragraph_overlap=small_paragraph_overlap,
        )
    elif not filenames:
        raise ValueError("filenames must be provided if skip_book_upload is True")

    query_id = str(uuid.uuid4())
    model_payload = {"user_query": test_query}

    async with session.post(
        f"{BASE_URL}/v1/model-response", json=model_payload
    ) as response:
        result = await response.json()
        if result.get("status") == "error":
            raise Exception(f"Error in model response: {result['message']}")

    search_payload = {
        "query": result["search_query"],
        "filenames": filenames,
        "top_k": 3,
        "query_id": query_id,
        "enhanced_query": True,
        "keywords": result["keywords"],
    }

    async with session.post(
        f"{BASE_URL}/v1/search-response", json=search_payload
    ) as response:
        search_results = await response.json()
        if search_results.get("status") == "error":
            raise Exception(f"Error in search response: {search_results['message']}")
        return search_results, filenames


async def run_all_tests(
    selected_questions: pd.DataFrame,
    param_combos: list,
    book_urls: list,
    skip_book_processing: bool = False,
    results_filename: str = "readiscovers_app_results_top_10_param_combos_RUN_X",
):
    run_id = str(uuid.uuid4())[:8]
    import datetime

    all_results = []

    async with aiohttp.ClientSession() as session:
        for test_num, params in enumerate(param_combos, 1):
            (
                target_chunk_size,
                sentence_overlap,
                small_paragraph_length,
                small_paragraph_overlap,
            ) = params

            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"[{timestamp}] [{run_id}] TEST {test_num}: chunk_size={target_chunk_size} sentence_overlap={sentence_overlap} small_paragraph_length={small_paragraph_length} small_paragraph_overlap={small_paragraph_overlap}"
            )

            if not skip_book_processing:
                start_time = datetime.datetime.now()
                filenames = await process_all_books(
                    book_urls=book_urls,
                    target_chunk_size=target_chunk_size,
                    sentence_overlap=sentence_overlap,
                    small_paragraph_length=small_paragraph_length,
                    small_paragraph_overlap=small_paragraph_overlap,
                )
                end_time = datetime.datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                print(
                    f"[{timestamp}] [{run_id}] Completed book processing for TEST {test_num}, "
                    f"it took {elapsed:.2f} seconds"
                )
            else:
                print(
                    "Skipping book processing. Assumption is dfs are already processed correctly and available in the directory."
                )
                # fallback to hardcoded filenames
                filenames = [
                    "the_wonderful_wizard_of_oz",
                    "the_marvelous_land_of_oz",
                    "ozma_of_oz",
                    "dorothy_and_the_wizard_in_oz",
                    "the_road_to_oz",
                    "the_emerald_city_of_oz",
                    "the_patchwork_girl_of_oz",
                    "jack_pumpkinhead_of_oz",
                ]

            print(
                f"[{timestamp}] [{run_id}] Starting tests per query for TEST {test_num}"
            )

            # Precompute expected chunk locations for all questions
            print("Precomputing expected chunk locations for all questions...")
            selected_questions_with_expected = precompute_text_locations_in_chunks(
                selected_questions
            )
            print(f"Completed precomputing expected locations.\n")

            # Process chunk length data
            chunk_lengths = {}
            avg_chunk_length = {}
            for filename in filenames:
                temp_df = pd.read_pickle(f"../temp/{filename}.pkl")
                chunk_lengths.update({filename: temp_df.chunk_length.tolist()})
                avg_chunk_length.update({filename: temp_df.chunk_length.mean()})

            for question_number, (_, question_row) in enumerate(
                selected_questions_with_expected.iterrows(), 1
            ):

                question = question_row["Question"]
                print(
                    f"[{timestamp}] [{run_id}]   Question {question_number}: {question}"
                )

                search_results, _ = await run_test_async(
                    session, test_query=question, filenames=filenames
                )

                # Get precomputed expected values from the row
                expected_chunk_index = question_row["expected_chunk_index"]
                expected_chapter_number = question_row["expected_chapter_number"]
                expected_chapter_title = question_row["expected_chapter_title"]
                expected_chunk_in_chapter_index = question_row[
                    "expected_chunk_in_chapter_index"
                ]
                all_expected_chunks = question_row["all_expected_chunks"]
                expected_filename = question_row["expected_filename"]
                expected_book_filepath = f"../temp/{expected_filename}.pkl"

                for result_num, result in enumerate(
                    search_results["search_results"], 1
                ):
                    matched_text = result["data"]["matched_texts"]
                    matched_hits = [
                        m.get("text", "")
                        for m in matched_text
                        if m.get("is_match") is True
                    ]
                    match_text = " ".join(matched_hits) if matched_hits else ""

                    book_match = (
                        result["data"]["book_title"].lower().strip()
                        == question_row["Book Title"].lower().strip()
                    )
                    text_match = question_row["Best Answer"] in match_text

                    # chunk distances
                    chunk_distance_from_expected = (
                        abs(result["data"]["chunk_index"] - expected_chunk_index)
                        if expected_chunk_index is not None and book_match
                        else "NaN"
                    )
                    if (
                        isinstance(chunk_distance_from_expected, (int, float))
                        and not pd.isna(chunk_distance_from_expected)
                        and os.path.exists(expected_book_filepath)
                    ):
                        character_distance_from_expected = round(
                            chunk_distance_from_expected
                            * avg_chunk_length[expected_filename]
                        )
                    else:
                        character_distance_from_expected = "NaN"

                    # chapter distances
                    chapter_match = (
                        (result["data"]["chapter_number"] == expected_chapter_number)
                        if book_match
                        else "NaN"
                    )
                    chapter_distance_from_expected = (
                        abs(result["data"]["chapter_number"] - expected_chapter_number)
                        if expected_chapter_number is not None and book_match
                        else "NaN"
                    )
                    chunk_in_chap_distance_from_expected = (
                        abs(
                            result["data"]["chunk_in_chapter_index"]
                            - expected_chunk_in_chapter_index
                        )
                        if expected_chunk_in_chapter_index is not None
                        and chapter_match is True
                        else "NaN"
                    )

                    result_row = {
                        "test_number": test_num,
                        "question_number": question_number,
                        "result_rank": result_num,
                        "original_query": question,
                        "enhanced_query": result["data"]["query"],
                        "target_chunk_size": target_chunk_size,
                        "sentence_overlap": sentence_overlap,
                        "small_paragraph_length": small_paragraph_length,
                        "small_paragraph_overlap": small_paragraph_overlap,
                        "matched_chapter_title": result["data"]["chapter_title"],
                        "expected_chapter_title": expected_chapter_title,
                        "matched_chapter_number": result["data"]["chapter_number"],
                        "expected_chapter_number": expected_chapter_number,
                        "correct_chapter_found": chapter_match,
                        "chapter_distance_from_expected": chapter_distance_from_expected,
                        "matched_chunk_in_chapter_index": result["data"][
                            "chunk_in_chapter_index"
                        ],
                        "exptected_chunk_in_chapter_index": expected_chunk_in_chapter_index,
                        "chunk_in_chap_distance_from_expected": chunk_in_chap_distance_from_expected,
                        "score": result["data"]["score"],
                        "matched_text": match_text,
                        "expected_text": question_row["Best Answer"],
                        "correct_text_found": text_match,
                        "matched_chunk_index": result["data"]["chunk_index"],
                        "expected_primary_chunk_index": (
                            expected_chunk_index if expected_chunk_index else "NaN"
                        ),
                        "expected_all_chunk_indices": (
                            all_expected_chunks if all_expected_chunks else "NaN"
                        ),
                        "chunk_distance_from_expected": chunk_distance_from_expected,
                        "char_distance_from_expected": character_distance_from_expected,
                        "matched_book_title": result["data"]["book_title"],
                        "expected_book_title": question_row["Book Title"],
                        "expected_book_filename": expected_filename,
                        "correct_book_found": book_match,
                        "avg_chunk_length": avg_chunk_length,
                        "all_chunks": chunk_lengths,
                    }
                    all_results.append(result_row)
                    pd.DataFrame(all_results).to_csv(
                        f"{results_filename}.csv",
                        index=False,
                    )

    results_df = pd.DataFrame(all_results)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n{'='*80}")
    print(f"[{timestamp}] [{run_id}] COMPLETED {len(param_combos)} TESTS")
    print(f"[{timestamp}] [{run_id}] Total results collected: {len(results_df)}")
    print(f"{'='*80}")
    return results_df
