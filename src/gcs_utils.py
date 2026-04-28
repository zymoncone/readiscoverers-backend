"""Utilities for interacting with Google Cloud Storage."""

import json
import os
import pickle
from typing import Union, Any

import redis
from google.cloud import storage

from .constants import (
    USE_REDIS,
    REDIS_URL,
    REDIS_TTL_SECONDS,
)


def get_gcs_client():
    """Initialize and return a GCS storage client."""
    return storage.Client()


def read_pickle_from_gcs(bucket_name: str, blob_name: str) -> Union[Any, None]:
    """Read a pickle file from GCS and deserialize it.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the blob in the bucket (e.g., 'embeddings/book.pkl')

    Returns:
        Deserialized pickle object, or None if error
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return None

        data_bytes = blob.download_as_bytes()
        return pickle.loads(data_bytes)
    except Exception as e:
        print(f"Error reading pickle from GCS {bucket_name}/{blob_name}: {str(e)}")
        return None


def write_pickle_to_gcs(
    bucket_name: str, blob_name: str, data: Any
) -> Union[bool, None]:
    """Serialize and write a pickle file to GCS.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the blob in the bucket (e.g., 'embeddings/book.pkl')
        data: Object to serialize and upload

    Returns:
        True if successful, False/None if error
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        data_bytes = pickle.dumps(data)
        blob.upload_from_string(data_bytes)
        return True
    except Exception as e:
        print(f"Error writing pickle to GCS {bucket_name}/{blob_name}: {str(e)}")
        return False


def read_json_from_gcs(bucket_name: str, blob_name: str) -> Union[dict, None]:
    """Read a JSON file from GCS and parse it.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the blob in the bucket (e.g., 'metadata/book.json')

    Returns:
        Parsed JSON dict, or None if error
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return None

        data_string = blob.download_as_text()
        return json.loads(data_string)
    except Exception as e:
        print(f"Error reading JSON from GCS {bucket_name}/{blob_name}: {str(e)}")
        return None


def write_json_to_gcs(
    bucket_name: str, blob_name: str, data: dict
) -> Union[bool, None]:
    """Serialize and write a JSON file to GCS.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the blob in the bucket (e.g., 'metadata/book.json')
        data: Dict to serialize and upload

    Returns:
        True if successful, False/None if error
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        data_string = json.dumps(data)
        blob.upload_from_string(data_string)
        return True
    except Exception as e:
        print(f"Error writing JSON to GCS {bucket_name}/{blob_name}: {str(e)}")
        return False


def blob_exists_in_gcs(bucket_name: str, blob_name: str) -> bool:
    """Check if a blob exists in GCS.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Path to the blob in the bucket

    Returns:
        True if blob exists, False otherwise
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        print(
            f"Error checking blob existence in GCS {bucket_name}/{blob_name}: {str(e)}"
        )
        return False


# Global Redis connection pool for better performance
_redis_pool = None


def get_redis_client():
    """Initialize and return a Redis client with connection pooling."""
    global _redis_pool

    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=10,
            decode_responses=False,  # Keep as bytes for pickle data
            socket_timeout=5,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )

    return redis.Redis(connection_pool=_redis_pool)


def get_cached_data(key: str) -> Union[Any, None]:
    """Get data from Redis cache.

    Args:
        key: Cache key

    Returns:
        Deserialized data or None if not found
    """
    if not USE_REDIS:
        return None

    try:
        client = get_redis_client()
        data_bytes = client.get(key)
        if data_bytes is None:
            return None
        return pickle.loads(data_bytes)
    except Exception as e:
        print(f"Error reading from Redis cache for key {key}: {str(e)}")
        return None


def set_cached_data(key: str, data: Any) -> bool:
    """Store data in Redis cache.

    Args:
        key: Cache key
        data: Data to cache

    Returns:
        True if successful, False otherwise
    """
    if not USE_REDIS:
        return False

    try:
        client = get_redis_client()
        data_bytes = pickle.dumps(data)
        return client.setex(key, REDIS_TTL_SECONDS, data_bytes)
    except Exception as e:
        print(f"Error writing to Redis cache for key {key}: {str(e)}")
        return False


def read_pickle_with_cache(bucket_name: str, blob_name: str) -> Union[Any, None]:
    """Read pickle from GCS with Redis caching.

    Args:
        bucket_name: GCS bucket name
        blob_name: GCS blob path

    Returns:
        Deserialized pickle data or None
    """
    cache_key = f"gcs_pickle:{bucket_name}:{blob_name}"

    # Try Redis cache first
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        if os.environ.get("ENV") == "dev":
            print(f"Cache hit for pickle: {blob_name}")
        return cached_data

    # Cache miss - fetch from GCS
    if os.environ.get("ENV") == "dev":
        print(f"Cache miss for pickle: {blob_name}, fetching from GCS")
    data = read_pickle_from_gcs(bucket_name, blob_name)

    # Cache the result if successful
    if data is not None:
        set_cached_data(cache_key, data)

    return data


def read_json_with_cache(bucket_name: str, blob_name: str) -> Union[dict, None]:
    """Read JSON from GCS with Redis caching.

    Args:
        bucket_name: GCS bucket name
        blob_name: GCS blob path

    Returns:
        Parsed JSON dict or None
    """
    cache_key = f"gcs_json:{bucket_name}:{blob_name}"

    # Try Redis cache first
    cached_data = get_cached_data(cache_key)
    if cached_data is not None:
        if os.environ.get("ENV") == "dev":
            print(f"Cache hit for JSON: {blob_name}")
        return cached_data

    # Cache miss - fetch from GCS
    if os.environ.get("ENV") == "dev":
        print(f"Cache miss for JSON: {blob_name}, fetching from GCS")
    data = read_json_from_gcs(bucket_name, blob_name)

    # Cache the result if successful
    if data is not None:
        set_cached_data(cache_key, data)

    return data
