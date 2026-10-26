"""CosmosDB based cacheing for test cost estimates.

Cache key is (costing_method, test_name) tuple.
"""

import dataclasses
import functools
import traceback
from typing import TYPE_CHECKING, Any

from config import CostingConfig

if TYPE_CHECKING:
    from costing.cost_types import CostEstimate


@functools.cache
def get_cache(config: CostingConfig) -> "CosmosDBCache":
    """Get or create a cached CosmosDBCache instance for the given config."""
    return CosmosDBCache(config)


def get_cached_cost(config: CostingConfig, test_name: str) -> "CostEstimate | None":
    """Look up a cached cost estimate for the given test name, returning None on error."""
    try:
        return get_cache(config).get(config, test_name)
    except Exception as e:
        print(f"Error accessing cache: {e}")
        traceback.print_exc()
        return None


def set_cached_cost(config: CostingConfig, test_name: str, cost_estimate: "CostEstimate") -> None:
    """Persist a cost estimate to the cache, silently ignoring errors."""
    try:
        get_cache(config).set(config, test_name, cost_estimate)
    except Exception as e:
        print(f"Error setting cache: {e}")
        traceback.print_exc()


class CosmosDBCache:
    """CosmosDB-backed cache for test cost estimates."""

    def __init__(self, config: CostingConfig):
        """Initialize the cache, connecting to CosmosDB if enabled in config."""
        self._debug = config.debug
        self._container: Any | None = None

        if not config.costing_cache_cosmosdb_enabled:
            return

        if not (
            config.costing_cache_cosmosdb_endpoint
            and config.costing_cache_cosmosdb_database
            and config.costing_cache_cosmosdb_container
        ):
            raise ValueError(
                "CosmosDB cache enabled but configuration incomplete. Required: endpoint, database, container"
            )

        # Imported lazily so the Azure Cosmos SDK is only required when the cache is enabled.
        try:
            from azure.cosmos import CosmosClient
            from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
        except ImportError as e:
            raise ImportError(
                "CosmosDB costing cache is enabled (costing_cache_cosmosdb_enabled=True) but the "
                "Azure Cosmos SDK is not installed. Install the optional dependency with "
                "`uv sync --extra azure` (or `pip install 'sdbench[azure]'`), or disable the cache."
            ) from e

        try:
            print(
                f"Initializing CosmosDB cache: {config.costing_cache_cosmosdb_endpoint}/{config.costing_cache_cosmosdb_database}/{config.costing_cache_cosmosdb_container}"
            )
            credential = (
                ManagedIdentityCredential()
                if config.costing_cache_use_managed_identity
                else DefaultAzureCredential()
            )
            client = CosmosClient(config.costing_cache_cosmosdb_endpoint, credential)
            database = client.get_database_client(config.costing_cache_cosmosdb_database)
            self._container = database.get_container_client(config.costing_cache_cosmosdb_container)
            print("CosmosDB cache initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize CosmosDB costing cache: {e}") from e

    @property
    def enabled(self) -> bool:
        """Check if the cache is enabled and ready to use."""
        return self._container is not None

    @staticmethod
    def normalize_key(config: CostingConfig, key: str) -> str:
        """Normalize a cache key for consistent lookups."""
        normalized = key.lower().strip()
        return f"{config.costing_method}:{config.costing_model}:{normalized}"

    @staticmethod
    def make_cosmosdb_safe_id(normalized_key: str) -> str:
        r"""Convert a normalized key to a CosmosDB-safe ID for storage.

        CosmosDB NoSQL API id field restrictions:
        - Cannot contain: / \ ? #
        - Max length: ~1KB (1024 bytes when UTF-8 encoded)
        - No leading/trailing whitespace

        Args:
            normalized_key: The already normalized cache key

        Returns:
            A CosmosDB-safe ID that can be used as the document id
        """
        # Replace special characters that aren't allowed in CosmosDB keys
        # Use safe replacements that maintain readability
        safe_id = normalized_key.replace("/", "_")
        safe_id = safe_id.replace("\\", "_")
        safe_id = safe_id.replace("?", "_")
        safe_id = safe_id.replace("#", "_")

        # Ensure key doesn't exceed 1024 bytes when UTF-8 encoded
        # Use conservative limit of 900 characters to account for UTF-8 encoding
        if len(safe_id) > 900:
            safe_id = safe_id[:900]

        return safe_id

    def get(self, config: CostingConfig, key: str) -> "CostEstimate | None":
        """Look up a cached value using a point read.

        Args:
            config: The costing configuration used
            key: The cache key (will be normalized automatically)

        Returns:
            Cached data if found, None otherwise
        """
        if not self.enabled:
            if self._debug:
                print(f"Cache skipped for test: {key}")
            return None

        normalized_key = self.normalize_key(config, key)
        safe_id = self.make_cosmosdb_safe_id(normalized_key)

        try:
            # Use point read for efficient lookup (requires id and partition key)
            # Misses are indicated by exceptions
            cached_data = self._container.read_item(item=safe_id, partition_key=safe_id)  # pyright: ignore[reportOptionalMemberAccess]

            # Validate the stored cache_key matches our normalized key
            stored_cache_key = cached_data.get("cache_key")
            if stored_cache_key != normalized_key:
                # We must have had a key collision due to CosmosDB ID restrictions
                # If this is an issue in practice, we can consider maintaining a list of cached
                # items for a collided CosmosDB safe key
                if self._debug:
                    print(
                        f"⚠️ Cache key mismatch: expected '{normalized_key}', got '{stored_cache_key}'"
                    )
                return None

            # Return the cached result, removing CosmosDB-specific fields
            cached_data = dict(cached_data)
            # Remove config field that was added during caching
            cached_data.pop("config", None)
            # Remove CosmosDB metadata fields
            cached_data.pop("id", None)
            cached_data.pop("_rid", None)
            cached_data.pop("_self", None)
            cached_data.pop("_etag", None)
            cached_data.pop("_attachments", None)
            cached_data.pop("_ts", None)
            cached_data.pop("cache_key", None)

            if self._debug:
                print(f"✓ Cache hit for: '{normalized_key}'")
            return cached_data  # type: ignore[return-value]

        except Exception as e:
            # Point read throws exception if item not found, which is normal
            if "NotFound" in str(type(e).__name__) or "ResourceNotFound" in str(type(e).__name__):
                if self._debug:
                    print(f"Cache miss for test: {normalized_key}")
                return None
            print(f"Warning: Failed to lookup cache for '{normalized_key}': {e}")
            return None

    def set(self, config: CostingConfig, key: str, value: "CostEstimate") -> None:
        """Save a value to the cache.

        Args:
            config: The costing configuration used
            key: The cache key (will be normalized automatically)
            value: The data to cache
        """
        if not self.enabled:
            return

        normalized_key = self.normalize_key(config, key)
        safe_id = self.make_cosmosdb_safe_id(normalized_key)

        try:
            # Prepare data for CosmosDB
            cache_data = dict(value)
            cache_data["id"] = safe_id  # CosmosDB-safe ID for storage
            cache_data["cache_key"] = normalized_key  # Original normalized key for matching
            cache_data["config"] = dataclasses.asdict(config)

            # Upsert the item (insert or update)
            self._container.upsert_item(body=cache_data)  # pyright: ignore[reportOptionalMemberAccess]
            print(f"✓ Cached result for: '{normalized_key}' (stored as: '{safe_id}')")

        except Exception as e:
            print(f"Warning: Failed to save to cache for '{normalized_key}': {e}")
