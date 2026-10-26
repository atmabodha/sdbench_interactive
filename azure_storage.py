"""Azure Blob Storage integration for SDBench.

The Azure SDK is imported lazily so the package can be used without the optional
``azure`` dependency installed (Azure result upload is off by default).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from config import HarnessConfig

if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient


@functools.cache
def get_blob_service_client(account_name: str) -> BlobServiceClient:
    """Create a BlobServiceClient using DefaultAzureCredential.

    The Azure SDK is imported lazily here (the single client factory) so the package can be
    used without the optional ``azure`` dependency installed.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise ImportError(
            "Azure Blob Storage support requires the 'azure' optional dependency. "
            "Install it with `uv sync --extra azure` (or `pip install 'sdbench[azure]'`)."
        ) from e
    account_url = f"https://{account_name}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    return BlobServiceClient(account_url=account_url, credential=credential)


def validate_azure_access(
    account_name: str, container_name: str, verify_write: bool = False
) -> BlobServiceClient:
    """Validate Azure Storage access and fail hard if invalid.

    Args:
        account_name: Azure storage account name
        container_name: Container name to validate
        verify_write: Whether to test write permissions (default: False)

    Returns:
        BlobServiceClient if successful
    """
    # Reuse the lazily-imported client factory (raises a friendly error if the `azure` extra is
    # not installed). Construction does not hit the network; auth is verified by the calls below.
    blob_service_client = get_blob_service_client(account_name)
    from azure.core.exceptions import (
        AzureError,
        ClientAuthenticationError,
        HttpResponseError,
        ResourceNotFoundError,
    )

    try:
        # Step 1: Authenticate
        print(f"🔐 Authenticating to Azure Storage account: {account_name}")
        print("✓ Authentication successful")

        # Step 2: Check if container exists
        print(f"📦 Checking container: {container_name}")
        container_client = blob_service_client.get_container_client(container_name)

        # This actually calls the API to verify access
        if not container_client.exists():
            print(f"❌ ERROR: Container '{container_name}' does not exist", file=sys.stderr)
            sys.exit(1)

        print("✓ Container exists")

        # Step 3: Verify write permissions with a test blob (optional)
        if verify_write:
            print("✍️  Verifying write permissions...")
            test_blob_name = ".health_check_test"
            blob_client = container_client.get_blob_client(test_blob_name)

            try:
                blob_client.upload_blob(b"health check", overwrite=True)
                blob_client.delete_blob()
                print("✓ Write permissions verified")
            except HttpResponseError as e:
                if e.status_code == 403:
                    print(
                        f"❌ ERROR: No write permissions to container '{container_name}'",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                raise

        print("✅ Azure Storage validation complete!\n")
        return blob_service_client

    except ClientAuthenticationError as e:
        print(f"❌ AUTHENTICATION ERROR: {e}", file=sys.stderr)
        print("   Try running: az login", file=sys.stderr)
        sys.exit(1)

    except ResourceNotFoundError as e:
        print(f"❌ RESOURCE NOT FOUND: {e}", file=sys.stderr)
        print(
            f"   Storage account '{account_name}' may not exist or you don't have access",
            file=sys.stderr,
        )
        sys.exit(1)

    except AzureError as e:
        print(f"❌ AZURE ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def write_output_file(config: HarnessConfig, filename: str, content: str) -> str | None:
    """Write a file to the configured Azure Blob Storage output folder."""
    if (
        config.azure_storage_account_name is None
        or config.azure_storage_container_name is None
        or config.azure_storage_result_folder is None
    ):
        raise ValueError(
            "Azure storage is not fully configured: azure_storage_account_name, "
            "azure_storage_container_name and azure_storage_result_folder must all be set "
            "to write output files."
        )
    blob_filename = str(Path(config.azure_storage_result_folder, Path(filename)))
    return upload_blob(
        config.azure_storage_account_name,
        config.azure_storage_container_name,
        blob_filename,
        content,
        debug=config.debug,
    )


def upload_blob(
    account_name: str, container_name: str, blob_name: str, content: str, debug: bool = False
) -> str | None:
    """Upload JSON data to blob storage.

    Args:
        account_name: Azure storage account name
        container_name: Container name
        blob_name: Name/path for the blob
        content: Content to upload
        debug: Whether to print upload status

    Returns:
        Blob URL if successful, None if failed
    """
    try:
        blob_client = get_blob_service_client(account_name).get_blob_client(
            container=container_name, blob=blob_name
        )

        blob_client.upload_blob(content, overwrite=True)

        blob_url = blob_client.url
        if debug:
            print(f"✓ Uploaded to Azure: {blob_url}")
        return blob_url

    except Exception as e:
        print(f"❌ Failed to upload {blob_name} to Azure: {e}", file=sys.stderr)
        return None


def list_blobs(account_name: str, container_name: str, prefix: str | None = None) -> list[str]:
    """List blobs in a container.

    Args:
        account_name: Azure storage account name
        container_name: Container name
        prefix: Optional prefix to filter blobs

    Returns:
        List of blob names
    """
    try:
        container_client = get_blob_service_client(account_name).get_container_client(
            container_name
        )
        blob_names: list[str] = []

        for blob in container_client.list_blobs(name_starts_with=prefix):
            blob_names.append(blob.name)

        return blob_names

    except Exception as e:
        print(f"❌ Failed to list blobs: {e}", file=sys.stderr)
        return []


def download_blob(account_name: str, container_name: str, blob_name: str) -> dict[str, Any] | None:
    """Download and parse a JSON blob.

    Args:
        account_name: Azure storage account name
        container_name: Container name
        blob_name: Name of the blob to download

    Returns:
        Parsed JSON dict if successful, None if failed
    """
    try:
        blob_client = get_blob_service_client(account_name).get_blob_client(
            container=container_name, blob=blob_name
        )

        blob_data = blob_client.download_blob().readall()
        return json.loads(blob_data.decode("utf-8"))

    except Exception as e:
        print(f"❌ Failed to download blob {blob_name}: {e}", file=sys.stderr)
        return None
