"""CLI tool for testing Azure Blob Storage integration.

This tool allows you to test Azure Storage connectivity, authentication,
and permissions before running the full harness.

Usage:
    python -m sdbench.cli.azure_storage --account-name myaccount --container-name results
    python -m sdbench.cli.azure_storage --account-name myaccount --container-name results --test-upload
    python -m sdbench.cli.azure_storage --account-name myaccount --container-name results --list-blobs
    python -m sdbench.cli.azure_storage --account-name myaccount --container-name results --list-blobs --prefix json/
"""

import argparse
import datetime
import json
import sys
from typing import Any

import dotenv

from sdbench.azure_storage import (
    download_blob,
    get_blob_service_client,
    list_blobs,
    upload_blob,
    validate_azure_access,
)


def main():
    """Test Azure Blob Storage connectivity and permissions."""
    dotenv.load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Test Azure Blob Storage integration for SDBench")
    parser.add_argument("--account-name", required=True, help="Azure storage account name")
    parser.add_argument("--container-name", required=True, help="Container name to test")
    parser.add_argument("--test-upload", action="store_true", help="Upload a test JSON blob")
    parser.add_argument("--list-blobs", action="store_true", help="List all blobs in the container")
    parser.add_argument(
        "--prefix", type=str, help="Prefix to filter blobs when listing (e.g., 'json/')"
    )
    parser.add_argument(
        "--download-blob",
        type=str,
        metavar="BLOB_NAME",
        help="Download and display a specific blob",
    )
    parser.add_argument(
        "--delete-test-blobs",
        action="store_true",
        help="Delete all test blobs created by this tool",
    )

    args = parser.parse_args()

    print("=== Azure Blob Storage Test Tool ===")
    print(f"Account: {args.account_name}")
    print(f"Container: {args.container_name}")
    print()

    # Validate access (this will exit on failure)
    validate_azure_access(args.account_name, args.container_name)

    # Test upload
    if args.test_upload:
        print("=" * 60)
        print("TEST UPLOAD")
        print("=" * 60)

        test_data: dict[str, Any] = {
            "test": "azure_storage_cli_test",
            "timestamp": datetime.datetime.now().isoformat(),
            "account_name": args.account_name,
            "container_name": args.container_name,
            "message": "This is a test blob created by sdbench.cli.azure_storage",
        }

        test_blob_name = (
            f"test/azure_cli_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        print(f"Uploading test blob: {test_blob_name}")
        test_content = json.dumps(test_data, indent=2)
        blob_url = upload_blob(
            args.account_name, args.container_name, test_blob_name, test_content, debug=True
        )

        if blob_url:
            print("✅ Test upload successful!")
            print(f"   Blob URL: {blob_url}")
            print("   You can view it in Azure Portal or download with:")
            print(
                f"   python -m sdbench.cli.azure_storage --account-name {args.account_name} --container-name {args.container_name} --download-blob {test_blob_name}"
            )
        else:
            print("❌ Test upload failed")
            sys.exit(1)

        print()

    # List blobs
    if args.list_blobs:
        print("=" * 60)
        print("LIST BLOBS")
        print("=" * 60)

        if args.prefix:
            print(f"Listing blobs with prefix: {args.prefix}")
        else:
            print("Listing all blobs in container")

        blob_names = list_blobs(args.account_name, args.container_name, prefix=args.prefix)

        if blob_names:
            print(f"\nFound {len(blob_names)} blob(s):\n")
            for i, name in enumerate(blob_names, 1):
                print(f"  {i}. {name}")
        else:
            print("\nNo blobs found")

        print()

    # Download blob
    if args.download_blob:
        print("=" * 60)
        print("DOWNLOAD BLOB")
        print("=" * 60)

        print(f"Downloading: {args.download_blob}")

        blob_data = download_blob(args.account_name, args.container_name, args.download_blob)

        if blob_data:
            print("✅ Downloaded successfully\n")
            print(json.dumps(blob_data, indent=2))
        else:
            print("❌ Download failed")
            sys.exit(1)

        print()

    # Delete test blobs
    if args.delete_test_blobs:
        print("=" * 60)
        print("DELETE TEST BLOBS")
        print("=" * 60)

        test_blob_names = list_blobs(args.account_name, args.container_name, prefix="test/")

        if not test_blob_names:
            print("No test blobs found to delete")
        else:
            print(f"Found {len(test_blob_names)} test blob(s) to delete:")
            for name in test_blob_names:
                print(f"  • {name}")

            confirm = input("\nDelete these blobs? [y/N]: ")
            if confirm.lower() == "y":
                container_client = get_blob_service_client(args.account_name).get_container_client(
                    args.container_name
                )
                deleted_count = 0
                for blob_name in test_blob_names:
                    try:
                        container_client.delete_blob(blob_name)
                        print(f"  ✓ Deleted: {blob_name}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to delete {blob_name}: {e}")

                print(f"\n✅ Deleted {deleted_count}/{len(test_blob_names)} test blobs")
            else:
                print("Cancelled")

        print()

    # If no actions specified, just show success message
    if not any([args.test_upload, args.list_blobs, args.download_blob, args.delete_test_blobs]):
        print("=" * 60)
        print("✅ Azure Storage access verified successfully!")
        print("=" * 60)
        print()

    sys.exit(0)


if __name__ == "__main__":
    main()
