import os
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.embed import embed_and_index


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    connection_string = require_env("AZURE_STORAGE_CONNECTION_STRING")
    container_name = require_env("STORAGE_CONTAINER")
    input_blob = require_env("INPUT_BLOB")
    output_prefix = require_env("OUTPUT_PREFIX").strip("/")
    tier_filter = int(os.getenv("TIER_FILTER", "1"))
    dim = int(os.getenv("EMBED_DIM", "256"))

    work_root = Path("/app")
    input_path = work_root / "data" / "processed" / Path(input_blob).name
    output_dir = work_root / "data" / "index"

    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to blob container '{container_name}'...")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)

    print(f"Downloading input blob '{input_blob}' to '{input_path}'...")
    with open(input_path, "wb") as handle:
        download_stream = container_client.download_blob(input_blob)
        handle.write(download_stream.readall())
    print("Input download complete.")

    print(f"Starting embedding pipeline (tier={tier_filter}, dim={dim})...")
    embed_and_index(
        jsonl_path=input_path,
        output_dir=output_dir,
        tier_filter=tier_filter,
        dim=dim,
    )
    print("Embedding pipeline complete.")

    for file_name in ("faiss.index", "metadata.jsonl"):
        local_path = output_dir / file_name
        blob_name = f"{output_prefix}/{file_name}" if output_prefix else file_name
        print(f"Uploading '{local_path}' to '{blob_name}'...")
        with open(local_path, "rb") as handle:
            container_client.upload_blob(blob_name, handle, overwrite=True)

    print("All output files uploaded successfully.")


if __name__ == "__main__":
    main()
