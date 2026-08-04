"""Index build pipeline — the path from corpus rows to a searchable index.

``worker.py`` is the queue-driven embedding worker deployed as an Azure Container
Apps job (see ``Dockerfile.embed`` and ``infra/aca-embed-job.bicep``);
``assemble.py`` stitches the shards it produces into the FAISS index and metadata
that ``src/qdrant/migrate.py`` loads.

Deliberately empty of imports. ``assemble`` imports ``faiss`` at module scope, and
the embedding container does not install ``faiss-cpu`` -- re-exporting anything
here would import it and crash every worker replica on startup.
"""
