# Deployment

BookSearch is **portable by design**. The application code speaks only
vendor-neutral protocols — the Kafka wire protocol for its work queue and a
small object-store interface (`src/indexing/backends/`) with interchangeable
Azure Blob and S3 implementations. That lets the exact same images run on two
targets:

| Target | Queue | Object store | Vector store | Orchestration | Registry |
|--------|-------|--------------|--------------|---------------|----------|
| **Kubernetes** (portable default) | Apache Kafka | Azure Blob (S3 optional) | Qdrant | KEDA `ScaledJob` + `Deployment`/`HPA` | GHCR |
| **Azure Container Apps** (reference cloud) | Storage Queue | Blob | Qdrant | Event-driven ACA Job + Container App | ACR |

The queue and compute are fully portable — Kafka on Kubernetes replaces Azure
Storage Queue on ACA. Object storage stays on managed **Azure Blob** by default
(no AWS account required), but the backend is pluggable: set
`OBJECT_STORE_BACKEND=s3` to run against any S3-compatible store. Switching is
configuration, not code:

```sh
# portable compute, managed Blob (default)
QUEUE_BACKEND=kafka  OBJECT_STORE_BACKEND=azure
# fully S3-compatible object store
QUEUE_BACKEND=kafka  OBJECT_STORE_BACKEND=s3
# reference cloud (ACA)
QUEUE_BACKEND=azure  OBJECT_STORE_BACKEND=azure
```

## Layout

```
deploy/
  helm/     third-party infra as pinned Helm charts (Kafka, Qdrant, KEDA)
  k8s/      first-party app as Kustomize (API Deployment/Service/HPA + embed ScaledJob)
../infra/terraform/   Azure platform as Terraform (RG, Storage, AKS, workload identity)
../infra/*.bicep      the reference ACA deployment (still supported)
../docker-compose.yml   local one-machine stack (Redpanda + Azurite + Qdrant)
```

The cloud substrate — resource group, storage account, the AKS cluster itself,
and the passwordless **workload identity** the embedding worker uses to reach
Blob storage — is provisioned with **Terraform** (`../infra/terraform`). Terraform
stops at the cluster boundary: Helm installs the in-cluster dependencies and
Kustomize deploys the app onto the cluster it created.

The **Helm-for-infrastructure / Kustomize-for-application** split is deliberate:
upstream dependencies come from their maintained charts pinned to a version, and
the code we own is templated by us. Application images are published to
**GHCR** (`ghcr.io/<owner>/booksearch-{api,embed}`) by
[`publish-images.yml`](../.github/workflows/publish-images.yml). See
[`helm/README.md`](helm/README.md) and [`k8s/README.md`](k8s/README.md).

## Local development

The fastest loop needs no cluster. `docker-compose.yml` brings up the same
component types the cluster runs, on one machine:

```sh
docker compose up -d          # Kafka (Redpanda) + Azurite (Blob) + Qdrant + Jaeger
# run the API / worker against them, or add the "app" profile to containerise:
docker compose --profile app up -d
```

> **Local stand-ins.** compose uses **Redpanda**, a protocol-compatible Kafka
> stand-in (single ~1GB binary, no ZooKeeper/KRaft setup), and **Azurite**, the
> official Azure Blob emulator. The Kubernetes deployment runs **upstream Apache
> Kafka** (Bitnami chart, KRaft mode) and a managed Azure Blob account. The
> application only ever speaks the Kafka protocol and the Azure Blob API, so
> this is transparent to the code; it is a dev-ergonomics choice, made explicit.

## Observability (distributed tracing)

The API and worker are instrumented with **OpenTelemetry**. Tracing is off
unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set, so production, tests, and a clean
clone pay nothing (see [`../src/telemetry.py`](../src/telemetry.py)). The compose
stack runs **Jaeger** and points both services at it, so traces work out of the
box:

```sh
docker compose --profile app up -d
# ... issue a search or run an embedding backfill ...
open http://localhost:16686        # Jaeger UI
```

A search request is one trace spanning `query.understand → retrieval.embed_query
→ retrieval.qdrant_query → retrieval.fusion_rank → rerank.cross_encoder`, and
`/ask` adds `rag.build_context → rag.generate → rag.validate_citations` with the
outbound Azure OpenAI call captured automatically via httpx instrumentation.

The embedding backfill is a **single distributed trace across the Kafka work
queue**: the enqueue span injects a W3C `traceparent` into each message's Kafka
headers, and every worker that picks up a slice continues that trace
(`enqueue.slice → embed.process_message → embed.download_slice → embed.encode →
embed.write_shard`). Point `OTEL_EXPORTER_OTLP_ENDPOINT` at an in-cluster
collector (OpenTelemetry Collector or Jaeger with OTLP) to get the same in
Kubernetes.

## Kubernetes quick start (kind)

```sh
# 1. infra
kubectl create namespace booksearch
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add qdrant  https://qdrant.github.io/qdrant-helm
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm upgrade --install keda   kedacore/keda  -n keda --create-namespace --version 2.15.1
helm upgrade --install kafka  bitnami/kafka  -n booksearch --version 30.0.4 -f deploy/helm/values-kafka.yaml
helm upgrade --install qdrant qdrant/qdrant  -n booksearch --version 1.12.1 -f deploy/helm/values-qdrant.yaml
# Object store = managed Azure Blob: set AZURE_STORAGE_CONNECTION_STRING in
# deploy/k8s/base/secret.yaml (or run Azurite in-cluster for a fully local demo).

# 2. app
docker build -f Dockerfile.api   -t booksearch-api:local   .
docker build -f Dockerfile.embed -t booksearch-embed:local .
kind load docker-image booksearch-api:local booksearch-embed:local
kubectl apply -k deploy/k8s/overlays/local
```

## Validation (no cluster required)

Both are run in CI by [`.github/workflows/validate-k8s.yml`](../.github/workflows/validate-k8s.yml):

```sh
# app manifests, incl. the KEDA ScaledJob CRD schema
kubectl kustomize deploy/k8s/overlays/local | kubeconform -strict -summary \
  -schema-location default \
  -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'

# dependency charts render with the pinned values
helm template kafka bitnami/kafka --version 30.0.4 -f deploy/helm/values-kafka.yaml >/dev/null
```

## Continuous deployment to AKS

CI/CD is a two-stage pipeline against the Terraform-provisioned cluster
([`infra/terraform`](../infra/terraform)):

1. **[`publish-images.yml`](../.github/workflows/publish-images.yml)** — on push
   to `master`, builds and pushes `booksearch-{api,embed}` to GHCR, tagged with
   both `latest` and the commit SHA.
2. **[`deploy.yml`](../.github/workflows/deploy.yml)** ("Deploy to AKS") — runs
   on that workflow's success, lints + tests, then pins the
   [`overlays/aks`](k8s/overlays/aks) image tags to the same SHA
   (`kustomize edit set image`), `kubectl apply -k`s it, and waits on the API
   rollout (its `/ready` probe).

The deploy job **skips neutrally** until the cluster coordinates are configured,
so pushes are not red before the platform exists. To turn it on:

| Kind | Name | Value |
|------|------|-------|
| Repo **variable** | `AKS_RESOURCE_GROUP` | Terraform `resource_group_name` output |
| Repo **variable** | `AKS_CLUSTER_NAME` | Terraform `aks_cluster_name` output |
| Repo **secret** | `AZURE_CREDENTIALS` | service-principal JSON for `azure/login` |

One-time cluster wiring after `terraform apply` (see
[`infra/terraform/README.md`](../infra/terraform/README.md)): set
`AZURE_STORAGE_ACCOUNT_URL` in [`k8s/base/configmap.yaml`](k8s/base/configmap.yaml)
to the `storage_account_url` output, and annotate the worker ServiceAccount with
the managed-identity client id for passwordless Blob access.

