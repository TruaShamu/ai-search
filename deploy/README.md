# Deployment

BookSearch is **portable by design**. The application code speaks only
vendor-neutral protocols — the Kafka wire protocol for its work queue and the S3
API for object storage — behind small backend interfaces
(`src/indexing/backends/`). That lets the exact same images run on two targets:

| Target | Queue | Object store | Vector store | Orchestration |
|--------|-------|--------------|--------------|---------------|
| **Kubernetes** (portable default) | Apache Kafka | S3 / MinIO | Qdrant | KEDA `ScaledJob` + `Deployment`/`HPA` |
| **Azure Container Apps** (reference cloud) | Storage Queue | Blob | Qdrant | Event-driven ACA Job + Container App |

Switching between them is configuration, not code:

```sh
# portable
QUEUE_BACKEND=kafka  OBJECT_STORE_BACKEND=s3
# reference cloud
QUEUE_BACKEND=azure  OBJECT_STORE_BACKEND=azure
```

## Layout

```
deploy/
  helm/     third-party infra as pinned Helm charts (Kafka, Qdrant, MinIO, KEDA)
  k8s/      first-party app as Kustomize (API Deployment/Service/HPA + embed ScaledJob)
../infra/   Azure Bicep — the reference ACA deployment (still supported)
../docker-compose.yml   local one-machine stack (Redpanda + MinIO + Qdrant)
```

The **Helm-for-infrastructure / Kustomize-for-application** split is deliberate:
upstream dependencies come from their maintained charts pinned to a version, and
the code we own is templated by us. See [`helm/README.md`](helm/README.md) and
[`k8s/README.md`](k8s/README.md).

## Local development

The fastest loop needs no cluster. `docker-compose.yml` brings up the same
component types the cluster runs, on one machine:

```sh
docker compose up -d          # Kafka (Redpanda) + MinIO + Qdrant
# run the API / worker against them, or add the "app" profile to containerise:
docker compose --profile app up -d
```

> **Kafka locally vs in-cluster.** compose uses **Redpanda**, a protocol-compatible
> Kafka stand-in that is a single ~1GB binary with no ZooKeeper/KRaft setup — ideal
> for fast iteration. The Kubernetes deployment runs **upstream Apache Kafka** (Bitnami
> chart, KRaft mode). The application only ever speaks the Kafka protocol, so this is
> transparent to the code; it is a dev-ergonomics choice, made explicit.

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
helm upgrade --install minio  bitnami/minio  -n booksearch --version 14.7.5 -f deploy/helm/values-minio.yaml

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
