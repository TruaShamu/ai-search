# Third-party infrastructure via Helm

Everything in this project that is *not* first-party application code is
installed from an upstream Helm chart, pinned to a specific chart version. The
application itself (the FastAPI service and the embedding worker) is deployed
with Kustomize from `../k8s` — the Helm-for-dependencies / Kustomize-for-app
split keeps the two concerns from leaking into each other.

| Component | Chart | Why |
|-----------|-------|-----|
| **Apache Kafka** | `bitnami/kafka` (KRaft, no ZooKeeper) | Work queue for the embedding backfill. The consumer-group lag on `embed-tasks` is what KEDA scales the workers on. |
| **Qdrant** | `qdrant/qdrant` | Vector store (dense + sparse). Runs as a StatefulSet with a PVC. |
| **MinIO** | `bitnami/minio` | S3-compatible object store for corpus slices, dense shards and error diagnostics. Swap for real S3 by pointing `S3_ENDPOINT_URL` at AWS and dropping this release. |
| **KEDA** | `kedacore/keda` | Event-driven autoscaler. Drives the embed worker `ScaledJob` from Kafka lag, scaling 0→N and back to 0. |

> Local dev note: `docker-compose.yml` at the repo root uses **Redpanda** as a
> protocol-compatible Kafka stand-in for fast, low-memory iteration. In-cluster
> we run **upstream Apache Kafka** so the deployed system is the real thing.
> The application speaks only the Kafka protocol, so nothing app-side changes.

## Install

Requires `helm` 3.14+ and a cluster (`kind`, `minikube`, AKS, EKS, …).

```sh
# one namespace for infra + app
kubectl create namespace booksearch

helm repo add bitnami   https://charts.bitnami.com/bitnami
helm repo add qdrant    https://qdrant.github.io/qdrant-helm
helm repo add kedacore   https://kedacore.github.io/charts
helm repo update

# KEDA (cluster-scoped operator; keep it in its own namespace)
helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace --version 2.15.1

# Apache Kafka (KRaft mode, single broker for a demo cluster)
helm upgrade --install kafka bitnami/kafka \
  --namespace booksearch --version 30.0.4 \
  -f values-kafka.yaml

# Qdrant
helm upgrade --install qdrant qdrant/qdrant \
  --namespace booksearch --version 1.12.1 \
  -f values-qdrant.yaml

# MinIO
helm upgrade --install minio bitnami/minio \
  --namespace booksearch --version 14.7.5 \
  -f values-minio.yaml
```

Then deploy the application (see `../k8s/README.md`):

```sh
kubectl apply -k ../k8s/overlays/local
```

## Uninstall

```sh
helm -n booksearch uninstall kafka qdrant minio
helm -n keda uninstall keda
kubectl delete namespace booksearch keda
```
