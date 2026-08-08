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
| **KEDA** | `kedacore/keda` | Event-driven autoscaler. Drives the embed worker `ScaledJob` from Kafka lag, scaling 0→N and back to 0. |

The object store is **Azure Blob Storage**, a managed service — there is no
in-cluster chart for it. The workers read `AZURE_STORAGE_CONNECTION_STRING` from
the `booksearch-secrets` Secret (see `../k8s/base/secret.yaml`). Locally, the
`docker-compose.yml` stack runs the **Azurite** Blob emulator in its place; on a
`kind` cluster you can run Azurite as a plain Deployment and point the
connection string at it. The object-store code is backend-agnostic, so setting
`OBJECT_STORE_BACKEND=s3` switches it to any S3-compatible store instead.

> Local dev note: `docker-compose.yml` at the repo root uses **Redpanda** as a
> protocol-compatible Kafka stand-in and **Azurite** as the Azure Blob emulator
> for fast, low-memory iteration. In-cluster we run **upstream Apache Kafka** so
> the deployed system is the real thing. The application speaks only the Kafka
> protocol and the Azure Blob API, so nothing app-side changes.

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
```

The object store is managed Azure Blob — set the connection string in
`booksearch-secrets` rather than installing a chart. (For a fully local `kind`
run, deploy Azurite yourself and point the connection string at it.)

Then deploy the application (see `../k8s/README.md`):

```sh
kubectl apply -k ../k8s/overlays/local
```

## Uninstall

```sh
helm -n booksearch uninstall kafka qdrant
helm -n keda uninstall keda
kubectl delete namespace booksearch keda
```
