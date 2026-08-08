# Application manifests (Kustomize)

First-party application deployment: the FastAPI search service and the embedding
worker. Third-party infrastructure (Kafka, Qdrant, KEDA) is installed separately
from Helm charts — see [`../helm`](../helm/README.md). Deploy the infra first;
this layer assumes those in-cluster service names exist. The object store is
managed Azure Blob (connection string in `secret.yaml`), not an in-cluster
service. Images are pulled from GHCR (`ghcr.io/<owner>/booksearch-{api,embed}`);
the local overlay remaps them to locally-built tags.

```
base/
  configmap.yaml         backend selection + service endpoints + worker tuning
  secret.yaml            demo credentials incl. Blob connection string (replace in any real cluster)
  api-deployment.yaml    FastAPI service; readiness /ready, liveness /health
  api-service.yaml       ClusterIP :80 -> :8000
  api-hpa.yaml           CPU autoscale 1 -> 3
  embed-scaledjob.yaml   KEDA ScaledJob, scales 0 -> 30 on Kafka lag
overlays/
  local/                 kind/minikube: local image tags + imagePullPolicy: Never
```

## Why this shape

The layout is a one-to-one re-platforming of the original Azure Container Apps
deployment onto vendor-neutral Kubernetes:

| ACA (before) | Kubernetes (now) |
|--------------|------------------|
| `booksearch-api` container app, `minReplicas:1 maxReplicas:3` | `Deployment` + `Service` + `HPA` (1→3), identical probes and 2 vCPU / 4Gi |
| `embed-worker` event Job, Storage-Queue trigger, `maxExecutions:30` | **KEDA `ScaledJob`**, Kafka-lag trigger, `maxReplicaCount:30` |
| Azure Storage Queue | Apache Kafka topic `embed-tasks` |
| Azure Blob | Azure Blob (default) — pluggable to S3 via `OBJECT_STORE_BACKEND=s3` |
| Azure Files–backed Qdrant | Qdrant StatefulSet + PVC (Helm) |
| ACR image registry | GHCR (`ghcr.io/<owner>/booksearch-*`) |

The queue and compute move fully off Azure (Kafka + Kubernetes); object storage
stays on managed **Azure Blob** so no AWS account is needed, while the backend
stays pluggable. The application selects backends purely through the
`booksearch-config` ConfigMap, so the same images run against the reference ACA
path by flipping `QUEUE_BACKEND` / `OBJECT_STORE_BACKEND` — see the Bicep in
[`../../infra`](../../infra) for that deployment, which remains supported.

## Deploy (local, kind)

```sh
# 0. infra (once) — see ../helm/README.md
kubectl create namespace booksearch
# ...helm installs for keda, kafka, qdrant...
# set AZURE_STORAGE_CONNECTION_STRING in base/secret.yaml (Azurite for a local demo)

# 1. build images and load them into the kind cluster
docker build -f Dockerfile.api   -t booksearch-api:local   .
docker build -f Dockerfile.embed -t booksearch-embed:local .
kind load docker-image booksearch-api:local booksearch-embed:local

# 2. apply the app
kubectl apply -k deploy/k8s/overlays/local

# 3. load the index + kick off an embedding backfill
kubectl -n booksearch get pods
kubectl -n booksearch port-forward svc/booksearch-api 8000:80
```

## Validate without a cluster

```sh
kubectl kustomize deploy/k8s/overlays/local | kubeconform -strict -summary \
  -schema-location default \
  -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
```

This is exactly what the `validate-k8s.yml` workflow runs on every change under
`deploy/`.
