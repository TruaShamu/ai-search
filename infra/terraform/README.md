# Infrastructure as Code (Terraform)

Terraform provisions the **cloud substrate** booksearch runs on. It stops at the
cluster boundary on purpose:

| Layer | Owned by | Where |
|-------|----------|-------|
| Resource group, **Storage account + container**, **AKS**, **workload identity** + role assignment | **Terraform** | here |
| Kafka, Qdrant | **Helm** (pinned charts) | [`../../deploy/helm`](../../deploy/helm) |
| API + embedding worker | **Kustomize** | [`../../deploy/k8s`](../../deploy/k8s) |

Terraform manages resources with a *lifecycle*; it does not wrap the Helm
releases or the Kustomize app (the `helm`/`kubectl` providers exist but muddy the
separation). The clean split is **Terraform provisions the platform → Helm and
Kustomize deploy onto it.**

## What it creates

- **Resource group** `rg-booksearch-<env>`.
- **Storage account** for the embedding pipeline's slices/shards, with
  **shared account keys disabled** — the only way in is Entra ID.
- **AKS** with the **OIDC issuer** and **workload identity** enabled, and **KEDA
  as the managed add-on** (the embed `ScaledJob` scales on Kafka lag through it,
  so no KEDA Helm release is needed).
- **User-assigned managed identity** + **federated credential** bound to the
  `booksearch:booksearch-worker` ServiceAccount, and a **`Storage Blob Data
  Contributor`** role assignment scoped to just the one storage account.

The result: the worker reads/writes Blob storage with **no secret** —
`DefaultAzureCredential` exchanges its projected ServiceAccount token for an
Entra ID token. `AZURE_STORAGE_CONNECTION_STRING` is only used locally (Azurite).

## Layout

```
infra/terraform/
├── bootstrap/            # one-time: creates the remote-state storage account
├── versions.tf          # provider pins + azurerm remote backend + provider config
├── variables.tf
├── main.tf              # RG, storage, AKS, workload identity
├── outputs.tf
├── terraform.tfvars.example
└── backend.hcl.example
```

## Usage

### 0. Prereqs
`terraform >= 1.9`, `az login` (or `ARM_*` env creds), and a subscription id.

### 1. Bootstrap remote state (once per subscription)

State can hold sensitive values, so it lives in Azure Storage, not on your
laptop. The `bootstrap/` module creates that account with **local** state
(chicken-and-egg):

```sh
cd infra/terraform/bootstrap
terraform init
terraform apply -var subscription_id=<sub> -var storage_account_name=<globally-unique>
terraform output -raw backend_hcl > ../backend.hcl
```

### 2. Provision the platform

```sh
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # fill subscription_id + storage_account_name
terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

### 3. Wire the cluster to what Terraform built

```sh
# kubeconfig
$(terraform output -raw get_credentials)

# passwordless: point the app at the account and annotate the worker SA
kubectl create namespace booksearch
# set AZURE_STORAGE_ACCOUNT_URL in deploy/k8s/base/configmap.yaml to:
terraform output -raw storage_account_url
# stamp the identity client id onto the ServiceAccount:
$(terraform output -raw annotate_service_account)
```

Then deploy the Helm dependencies and the Kustomize app per
[`../../deploy`](../../deploy). KEDA is already present as the AKS add-on, so skip
its Helm release.

### Offline validation

No Azure or state account needed:

```sh
terraform fmt -recursive -check
terraform init -backend=false
terraform validate
```

## Cost note

AKS + a 2-node pool bills by the hour. Provision, screenshot / demo, then
`terraform destroy`. The code is the artifact — you do not need to leave it
running.
