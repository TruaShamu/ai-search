# Booksearch cloud substrate.
#
# Terraform owns the *platform*: the resource group, the object store, the AKS
# cluster, and the workload identity that lets the embedding worker reach Blob
# storage without a secret. The in-cluster dependencies (Kafka, Qdrant) come
# from Helm and the app itself from Kustomize (see ../../deploy) — Terraform
# stops at the cluster boundary on purpose.

locals {
  name_prefix = "${var.project}-${var.environment}"
  tags = {
    project     = var.project
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "azurerm_resource_group" "this" {
  name     = "rg-${local.name_prefix}"
  location = var.location
  tags     = local.tags
}

# --------------------------------------------------------------------------- #
# Object store — the embedding pipeline's slices and shards.                   #
# --------------------------------------------------------------------------- #
resource "azurerm_storage_account" "embeddings" {
  name                = var.storage_account_name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location

  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"

  # Passwordless by construction: disable the shared account keys entirely so the
  # only way in is Entra ID (the worker's federated workload identity). This is
  # what makes the AZURE_STORAGE_CONNECTION_STRING secret unnecessary in-cluster.
  shared_access_key_enabled       = false
  https_traffic_only_enabled      = true
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  tags = local.tags
}

resource "azurerm_storage_container" "embeddings" {
  name                  = var.storage_container_name
  storage_account_id    = azurerm_storage_account.embeddings.id
  container_access_type = "private"
}

# --------------------------------------------------------------------------- #
# AKS — the compute substrate for the API + KEDA-scaled embedding workers.     #
# --------------------------------------------------------------------------- #
resource "azurerm_kubernetes_cluster" "this" {
  name                = "aks-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  dns_prefix          = local.name_prefix
  kubernetes_version  = var.kubernetes_version

  # Workload Identity Federation: the OIDC issuer lets a Kubernetes
  # ServiceAccount token be exchanged for an Entra ID token, no secret involved.
  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  # KEDA as the managed add-on instead of a Helm release — the embed worker
  # ScaledJob (deploy/k8s/base/embed-scaledjob.yaml) scales on Kafka lag through
  # exactly this. One less chart to run and patch.
  workload_autoscaler_profile {
    keda_enabled = true
  }

  default_node_pool {
    name       = "system"
    vm_size    = var.node_vm_size
    node_count = var.node_count
    # Embedding pods are memory-heavy (4Gi requests==limits); keep temp disk off
    # the OS disk and let the scheduler reserve real footprint.
    os_disk_size_gb = 64
  }

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

# --------------------------------------------------------------------------- #
# Workload identity — passwordless Blob access for the embedding worker.       #
# --------------------------------------------------------------------------- #
resource "azurerm_user_assigned_identity" "worker" {
  name                = "id-${local.name_prefix}-worker"
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  tags                = local.tags
}

# Bind the identity to the worker's Kubernetes ServiceAccount. The subject must
# match `system:serviceaccount:<namespace>:<sa>` exactly, hence the shared
# defaults with deploy/k8s.
resource "azurerm_federated_identity_credential" "worker" {
  name                = "fic-${local.name_prefix}-worker"
  resource_group_name = azurerm_resource_group.this.name
  parent_id           = azurerm_user_assigned_identity.worker.id
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.this.oidc_issuer_url
  subject             = "system:serviceaccount:${var.k8s_namespace}:${var.k8s_service_account}"
}

# Least privilege: the worker gets data-plane Blob access to this one account,
# nothing else. No management-plane rights, no other accounts.
resource "azurerm_role_assignment" "worker_blob" {
  scope                = azurerm_storage_account.embeddings.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.worker.principal_id
}
