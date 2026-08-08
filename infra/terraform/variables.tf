variable "subscription_id" {
  type        = string
  description = "Azure subscription ID to deploy into. Or set ARM_SUBSCRIPTION_ID."
}

variable "project" {
  type        = string
  description = "Short project slug used in resource names."
  default     = "booksearch"
}

variable "environment" {
  type        = string
  description = "Environment slug (e.g. dev, prod) used in resource names and tags."
  default     = "dev"
}

variable "location" {
  type        = string
  description = "Azure region for all resources."
  default     = "westus2"
}

variable "storage_account_name" {
  type        = string
  description = "Globally-unique name for the embeddings storage account (3-24 lowercase alphanumerics)."

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "storage_account_name must be 3-24 lowercase letters/digits (Azure storage naming rule)."
  }
}

variable "storage_container_name" {
  type        = string
  description = "Blob container the embedding pipeline reads slices from and writes shards to."
  default     = "embeddings"
}

variable "kubernetes_version" {
  type        = string
  description = "AKS control-plane version. Leave null to take the region default."
  default     = null
}

variable "node_vm_size" {
  type        = string
  description = "VM size for the default (system) node pool."
  default     = "Standard_D4s_v5"
}

variable "node_count" {
  type        = number
  description = "Node count for the default node pool."
  default     = 2
}

variable "k8s_namespace" {
  type        = string
  description = "Kubernetes namespace the workloads run in. Must match deploy/k8s (kustomization namespace)."
  default     = "booksearch"
}

variable "k8s_service_account" {
  type        = string
  description = "Worker ServiceAccount name federated to the managed identity. Must match deploy/k8s/base/serviceaccount.yaml."
  default     = "booksearch-worker"
}
