output "resource_group_name" {
  description = "Resource group all booksearch resources live in."
  value       = azurerm_resource_group.this.name
}

output "aks_cluster_name" {
  description = "AKS cluster name. Fetch kubeconfig with the get_credentials output."
  value       = azurerm_kubernetes_cluster.this.name
}

output "get_credentials" {
  description = "Command to merge the AKS cluster into your kubeconfig."
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.this.name} --name ${azurerm_kubernetes_cluster.this.name}"
}

output "oidc_issuer_url" {
  description = "AKS OIDC issuer backing the worker's federated credential."
  value       = azurerm_kubernetes_cluster.this.oidc_issuer_url
}

output "storage_account_name" {
  description = "Embeddings storage account name."
  value       = azurerm_storage_account.embeddings.name
}

output "storage_account_url" {
  description = "Blob endpoint. Set as AZURE_STORAGE_ACCOUNT_URL for passwordless access (configmap / scaledjob env)."
  value       = azurerm_storage_account.embeddings.primary_blob_endpoint
}

output "worker_identity_client_id" {
  description = "Managed-identity client id. Annotate the worker ServiceAccount: azure.workload.identity/client-id=<this>."
  value       = azurerm_user_assigned_identity.worker.client_id
}

output "annotate_service_account" {
  description = "Command to wire the ServiceAccount to the managed identity after apply."
  value       = "kubectl annotate serviceaccount ${var.k8s_service_account} -n ${var.k8s_namespace} azure.workload.identity/client-id=${azurerm_user_assigned_identity.worker.client_id} --overwrite"
}
