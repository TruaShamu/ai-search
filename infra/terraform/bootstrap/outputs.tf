output "resource_group_name" {
  description = "Resource group holding the state account. Put in ../backend.hcl."
  value       = azurerm_resource_group.state.name
}

output "storage_account_name" {
  description = "State storage account name. Put in ../backend.hcl."
  value       = azurerm_storage_account.state.name
}

output "container_name" {
  description = "State blob container. Put in ../backend.hcl."
  value       = azurerm_storage_container.state.name
}

output "backend_hcl" {
  description = "Paste into ../backend.hcl, then `terraform init` the root module."
  value       = <<-EOT
    resource_group_name  = "${azurerm_resource_group.state.name}"
    storage_account_name = "${azurerm_storage_account.state.name}"
    container_name       = "${azurerm_storage_container.state.name}"
    key                  = "booksearch.tfstate"
  EOT
}
