variable "subscription_id" {
  type        = string
  description = "Azure subscription ID to deploy the state account into. Or set ARM_SUBSCRIPTION_ID."
}

variable "resource_group_name" {
  type        = string
  description = "Resource group that holds the Terraform state storage account."
  default     = "rg-booksearch-tfstate"
}

variable "storage_account_name" {
  type        = string
  description = "Globally-unique name for the state storage account (3-24 lowercase alphanumerics)."

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "storage_account_name must be 3-24 lowercase letters/digits (Azure storage naming rule)."
  }
}

variable "location" {
  type        = string
  description = "Azure region for the state account."
  default     = "westus2"
}
