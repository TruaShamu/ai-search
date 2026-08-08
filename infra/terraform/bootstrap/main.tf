# Bootstrap: the remote-state backend the root module uses.
#
# Chicken-and-egg: you cannot store Terraform state in an Azure Storage account
# that Terraform has not created yet. This tiny module runs with LOCAL state and
# provisions exactly that account + a versioned `tfstate` container. Run it once
# per subscription; the root module (../) then keys its `azurerm` backend here.
#
#   cd infra/terraform/bootstrap
#   terraform init && terraform apply
#   # feed the outputs into ../backend.hcl, then `terraform init` the root module

terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "azurerm_resource_group" "state" {
  name     = var.resource_group_name
  location = var.location
  tags = {
    project    = "booksearch"
    purpose    = "terraform-remote-state"
    managed_by = "terraform"
  }
}

resource "azurerm_storage_account" "state" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.state.name
  location                 = azurerm_resource_group.state.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  min_tls_version          = "TLS1_2"

  # State can contain sensitive values; keep it locked down and recoverable.
  allow_nested_items_to_be_public = false

  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 30
    }
  }

  tags = {
    project    = "booksearch"
    purpose    = "terraform-remote-state"
    managed_by = "terraform"
  }
}

resource "azurerm_storage_container" "state" {
  name                  = "tfstate"
  storage_account_id    = azurerm_storage_account.state.id
  container_access_type = "private"
}
