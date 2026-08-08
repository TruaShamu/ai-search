terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }

  # Remote state in Azure Storage, provisioned by ./bootstrap. Partial config:
  # the account/container/key live in backend.hcl (git-ignored, copy from
  # backend.hcl.example), supplied at init time:
  #
  #   terraform init -backend-config=backend.hcl
  #
  # Run with `-backend=false` for offline validation (no state account needed).
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}
