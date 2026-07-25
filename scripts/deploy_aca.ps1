[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===== Required placeholders =====
$SubscriptionId = "<subscription-id>"
$ResourceGroup = "ai-search-rg"
$Location = "eastus"
$AcrName = "booksearchacr"
$AcrLoginServer = "booksearchacr.azurecr.io"
$ApiImageName = "booksearch-api:latest"
$AzureOpenAIEndpoint = "https://<azure-openai-resource>.openai.azure.com/"
$AzureOpenAIKey = "<azure-openai-api-key>"
$StorageAccountName = "booksearchblobs"
$CreateStorageAccount = $false
$ContainerAppEnvironmentName = "booksearch-aca-env"
$QdrantAppName = "qdrant"
$ApiAppName = "booksearch-api"
$FileShareName = "qdrant-data"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BicepFile = Join-Path $RepoRoot "infra\aca-search.bicep"
$Dockerfile = Join-Path $RepoRoot "Dockerfile.api"
$VectorizerPath = Join-Path $RepoRoot "data\index\tfidf_vectorizer.pkl"

if (-not (Test-Path -LiteralPath $VectorizerPath)) {
    throw "Missing required file: $VectorizerPath. Generate or copy tfidf_vectorizer.pkl before building the API image."
}

Write-Host "Selecting subscription..."
& az account set --subscription $SubscriptionId --only-show-errors | Out-Null

Write-Host "Validating Bicep template..."
& az bicep build --file $BicepFile --stdout | Out-Null

Write-Host "Building API image in ACR..."
& az acr build `
    --registry $AcrName `
    --image $ApiImageName `
    --file $Dockerfile `
    --no-logs `
    $RepoRoot `
    --only-show-errors | Out-Null

Write-Host "Deploying Container Apps infrastructure..."
$deployment = & az deployment group create `
    --resource-group $ResourceGroup `
    --template-file $BicepFile `
    --parameters `
        location=$Location `
        containerAppEnvironmentName=$ContainerAppEnvironmentName `
        qdrantAppName=$QdrantAppName `
        apiAppName=$ApiAppName `
        storageAccountName=$StorageAccountName `
        createStorageAccount=$CreateStorageAccount `
        fileShareName=$FileShareName `
        acrName=$AcrName `
        acrLoginServer=$AcrLoginServer `
        apiImageName=$ApiImageName `
        azureOpenAIEndpoint=$AzureOpenAIEndpoint `
        azureOpenAIKey=$AzureOpenAIKey `
    --query properties.outputs `
    --output json `
    --only-show-errors | ConvertFrom-Json

$apiUrl = $deployment.apiUrl.value

Write-Host ""
Write-Host "Deployment complete."
Write-Host "Public API URL: $apiUrl"
