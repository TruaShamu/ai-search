[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===== Required placeholders =====
$SubscriptionId = "<subscription-id>"
$ResourceGroup = "<resource-group>"
$Location = "<location>"
$ContainerAppEnvironmentResourceId = "/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.App/managedEnvironments/<aca-environment>"
$JobName = "embed-books-job"
$AcrName = "<acr-name>"
$AcrLoginServer = "<acr-name>.azurecr.io"
$ImageName = "embed-worker:latest"
$StorageAccountName = "<storage-account-name>"
$StorageContainerName = "embeddings"
$InputFile = "data\processed\books_augmented.jsonl"
$InputBlob = "inputs/books_augmented.jsonl"
$OutputPrefix = "outputs\embed-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$LocalOutputDir = "data\index"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Dockerfile = Join-Path $RepoRoot "Dockerfile.embed"
$BicepFile = Join-Path $RepoRoot "infra\aca-embed-job.bicep"
$ImageRef = "$AcrLoginServer/$ImageName"
$InputFilePath = Join-Path $RepoRoot $InputFile
$DownloadDir = Join-Path $RepoRoot $LocalOutputDir

function Invoke-AzCliJson {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $output = & az @Arguments --only-show-errors | Out-String
    if (-not $output.Trim()) {
        return $null
    }

    return $output | ConvertFrom-Json
}

Write-Host "Selecting subscription..."
& az account set --subscription $SubscriptionId --only-show-errors | Out-Null

Write-Host "Logging into ACR..."
& az acr login --name $AcrName --only-show-errors | Out-Null

Write-Host "Ensuring blob container exists..."
$StorageConnectionString = & az storage account show-connection-string `
    --name $StorageAccountName `
    --resource-group $ResourceGroup `
    --query connectionString `
    --output tsv `
    --only-show-errors

& az storage container create `
    --name $StorageContainerName `
    --connection-string $StorageConnectionString `
    --auth-mode key `
    --only-show-errors `
    --output none

Write-Host "Building Docker image $ImageRef ..."
docker build -f $Dockerfile -t $ImageRef $RepoRoot

Write-Host "Pushing Docker image..."
docker push $ImageRef

Write-Host "Uploading input file to blob storage..."
& az storage blob upload `
    --connection-string $StorageConnectionString `
    --container-name $StorageContainerName `
    --name $InputBlob `
    --file $InputFilePath `
    --overwrite true `
    --only-show-errors `
    --output none

Write-Host "Deploying or updating ACA job..."
& az deployment group create `
    --resource-group $ResourceGroup `
    --template-file $BicepFile `
    --parameters `
        location=$Location `
        jobName=$JobName `
        containerAppEnvironmentResourceId=$ContainerAppEnvironmentResourceId `
        acrName=$AcrName `
        acrLoginServer=$AcrLoginServer `
        imageName=$ImageName `
        storageAccountName=$StorageAccountName `
        storageContainerName=$StorageContainerName `
        inputBlob=$InputBlob `
        outputPrefix=$OutputPrefix `
    --only-show-errors `
    --output none

Write-Host "Capturing existing job executions..."
$existingExecutions = @(
    & az containerapp job execution list `
        --name $JobName `
        --resource-group $ResourceGroup `
        --query "[].name" `
        --output tsv `
        --only-show-errors
) | Where-Object { $_ }

Write-Host "Starting ACA job..."
$startResult = Invoke-AzCliJson -Arguments @(
    "containerapp", "job", "start",
    "--name", $JobName,
    "--resource-group", $ResourceGroup
)

$executionName = $null
if ($startResult -and $startResult.PSObject.Properties.Name -contains "name") {
    $executionName = $startResult.name
}

if (-not $executionName) {
    Write-Host "Waiting for execution record..."
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 10
        $currentExecutions = @(
            & az containerapp job execution list `
                --name $JobName `
                --resource-group $ResourceGroup `
                --query "[].name" `
                --output tsv `
                --only-show-errors
        ) | Where-Object { $_ }
        $executionName = Compare-Object -ReferenceObject $existingExecutions -DifferenceObject $currentExecutions |
            Where-Object { $_.SideIndicator -eq "=>" } |
            ForEach-Object { $_.InputObject } |
            Select-Object -First 1
        if ($executionName) {
            break
        }
    }
}

if (-not $executionName) {
    throw "Unable to determine ACA job execution name."
}

Write-Host "Watching execution $executionName ..."
while ($true) {
    Start-Sleep -Seconds 15
    $execution = Invoke-AzCliJson -Arguments @(
        "containerapp", "job", "execution", "show",
        "--name", $JobName,
        "--resource-group", $ResourceGroup,
        "--job-execution-name", $executionName
    )

    $status = $execution.properties.status
    $startTime = $execution.properties.startTime
    Write-Host ("[{0}] status={1}" -f $startTime, $status)

    if ($status -in @("Succeeded", "Failed", "Canceled")) {
        if ($status -ne "Succeeded") {
            throw "ACA job execution ended with status '$status'."
        }
        break
    }
}

Write-Host "Downloading generated index files..."
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

& az storage blob download `
    --connection-string $StorageConnectionString `
    --container-name $StorageContainerName `
    --name "$OutputPrefix/faiss.index" `
    --file (Join-Path $DownloadDir "faiss.index") `
    --overwrite true `
    --only-show-errors `
    --output none

& az storage blob download `
    --connection-string $StorageConnectionString `
    --container-name $StorageContainerName `
    --name "$OutputPrefix/metadata.jsonl" `
    --file (Join-Path $DownloadDir "metadata.jsonl") `
    --overwrite true `
    --only-show-errors `
    --output none

Write-Host "Done. Outputs downloaded to $DownloadDir"
