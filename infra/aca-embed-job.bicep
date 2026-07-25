@description('Location for the ACA job.')
param location string = resourceGroup().location

@description('Name of the ACA job to create.')
param jobName string = 'embed-books-job'

@description('Resource ID of the existing Azure Container Apps managed environment.')
param containerAppEnvironmentResourceId string

@description('Azure Container Registry name.')
param acrName string

@description('Azure Container Registry login server, for example myregistry.azurecr.io.')
param acrLoginServer string

@description('Container image name including tag, for example embed-worker:latest.')
param imageName string

@description('Azure Storage account name used for job input/output blobs.')
param storageAccountName string

@description('Blob container name used for the job input/output blobs.')
param storageContainerName string

@description('Input blob path inside the container.')
param inputBlob string = 'inputs/books_augmented.jsonl'

@description('Blob prefix to upload outputs under.')
param outputPrefix string = 'outputs/embed'

@description('Optional ACA workload profile name. Set this when 4 CPU / 8 Gi requires a dedicated workload profile.')
param workloadProfileName string = ''

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

var storageConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
var imageReference = '${acrLoginServer}/${imageName}'
var acrPullRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var jobProperties = union({
  configuration: {
    triggerType: 'Manual'
    replicaTimeout: 1800
    replicaRetryLimit: 0
    manualTriggerConfig: {
      parallelism: 1
      replicaCompletionCount: 1
    }
    registries: [
      {
        server: acrLoginServer
        identity: 'system'
      }
    ]
    secrets: [
      {
        name: 'azure-storage-connection-string'
        value: storageConnectionString
      }
    ]
  }
  environmentId: containerAppEnvironmentResourceId
  template: {
    containers: [
      {
        name: 'embed-worker'
        image: imageReference
        command: [
          'python'
        ]
        args: [
          'scripts/embed_entrypoint.py'
        ]
        env: [
          {
            name: 'AZURE_STORAGE_CONNECTION_STRING'
            secretRef: 'azure-storage-connection-string'
          }
          {
            name: 'STORAGE_CONTAINER'
            value: storageContainerName
          }
          {
            name: 'INPUT_BLOB'
            value: inputBlob
          }
          {
            name: 'OUTPUT_PREFIX'
            value: outputPrefix
          }
          {
            name: 'TIER_FILTER'
            value: '1'
          }
          {
            name: 'EMBED_DIM'
            value: '256'
          }
        ]
        resources: {
          cpu: 4
          memory: '8Gi'
        }
      }
    ]
  }
}, empty(workloadProfileName) ? {} : {
  workloadProfileName: workloadProfileName
})

resource embedJob 'Microsoft.App/jobs@2025-01-01' = {
  name: jobName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: jobProperties
}

resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, embedJob.name, 'AcrPull')
  scope: acr
  properties: {
    principalId: embedJob.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPullRoleDefinitionId
  }
}

output jobResourceId string = embedJob.id
output resolvedImage string = imageReference
