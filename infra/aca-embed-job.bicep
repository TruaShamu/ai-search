// Event-driven embedding worker — ACA Job triggered by Azure Storage Queue messages.
// Scales from 0 when messages arrive, processes embedding batches, upserts to Qdrant.

@description('Location for all resources.')
param location string = resourceGroup().location

@description('Existing ACA managed environment resource ID.')
param containerAppEnvironmentId string

@description('ACR login server (e.g., booksearchacr.azurecr.io).')
param acrLoginServer string

@description('Embed worker image name with tag.')
param embedImageName string = 'embed-worker:latest'

@description('Storage account name for queue + blob access.')
param storageAccountName string = 'booksearchblobs'

@description('Queue name for embed tasks.')
param queueName string = 'embed-tasks'

@description('Qdrant internal URL within the ACA environment.')
param qdrantUrl string = 'http://qdrant'

@description('Qdrant collection name.')
param qdrantCollection string = 'books'

@description('Embedding dimension (Matryoshka).')
param embedDim int = 256

@description('ACR name for credential lookup.')
param acrName string

// Existing resources
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

var storageConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
var acrCredentials = acr.listCredentials()
var embedImage = '${acrLoginServer}/${embedImageName}'

resource embedJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'embed-worker'
  location: location
  properties: {
    environmentId: containerAppEnvironmentId
    configuration: {
      replicaTimeout: 3600  // 1 hour max per execution
      replicaRetryLimit: 2
      triggerType: 'Event'
      eventTriggerConfig: {
        parallelism: 1        // One worker at a time (memory-bound)
        replicaCompletionCount: 1
        scale: {
          minExecutions: 0    // Scale to zero when queue is empty
          maxExecutions: 3    // Max concurrent batches
          pollingInterval: 30
          rules: [
            {
              name: 'queue-trigger'
              type: 'azure-queue'
              metadata: {
                queueName: queueName
                queueLength: '1'
                accountName: storageAccountName
              }
              auth: [
                {
                  secretRef: 'storage-connection'
                  triggerParameter: 'connection'
                }
              ]
            }
          ]
        }
      }
      registries: [
        {
          server: acrLoginServer
          username: acrCredentials.username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acrCredentials.passwords[0].value
        }
        {
          name: 'storage-connection'
          value: storageConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'embed-worker'
          image: embedImage
          command: ['python', '-m', 'scripts.embed_worker', '--loop']
          env: [
            {
              name: 'AZURE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection'
            }
            {
              name: 'QUEUE_NAME'
              value: queueName
            }
            {
              name: 'STORAGE_CONTAINER'
              value: 'embeddings'
            }
            {
              name: 'QDRANT_URL'
              value: qdrantUrl
            }
            {
              name: 'QDRANT_COLLECTION'
              value: qdrantCollection
            }
            {
              name: 'EMBED_DIM'
              value: string(embedDim)
            }
            {
              name: 'PYTHONPATH'
              value: '/app'
            }
          ]
          resources: {
            cpu: 4
            memory: '16Gi'
          }
        }
      ]
    }
  }
}

output jobName string = embedJob.name
output jobId string = embedJob.id
