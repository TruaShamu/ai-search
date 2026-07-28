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

@description('Where workers write results. "blob" writes dense shards for offline assembly; "qdrant" upserts directly.')
@allowed([
  'blob'
  'qdrant'
])
param embedOutputMode string = 'blob'

@description('Blob prefix for dense shards when embedOutputMode is "blob".')
param shardPrefix string = 'shards'

@description('Max concurrent job executions. Each execution embeds one slice.')
param maxExecutions int = 30

@description('vCPU per replica. The Consumption plan tops out at 2.0.')
param workerCpu string = '2.0'

@description('Memory per replica. Must pair with workerCpu at a 1:2 ratio.')
param workerMemory string = '4.0Gi'

@description('Seconds a single execution may run before ACA kills it.')
param replicaTimeoutSeconds int = 10800

@description('Sentence-transformer encode batch size. Sized against workerMemory, not throughput.')
param embedBatchSize int = 32

@description('Token cap per document. Measured max for this corpus is ~600 tokens, so 1024 is lossless.')
param embedMaxSeqLen int = 1024

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
      replicaTimeout: replicaTimeoutSeconds
      replicaRetryLimit: 2
      triggerType: 'Event'
      eventTriggerConfig: {
        parallelism: 1        // One replica per execution; concurrency comes from maxExecutions
        replicaCompletionCount: 1
        scale: {
          minExecutions: 0    // Scale to zero when queue is empty
          maxExecutions: maxExecutions
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
              name: 'EMBED_OUTPUT_MODE'
              value: embedOutputMode
            }
            {
              name: 'SHARD_PREFIX'
              value: shardPrefix
            }
            {
              name: 'EMBED_BATCH_SIZE'
              value: string(embedBatchSize)
            }
            {
              name: 'EMBED_MAX_SEQ_LEN'
              value: string(embedMaxSeqLen)
            }
            {
              name: 'PYTHONPATH'
              value: '/app'
            }
          ]
          // The Consumption plan only accepts fixed cpu/memory pairs at a 1:2
          // ratio and tops out at 2.0 vCPU / 4.0Gi. Asking for 4 vCPU is
          // rejected at deploy time with ContainerAppInvalidResourceTotal, so
          // throughput comes from maxExecutions rather than bigger replicas.
          resources: {
            cpu: json(workerCpu)
            memory: workerMemory
          }
        }
      ]
    }
  }
}

output jobName string = embedJob.name
output jobId string = embedJob.id
