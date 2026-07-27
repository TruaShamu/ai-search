@description('Location for all resources.')
param location string = 'eastus'

@description('Azure Container Apps managed environment name.')
param containerAppEnvironmentName string = 'booksearch-aca-env'

@description('Qdrant Container App name.')
param qdrantAppName string = 'qdrant'

@description('FastAPI Container App name.')
param apiAppName string = 'booksearch-api'

@description('Azure Storage account name used for the Qdrant Azure Files share.')
param storageAccountName string = 'booksearchblobs'

@description('Set to true to create the storage account in this deployment. Leave false to use an existing account.')
param createStorageAccount bool = false

@description('Azure Files share name used for Qdrant persistence.')
param fileShareName string = 'qdrant-data'

@description('Azure Container Registry name.')
param acrName string

@description('Azure Container Registry login server, for example myregistry.azurecr.io.')
param acrLoginServer string

@description('API image name including tag, for example booksearch-api:latest.')
param apiImageName string

@description('Azure OpenAI endpoint for the API.')
param azureOpenAIEndpoint string

@secure()
@description('Azure OpenAI API key for the API.')
param azureOpenAIKey string

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = if (createStorageAccount) {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowSharedKeyAccess: true
    minimumTlsVersion: 'TLS1_2'
  }
}

resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  name: '${storageAccountName}/default/${fileShareName}'
  dependsOn: createStorageAccount ? [
    storageAccount
  ] : []
}

resource managedEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: containerAppEnvironmentName
  location: location
  properties: {}
}

var storageAccountId = resourceId('Microsoft.Storage/storageAccounts', storageAccountName)
var storageAccountKey = listKeys(storageAccountId, '2023-05-01').keys[0].value
var acrCredentials = acr.listCredentials()
var apiImage = '${acrLoginServer}/${apiImageName}'

resource qdrantStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  name: 'qdrant-storage'
  parent: managedEnvironment
  properties: {
    azureFile: {
      accountName: storageAccountName
      accountKey: storageAccountKey
      accessMode: 'ReadWrite'
      shareName: fileShareName
    }
  }
  dependsOn: [
    fileShare
  ]
}

@description('Whether Qdrant ingress is external (set true for initial data migration, false for prod).')
param qdrantExternalIngress bool = true

resource qdrantApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: qdrantAppName
  location: location
  properties: {
    managedEnvironmentId: managedEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: qdrantExternalIngress
        targetPort: 6333
        transport: 'http'
      }
    }
    template: {
      containers: [
        {
          name: 'qdrant'
          image: 'qdrant/qdrant:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          volumeMounts: [
            {
              volumeName: 'qdrant-data'
              mountPath: '/qdrant/storage'
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'qdrant-data'
          storageType: 'AzureFile'
          storageName: qdrantStorage.name
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: apiAppName
  location: location
  properties: {
    managedEnvironmentId: managedEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
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
          name: 'azure-openai-key'
          value: azureOpenAIKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'booksearch-api'
          image: apiImage
          env: [
            {
              name: 'QDRANT_URL'
              value: 'http://qdrant:80'
            }
            {
              name: 'QDRANT_COLLECTION'
              value: 'books'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'AZURE_OPENAI_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'PYTHONPATH'
              value: '/app'
            }
          ]
          resources: {
            // 2 vCPU: the cross-encoder scores 25 candidates at up to 512
            // tokens each. On 1 vCPU that exceeded the ingress timeout once
            // passage truncation was removed. ACA Consumption requires
            // memory to be 2x cpu in Gi.
            cpu: 2
            memory: '4Gi'
          }
          probes: [
            {
              // Keeps ingress from routing to a replica that is still loading
              // models. /ready returns 503 until warmup completes.
              // 40 x 10s allows ~6.5min for a cold start before giving up.
              type: 'Readiness'
              httpGet: {
                path: '/ready'
                port: 8000
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 40
            }
            {
              // Liveness hits /health, which answers 200 as soon as the
              // process is up. It must not depend on warmup, or a slow model
              // load would restart the container in a loop.
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 20
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 5
            }
          ]
        }
      ]
      scale: {
        // Warm at all times. At minReplicas 0 the first request after an idle
        // period paid a ~20s model load, and scale-up sent traffic to cold
        // replicas faster than they could warm.
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
  dependsOn: [
    qdrantApp
  ]
}

output apiUrl string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output qdrantInternalUrl string = 'http://${qdrantApp.name}'
output qdrantExternalUrl string = qdrantExternalIngress ? 'https://${qdrantApp.properties.configuration.ingress.fqdn}' : 'internal-only'
output containerAppEnvironmentResourceId string = managedEnvironment.id
