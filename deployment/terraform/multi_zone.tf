# Azure Provider Configuration
provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "ds_rg" {
  name     = "deep-sentinel-aks-rg"
  location = "East US"
}

# Azure Kubernetes Service Cluster
resource "azurerm_kubernetes_cluster" "ds_aks" {
  name                = "deep-sentinel-cluster"
  location            = azurerm_resource_group.ds_rg.location
  resource_group_name = azurerm_resource_group.ds_rg.name
  dns_prefix          = "deepsentinel"

  default_node_pool {
    name       = "default"
    node_count = 3
    # Standard_D4s_v3 has precisely 4 vCPUs and 16 GiB RAM, meeting the 16GB RAM hardware constraint
    vm_size    = "Standard_D4s_v3"
    
    # Ensure sufficient local storage for the 200GB volume request
    os_disk_size_gb = 250 
    
    # Enable multiple availability zones for SLA resilience
    zones = ["1", "2", "3"]
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "kubenet"
    load_balancer_sku = "standard"
  }

  tags = {
    Environment = "Production"
    Project     = "DeepSentinel"
  }
}

output "kube_config" {
  value     = azurerm_kubernetes_cluster.ds_aks.kube_config_raw
  sensitive = true
}
