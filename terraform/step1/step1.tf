provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "hatespeech_rg" {
  name     = "hatespeech_test"
  location = "East US"  # Azure Switzerland region
}

resource "azurerm_service_plan" "hatespeech_asp" {
  name                = "hatespeech_asp"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  os_type = "Linux"
  sku_name = "F1"
}

resource "azurerm_container_registry" "hatespeechctn" {
  name                = "hatespeechctn"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  sku                 = "Basic"
  admin_enabled       = true
}

# Azure PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "hatespeech_flexible_pg" {
  name                          = "hatespeech-flexible-pg"
  location                      = azurerm_resource_group.hatespeech_rg.location
  resource_group_name           = azurerm_resource_group.hatespeech_rg.name
  version                       = "12"  
  public_network_access_enabled = true
  administrator_login           = "psqladmin"
  administrator_password        = "H@Sh1CoR3!"
  zone                          = "1"

  storage_mb   = 32768
  storage_tier = "P30"

  sku_name = "B_Standard_B1ms"
}


resource "azurerm_postgresql_flexible_server_database" "hatespeechdb" {
  name        = "hatespeechdb"
  server_id   = azurerm_postgresql_flexible_server.hatespeech_flexible_pg.id  # Use .id instead of .name
  charset     = "UTF8"  
}

