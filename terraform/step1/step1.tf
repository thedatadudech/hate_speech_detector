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

resource "azurerm_postgresql_server" "hatespeech_pg" {
  name                = "hatespeech-pg"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name

  sku_name            = "B_Gen5_1"
  storage_mb          = 5120
  backup_retention_days = 7
  administrator_login = "psqladminun"
  administrator_login_password = "H@tespeechP@ssw0rd!"

  version             = "11"
  ssl_enforcement_enabled     = true
}

resource "azurerm_postgresql_database" "hatespeechdb" {
  name                = "hatespeechdb"
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  server_name         = azurerm_postgresql_server.hatespeech_pg.name
  charset             = "UTF8"
  collation           = "English_United States.1252"
}
