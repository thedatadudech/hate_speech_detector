provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "hatespeech_rg" {
  name     = "hatespeech_test"
  location = "East US"  # Azure
}

resource "azurerm_service_plan" "hatespeech_asp" {
  name                = "hatespeech_asp"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  os_type = "Linux"
  sku_name = "F1"
}

resource "azurerm_storage_account" "hatespeechstorage" {
  name                     = "hatespeech-storage"
  resource_group_name      = azurerm_resource_group.hatespeech_rg.name
  location                 = azurerm_resource_group.hatespeech_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_share" "fileshare" {
  name                 = "hatespeechshare"
  storage_account_name = azurerm_storage_account.hatespeechstorage.name
  quota                = 50
}


resource "azurerm_linux_web_app" "hatespeech-mage" {
  name                = "hatespeech-mage"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  service_plan_id = azurerm_service_plan.hatespeech_asp.id


   site_config {
    always_on = false
    application_stack {
      docker_image_name   = "mage_initial:latest"
      docker_registry_url = "https://hatespeechctn.azurecr.io"
    }

   }

   app_settings = {
        USER_CODE_PATH="/home/src/hate_speech_detector"
        MAGE_DATABASE_CONNECTION_URL="postgresql+psycopg2://POSTGRES_USER:POSTGRES_PASSWORD@POSTGRES_HOST:POSTGRES_PORT/POSTGRES_DB"
        SMTP_EMAIL="SMTP_EMAIL"
        SMTP_PASSWORD="SMTP_PASSWORD" 
    }
   storage_account {
    type        = "AzureFiles"
    account_name = azurerm_storage_account.hatespeechstorage.name
    name        = azurerm_storage_account.hatespeechstorage.name
    access_key  = azurerm_storage_account.hatespeechstorage.primary_access_key
    mount_path  = "/data"
    share_name  = azurerm_storage_share.fileshare.name
  }
}

resource "azurerm_linux_web_app" "hatespeech-mlflow" {
  name                = "hatespeech-mlflow"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  service_plan_id = azurerm_service_plan.hatespeech_asp.id


   site_config {
    always_on = false
    application_stack {
      docker_image_name   = "mlflow_initial:latest"
      docker_registry_url = "https://hatespeechctn.azurecr.io"
    }

   }

   app_settings = {
        MLFLOW_CONNECTION_URI="TODO:"
        MAGE_DATABASE_CONNECTION_URL="TODO:"
        SMTP_EMAIL="TODO:"
        SMTP_PASSWORD="TODO:" 
    } 
}

resource "azurerm_linux_web_app" "hatespeech-fastapi" {
  name                = "hatespeech-mlflow"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  service_plan_id = azurerm_service_plan.hatespeech_asp.id


   site_config {
    always_on = false
    application_stack {
      docker_image_name   = "fastapi_initial:latest"
      docker_registry_url = "https://hatespeechctn.azurecr.io"
    }

   }
}

resource "azurerm_linux_web_app" "hatespeech-gradio" {
  name                = "hatespeech-gradio"
  location            = azurerm_resource_group.hatespeech_rg.location
  resource_group_name = azurerm_resource_group.hatespeech_rg.name
  service_plan_id = azurerm_service_plan.hatespeech_asp.id


   site_config {
    always_on = false
    application_stack {
      docker_image_name   = "gradio_initial:latest"
      docker_registry_url = "https://hatespeechctn.azurecr.io"
    }
   }
}