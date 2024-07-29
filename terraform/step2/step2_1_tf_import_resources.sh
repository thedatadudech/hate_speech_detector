az login

# Subscription ID abrufen
SUBSCRIPTION_ID=$(az account show --query 'id' -o tsv)

# Subscription ID anzeigen
echo "Azure Subscription ID: $SUBSCRIPTION_ID"

# Ihre Terraform-Importbefehle hier einfügen
terraform import azurerm_resource_group.hatespeech_rg /subscriptions/$SUBSCRIPTION_ID/resourceGroups/hatespeech_test;
terraform import azurerm_service_plan.hatespeech_asp "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/hatespeech_test/providers/Microsoft.Web/serverFarms/hatespeech_asp"
#terraform import azurerm_storage_account.hatespeechstorage /subscriptions/$SUBSCRIPTION_ID/resourceGroups/hatespeech_test/providers/Microsoft.Storage/storageAccounts/hatespeechstor
#terraform import azurerm_storage_share.fileshare https://hatespeechstor.file.core.windows.net/hatespeechshare
#terraform import azurerm_container_registry.hatespeechctn /subscriptions/$SUBSCRIPTION_ID/resourceGroups/hatespeech_test/providers/Microsoft.ContainerRegistry/registries/hatespeechctn
