## Step-by-Step Deployment Guide

### Step 1: Initialize and Apply Initial Terraform Configuration

1. **Navigate to the `step1` directory:**

   ```sh
   cd /terraform/step1
   ```

2. **Initialize Terraform:**

   ```sh
   terraform init
   ```

3. **Plan the Terraform deployment:**

   ```sh
   terraform plan -out step1
   ```

4. **Apply the Terraform plan:**

   ```sh
   terraform apply "step1"
   ```

### Step 2: Load Docker Images into the Container Registry

1. **Navigate to the project root directory:**

   ```sh
   cd /terraform
   ```

2. **Run the shell script to create and push Docker images to the container registry:**

   ```sh
   chmod +x step1_1_create_push_image2registry.sh
   ./step1_1_create_push_image2registry.sh
   ```

### Step 3: Initialize and Import Existing Resources

1. **Navigate to the `step2` directory:**

   ```sh
   cd /terraform/step2
   ```

2. **Initialize Terraform:**

   ```sh
   terraform init
   ```

3. **Import existing resources using the provided script:**

   ```sh
   chmod +x ../step2_1_tf_import_resources.sh
   ../step2_1_tf_import_resources.sh
   ```

### Step 4: Plan and Apply Additional Terraform Configuration

1. **Plan the Terraform deployment:**

   ```sh
   terraform plan -out step2
   ```

2. **Apply the Terraform plan:**

   ```sh
   terraform apply "step2"
   ```

## Detailed Instructions for Each Step

### step1.tf

Ensure that the `step1.tf` file in the `step1` directory contains your initial Terraform configuration to set up the basic infrastructure.

### step1_1_create_push_image2registry.sh

Ensure that this script creates Docker images and pushes them to the Azure Container Registry created in step 1. Here is a sample structure for the script:

```sh
#!/bin/bash

# Variables
ACR_NAME="hatespeechctn"
IMAGE_NAME="your_image_name"
TAG="latest"

# Login to Azure
az login

# Login to Azure Container Registry
az acr login --name $ACR_NAME

# Build the Docker image
docker build -t $IMAGE_NAME:$TAG .

# Tag the Docker image
docker tag $IMAGE_NAME:$TAG $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG

# Push the Docker image to the Azure Container Registry
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG
```

### step2_1_tf_import_resources.sh

Ensure that this script imports existing resources into the Terraform state. Here is a sample structure for the script:

```sh
#!/bin/bash

# Import the existing resource group
terraform import azurerm_resource_group.hatespeech_rg /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/hatespeech_test

# Import other resources as needed
# Example:
# terraform import azurerm_storage_account.hatespeechstorage /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/hatespeech_test/providers/Microsoft.Storage/storageAccounts/hatespeechstorage
```

### step2.tf

Ensure that the `step2.tf` file in the `step2` directory contains the additional Terraform configuration to set up the remaining infrastructure and services.

---

By following these steps, you will successfully deploy and configure your hate speech detection system on Azure using Terraform. Modify the configurations and scripts as needed to fit your specific requirements.
