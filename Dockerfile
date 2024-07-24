FROM mageai/mageai:alpha


ARG PROJECT_NAME=${PROJECT_NAME}
ARG MAGE_CODE_PATH=${MAGE_CODE_PATH}
ARG USER_CODE_PATH=${MAGE_CODE_PATH}/${PROJECT_NAME}

WORKDIR ${USER_CODE_PATH}



COPY requirements.txt .

ENV USER_CODE_PATH=${USER_CODE_PATH}

# Install custom Python libraries and dependencies for your project.
RUN pip3 install -r ${USER_CODE_PATH}/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:${MAGE_CODE_PATH}/${PROJECT_NAME}"

# Installing necessary utilities and Terraform.
# Uncomment the following lines if you want to use Terraform in Docker.
# RUN apt-get update && \
#   apt-get install -y wget unzip && \
#   wget https://releases.hashicorp.com/terraform/1.8.3/terraform_1.8.3_linux_amd64.zip && \
#   unzip terraform_1.8.3_linux_amd64.zip -d /usr/local/bin/ && \
#   rm terraform_1.8.3_linux_amd64.zip

CMD ["/bin/sh", "-c", "/app/run_app.sh"]
