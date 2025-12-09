
# Helpful Tips For Building

## Build locally with docker

Build a new docker image and spin up container in local environment.

```bash
docker compose up
```

## Build and push docker images to GCP Artifact Registry

Create new GCP artificat repository for docker.

```bash
gcloud artifacts repositories create <repo-name> --repository-format=docker \
    --location=<location>--description="<descripiton>" \
    --project=<project-id>
```

Configure docker for GCP artifact image storage (standardizing on us-east4, but adjust for your closest location.)

```bash
gcloud auth configure-docker us-east4-docker.pkg.dev
```

Build and push new image on GCP Artifact Registry via macOS (on windows no need to specify `--platform linux/amd64`)

```bash
docker build -t us-east4-docker.pkg.dev/<gcp-project-id>/<repo-name>/<folder>:<version> . --platform linux/amd64
docker push us-east4-docker.pkg.dev/<gcp-project-id>/<repo-name>/<folder>:<version>
```

## Creating a new API gateway

Create a new API gateway

```bash
gcloud api-gateway apis create <api-name> --project=<project-id> --display-name="<display-name>"
```

```bash
gcloud api-gateway api-configs create <config-name> \
    --api=<api-name> \
    --openapi-spec=api_config.yaml \
    --project=<project-id> \
    --backend-auth-service-account=<your-api-service-api-gateway-email> \
    --display-name="<config-display-name>"
```

```bash
gcloud api-gateway gateways create <gateway-name>\
    --api=<api-name> \
    --api-config=<config-name> \
    --location=<location> \
    --project=<project-id>
```

```bash
gcloud services enable MANAGED_SERVICE_NAME
```

Once you have created one to update your API gateway with a new config simply follow the same instructions to create a config and then update the gateway using the following:

```bash
gcloud api-gateway gateways update <gateway-name>\
    --api=<api-name> \
    --api-config=<config-name> \
    --location=<location> \
    --project=<project-id>
```