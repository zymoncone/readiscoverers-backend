# readiscover.app

[![App Status](https://img.shields.io/badge/Live-readiscover.app-6366f1?style=for-the-badge)](https://readiscover.app)
[![Pylint](https://github.com/zymoncone/readiscoverers-backend/actions/workflows/pylint.yml/badge.svg)](https://github.com/zymoncone/readiscoverers-backend/actions/workflows/pylint.yml)

## 🌐 Visit the Live Website

Check out the live app at **[readiscover.app](https://readiscover.app)**!

> **Note:** We're currently in active development. To access the app, please email **szymons@umich.edu** for the password.

## 🚀 Running Locally

### Prerequisites

1. **Install Docker**
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. **Install Google Cloud CLI**
   - Download and install the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
   - After installation, initialize gcloud:
     ```bash
     gcloud init
     ```
   - Authenticate with your Google account:
     ```bash
     gcloud auth login
     gcloud auth application-default login
     ```

### Build locally with docker

Build a new docker image and spin up container in local environment.

```bash
docker compose up --build
```