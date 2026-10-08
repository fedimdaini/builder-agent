# Verify findings: mlops-zoomcamp-project

Source: git HEAD (e157717). Generated 2026-10-08T20:48:12+00:00. Static rules only; nothing was run or changed.

**8 error(s), 5 warning(s).** Patch: `fixes.patch`, applies cleanly to git HEAD (e157717) (git apply --check on a clean export). The Builder never applies it.

| # | Severity | File | Line | Rule | Fault | Evidence |
|---|---|---|---|---|---|---|
| 1 | error | `docker-compose.yaml` | 263 | floating-image-tag | fault-002 | `image: localstack/localstack:stable` |
| 2 | error | `docker-compose.yaml` | 309 | floating-image-tag | fault-002 | `image: postgres:latest` |
| 3 | error | `docker-compose.yaml` | 309 | postgres18-mount-path | fault-003 | `image: postgres:latest  +  line 315: postgres-nyc-volume:/var/lib/postgresql/...` |
| 4 | error | `docker-compose.yaml` | 359 | floating-image-tag | fault-002 | `image: postgres:latest` |
| 5 | error | `docker-compose.yaml` | 359 | postgres18-mount-path | fault-003 | `image: postgres:latest  +  line 365: postgres-mlflow-volume:/var/lib/postgres...` |
| 6 | error | `docker-compose.yaml` | 394 | floating-image-tag | fault-002 | `image: postgres:latest` |
| 7 | error | `docker-compose.yaml` | 394 | postgres18-mount-path | fault-003 | `image: postgres:latest  +  line 400: postgres-grafana-volume:/var/lib/postgre...` |
| 8 | error | `mlflow/Dockerfile` | 35 | mlflow-local-artifact-root | fault-005 | `CMD ["sh", "-c", "mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI...` |
| 9 | warning | `.gitattributes` | - | crlf-shell-script | fault-004 | `2 shell script(s) (flask/test_requests.sh, localstack_s3_client/init.sh); no ...` |
| 10 | warning | `docker-compose.yaml` | 329 | service-without-healthcheck | fault-006 | `mlflow: publishes ${MLFLOW_PORT}:5000 and has no healthcheck` |
| 11 | warning | `docker-compose.yaml` | 374 | service-without-healthcheck | fault-006 | `flask-app: publishes 8000:8000 and has no healthcheck` |
| 12 | warning | `docker-compose.yaml` | 408 | service-without-healthcheck | fault-006 | `grafana: publishes ${GRAFANA_PORT}:3000 and has no healthcheck` |
| 13 | warning | `docker-compose.yaml` | 409 | floating-image-tag | fault-002 | `image: grafana/grafana` |

## 1. floating-image-tag (fault-002, error): `docker-compose.yaml` line 263

localstack/localstack:stable has a floating tag (stable): it changes under you (fault-002: localstack:stable started requiring an account)

Evidence: `image: localstack/localstack:stable`

Fix: pin it to localstack/localstack:4.12 (known to work, fault-002)

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -260,7 +260,7 @@
   # LOCALSTACK 
   # ---------------------------------------------------------------------------
   localstack:
-    image: localstack/localstack:stable
+    image: localstack/localstack:4.12
     container_name: localstack
     # env_file:
     #   - .env
```

## 2. floating-image-tag (fault-002, error): `docker-compose.yaml` line 309

postgres:latest has a floating tag (latest): it changes under you (fault-002: localstack:stable started requiring an account)

Evidence: `image: postgres:latest`

Fix: pin it to postgres:16 (known to work, fault-003)

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -306,7 +306,7 @@
   # ---------------------------------------------------------------------------
 
   postgresnyc:
-    image: postgres:latest
+    image: postgres:16
     environment:
       - POSTGRES_USER=${NYC_POSTGRES_USERNAME:-nycpostgres}
       - POSTGRES_PASSWORD=${NYC_POSTGRES_PASSWORD:-nycpostgres}
```

## 3. postgres18-mount-path (fault-003, error): `docker-compose.yaml` line 309

postgres:latest can resolve to PostgreSQL 18+, which exits when a volume is mounted at /var/lib/postgresql/data (line 315)

Evidence: `image: postgres:latest  +  line 315: postgres-nyc-volume:/var/lib/postgresql/data`

Fix: pin it to postgres:16 (keeps the data path), or move the mount to /var/lib/postgresql before going to 18

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -306,7 +306,7 @@
   # ---------------------------------------------------------------------------
 
   postgresnyc:
-    image: postgres:latest
+    image: postgres:16
     environment:
       - POSTGRES_USER=${NYC_POSTGRES_USERNAME:-nycpostgres}
       - POSTGRES_PASSWORD=${NYC_POSTGRES_PASSWORD:-nycpostgres}
```

## 4. floating-image-tag (fault-002, error): `docker-compose.yaml` line 359

postgres:latest has a floating tag (latest): it changes under you (fault-002: localstack:stable started requiring an account)

Evidence: `image: postgres:latest`

Fix: pin it to postgres:16 (known to work, fault-003)

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -356,7 +356,7 @@
     #   - app-network
 
   postgresmlflow:
-    image: postgres:latest
+    image: postgres:16
     environment:
       - POSTGRES_USER=$MLFLOW_POSTGRES_USER
       - POSTGRES_PASSWORD=$MLFLOW_POSTGRES_PASS
```

## 5. postgres18-mount-path (fault-003, error): `docker-compose.yaml` line 359

postgres:latest can resolve to PostgreSQL 18+, which exits when a volume is mounted at /var/lib/postgresql/data (line 365)

Evidence: `image: postgres:latest  +  line 365: postgres-mlflow-volume:/var/lib/postgresql/data`

Fix: pin it to postgres:16 (keeps the data path), or move the mount to /var/lib/postgresql before going to 18

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -356,7 +356,7 @@
     #   - app-network
 
   postgresmlflow:
-    image: postgres:latest
+    image: postgres:16
     environment:
       - POSTGRES_USER=$MLFLOW_POSTGRES_USER
       - POSTGRES_PASSWORD=$MLFLOW_POSTGRES_PASS
```

## 6. floating-image-tag (fault-002, error): `docker-compose.yaml` line 394

postgres:latest has a floating tag (latest): it changes under you (fault-002: localstack:stable started requiring an account)

Evidence: `image: postgres:latest`

Fix: pin it to postgres:16 (known to work, fault-003)

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -391,7 +391,7 @@
   # ---------------------------------------------------------------------------
 
   postgresgrafana:
-    image: postgres:latest
+    image: postgres:16
     environment:
       - POSTGRES_USER=grafana
       - POSTGRES_PASSWORD=grafana
```

## 7. postgres18-mount-path (fault-003, error): `docker-compose.yaml` line 394

postgres:latest can resolve to PostgreSQL 18+, which exits when a volume is mounted at /var/lib/postgresql/data (line 400)

Evidence: `image: postgres:latest  +  line 400: postgres-grafana-volume:/var/lib/postgresql/data`

Fix: pin it to postgres:16 (keeps the data path), or move the mount to /var/lib/postgresql before going to 18

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -391,7 +391,7 @@
   # ---------------------------------------------------------------------------
 
   postgresgrafana:
-    image: postgres:latest
+    image: postgres:16
     environment:
       - POSTGRES_USER=grafana
       - POSTGRES_PASSWORD=grafana
```

## 8. mlflow-local-artifact-root (fault-005, error): `mlflow/Dockerfile` line 35

mlflow server uses a plain local path as --default-artifact-root ($MLFLOW_DEFAULT_ARTIFACT_ROOT = /tmp/artifacts, set by service mlflow in docker-compose.yaml): every client writes artifacts to its own disk, so other containers can't load the models (fault-005). --serve-artifacts alone doesn't help while the root is a local path

Evidence: `CMD ["sh", "-c", "mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT --host 0.0.0.0 --port 5000"]`

Fix: let the server store and serve the artifacts: --serve-artifacts --artifacts-destination <path or s3://...> instead of --default-artifact-root; new experiments then get mlflow-artifacts:/ URIs (existing experiments keep their old location)

```diff
diff --git a/mlflow/Dockerfile b/mlflow/Dockerfile
--- a/mlflow/Dockerfile
+++ b/mlflow/Dockerfile
@@ -32,4 +32,4 @@
 #     "--port", "5000" \
 # ]
 
-CMD ["sh", "-c", "mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT --host 0.0.0.0 --port 5000"]
+CMD ["sh", "-c", "mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --serve-artifacts --artifacts-destination $MLFLOW_DEFAULT_ARTIFACT_ROOT --host 0.0.0.0 --port 5000"]
```

## 9. crlf-shell-script (fault-004, warning): `.gitattributes`

nothing keeps the shell scripts LF: a Windows checkout with core.autocrlf=true turns them into CRLF and they fail in Linux containers (fault-004)

Evidence: `2 shell script(s) (flask/test_requests.sh, localstack_s3_client/init.sh); no .gitattributes rule with eol=lf for *.sh`

Fix: create .gitattributes with `*.sh text eol=lf` so every checkout keeps them LF, whatever each teammate's git settings

```diff
diff --git a/.gitattributes b/.gitattributes
new file mode 100644
--- /dev/null
+++ b/.gitattributes
@@ -0,0 +1,2 @@
+# Shell scripts run in Linux containers: keep LF on every checkout (fault-004)
+*.sh text eol=lf
```

## 10. service-without-healthcheck (fault-006, warning): `docker-compose.yaml` line 329

service mlflow has ports but no healthcheck: nothing can wait for it (depends_on: service_healthy) and a broken service still shows as Up (fault-006)

Evidence: `mlflow: publishes ${MLFLOW_PORT}:5000 and has no healthcheck`

Fix: add a healthcheck on http://localhost:5000/health using python3, which its Dockerfile installs

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -332,6 +332,11 @@
       dockerfile: Dockerfile
     ports:
       - ${MLFLOW_PORT}:5000
+    healthcheck:
+      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=5)"]
+      interval: 30s
+      timeout: 10s
+      retries: 5
     environment:
       # - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://${MLFLOW_BUCKET}/${PROJECT_NAME}/artifacts
       - MLFLOW_DEFAULT_ARTIFACT_ROOT=/tmp/artifacts
```

## 11. service-without-healthcheck (fault-006, warning): `docker-compose.yaml` line 374

service flask-app has ports but no healthcheck: nothing can wait for it (depends_on: service_healthy) and a broken service still shows as Up (fault-006)

Evidence: `flask-app: publishes 8000:8000 and has no healthcheck`

Fix: add a healthcheck on http://localhost:8000/ using the healthcheck another compose file already uses for this image

```diff
diff --git a/docker-compose.yaml b/docker-compose.yaml
--- a/docker-compose.yaml
+++ b/docker-compose.yaml
@@ -377,6 +377,12 @@
       dockerfile: Dockerfile
     ports:
       - "8000:8000"
+    healthcheck:
+      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
+      interval: 30s
+      timeout: 10s
+      retries: 3
+      start_period: 10s
     environment:
       - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
       - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
```

## 12. service-without-healthcheck (fault-006, warning): `docker-compose.yaml` line 408

service grafana has ports but no healthcheck: nothing can wait for it (depends_on: service_healthy) and a broken service still shows as Up (fault-006)

Evidence: `grafana: publishes ${GRAFANA_PORT}:3000 and has no healthcheck`

Fix: add a healthcheck; its image has neither curl nor python3 visible in this repo, so no command is proposed

## 13. floating-image-tag (fault-002, warning): `docker-compose.yaml` line 409

grafana/grafana has a floating tag (no tag): it changes under you (fault-002: localstack:stable started requiring an account)

Evidence: `image: grafana/grafana`

Fix: pin grafana/grafana to the explicit version you run today; no known-good version is recorded, so no diff is proposed
