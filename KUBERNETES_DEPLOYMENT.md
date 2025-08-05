# Kubernetes Deployment Guide for DiskANN Rust

This guide covers deploying the DiskANN Rust library as a scalable service in Kubernetes environments.

## Overview

DiskANN Rust is deployed as a cloud-native vector search service with horizontal scaling, high availability, and efficient resource utilization. This guide supports both ARM64 and x86-64 Kubernetes clusters.

## Prerequisites

- Kubernetes cluster 1.24+ (EKS, GKE, AKS, or self-managed)
- kubectl configured for your cluster
- Helm 3.0+ (recommended for complex deployments)
- At least 3 worker nodes with 8GB+ RAM each
- Storage class supporting ReadWriteMany (for shared vector data)
- Ingress controller (nginx, traefik, or cloud provider)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                   │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │   Ingress   │ │ LoadBalancer│ │   Service   │            │
│ │  Controller │ │             │ │    Mesh     │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │  DiskANN    │ │  DiskANN    │ │  DiskANN    │            │
│ │   Pod 1     │ │   Pod 2     │ │   Pod 3     │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │    Redis    │ │ PostgreSQL  │ │ Prometheus  │            │
│ │   Cluster   │ │  Primary    │ │  + Grafana  │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │ Persistent  │ │   Config    │ │   Secret    │            │
│ │  Volumes    │ │    Maps     │ │   Store     │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
k8s-manifests/
├── namespace.yaml
├── configmaps/
│   ├── diskann-config.yaml
│   ├── nginx-config.yaml
│   └── prometheus-config.yaml
├── secrets/
│   ├── diskann-secrets.yaml
│   └── tls-secrets.yaml
├── storage/
│   ├── storage-class.yaml
│   ├── persistent-volumes.yaml
│   └── vector-data-pvc.yaml
├── deployments/
│   ├── diskann-deployment.yaml
│   ├── redis-deployment.yaml
│   ├── postgres-deployment.yaml
│   └── monitoring-deployment.yaml
├── services/
│   ├── diskann-service.yaml
│   ├── redis-service.yaml
│   └── postgres-service.yaml
├── ingress/
│   ├── diskann-ingress.yaml
│   └── monitoring-ingress.yaml
├── autoscaling/
│   ├── diskann-hpa.yaml
│   ├── diskann-vpa.yaml
│   └── cluster-autoscaler.yaml
├── rbac/
│   ├── service-account.yaml
│   └── rbac.yaml
├── monitoring/
│   ├── servicemonitor.yaml
│   ├── alerting-rules.yaml
│   └── dashboards/
└── helm/
    └── diskann-chart/
```

## Core Kubernetes Manifests

### 1. Namespace and RBAC

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: diskann
  labels:
    name: diskann
    environment: production
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: diskann-service-account
  namespace: diskann
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: diskann-cluster-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["*"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: diskann-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: diskann-cluster-role
subjects:
- kind: ServiceAccount
  name: diskann-service-account
  namespace: diskann
```

### 2. Configuration Management

**configmaps/diskann-config.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: diskann-config
  namespace: diskann
data:
  # DiskANN Configuration
  max_threads: "8"
  cache_size_gb: "6"
  index_type: "memory"
  distance_metric: "l2"
  search_list_size: "100"
  max_degree: "32"
  alpha: "1.2"
  
  # API Configuration
  bind_address: "0.0.0.0:8080"
  api_rate_limit: "1000"
  api_max_query_size: "100"
  api_enable_cors: "true"
  api_timeout_seconds: "30"
  
  # Monitoring
  enable_metrics: "true"
  metrics_path: "/metrics"
  log_level: "info"
  
  # Storage paths
  data_dir: "/app/data"
  vector_dir: "/app/data/vectors"
  index_dir: "/app/data/indices"
  temp_dir: "/app/data/temp"
  
  # Redis configuration
  redis_url: "redis://redis-service:6379"
  redis_pool_size: "10"
  
  # Database configuration
  database_url: "postgresql://diskann:password@postgres-service:5432/diskann"
  database_pool_size: "10"
```

### 3. Secrets Management

**secrets/diskann-secrets.yaml**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: diskann-secrets
  namespace: diskann
type: Opaque
data:
  # Base64 encoded values
  postgres-password: cGFzc3dvcmQxMjM=  # password123
  redis-password: cmVkaXNwYXNzd29yZA==    # redispassword
  api-key: YXBpa2V5MTIzNDU2Nzg5MA==      # apikey1234567890
  jwt-secret: and0c2VjcmV0a2V5Zm9yand0dG9rZW5z  # jwtsecretkeyforjwttokens
---
apiVersion: v1
kind: Secret
metadata:
  name: diskann-tls
  namespace: diskann
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi... # Your TLS certificate
  tls.key: LS0tLS1CRUdJTi... # Your TLS private key
```

### 4. Storage Configuration

**storage/storage-class.yaml**:
```yaml
# Fast SSD storage class for indices
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # Change for your cloud provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
---
# Shared storage for vector data
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: shared-storage
provisioner: efs.csi.aws.com  # Use appropriate shared storage provider
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-1234567890abcdef0
  directoryPerms: "0755"
volumeBindingMode: Immediate
allowVolumeExpansion: true
```

**storage/persistent-volumes.yaml**:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: diskann-index-storage
  namespace: diskann
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: diskann-vector-storage
  namespace: diskann
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: shared-storage
  resources:
    requests:
      storage: 500Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: diskann-temp-storage
  namespace: diskann
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi
```

### 5. DiskANN Deployment

**deployments/diskann-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diskann-deployment
  namespace: diskann
  labels:
    app: diskann
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: diskann
  template:
    metadata:
      labels:
        app: diskann
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: diskann-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      # Node selection for ARM64 optimization
      nodeSelector:
        kubernetes.io/arch: arm64  # Change to amd64 for x86-64
      
      # Prefer spreading across zones
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - diskann
              topologyKey: kubernetes.io/hostname
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 50
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values:
                - m6g.2xlarge  # ARM64 instances
                - c6g.4xlarge
      
      containers:
      - name: diskann
        image: diskann-rust:v1.0.0
        imagePullPolicy: IfNotPresent
        
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        
        env:
        - name: RUST_LOG
          value: "info"
        - name: RAYON_NUM_THREADS
          valueFrom:
            configMapKeyRef:
              name: diskann-config
              key: max_threads
        - name: DISKANN_DATA_DIR
          valueFrom:
            configMapKeyRef:
              name: diskann-config
              key: data_dir
        - name: DISKANN_CACHE_SIZE_GB
          valueFrom:
            configMapKeyRef:
              name: diskann-config
              key: cache_size_gb
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: diskann-secrets
              key: postgres-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: diskann-secrets
              key: redis-password
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        
        volumeMounts:
        - name: vector-storage
          mountPath: /app/data/vectors
          readOnly: true
        - name: index-storage
          mountPath: /app/data/indices
        - name: temp-storage
          mountPath: /app/data/temp
        - name: config
          mountPath: /app/config
          readOnly: true
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Graceful shutdown
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      
      volumes:
      - name: vector-storage
        persistentVolumeClaim:
          claimName: diskann-vector-storage
      - name: index-storage
        persistentVolumeClaim:
          claimName: diskann-index-storage
      - name: temp-storage
        persistentVolumeClaim:
          claimName: diskann-temp-storage
      - name: config
        configMap:
          name: diskann-config
      
      terminationGracePeriodSeconds: 30
      
      # Topology spread for high availability
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: diskann
```

### 6. Service Configuration

**services/diskann-service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: diskann-service
  namespace: diskann
  labels:
    app: diskann
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"  # For AWS
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
spec:
  type: LoadBalancer
  selector:
    app: diskann
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 300
---
apiVersion: v1
kind: Service
metadata:
  name: diskann-internal
  namespace: diskann
  labels:
    app: diskann
spec:
  type: ClusterIP
  selector:
    app: diskann
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 8080
    protocol: TCP
```

### 7. Ingress Configuration

**ingress/diskann-ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: diskann-ingress
  namespace: diskann
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-connections: "10"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.diskann.example.com
    secretName: diskann-tls
  rules:
  - host: api.diskann.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: diskann-service
            port:
              number: 80
      - path: /api/v1/
        pathType: Prefix
        backend:
          service:
            name: diskann-internal
            port:
              number: 8080
```

## Auto-scaling Configuration

### 8. Horizontal Pod Autoscaler

**autoscaling/diskann-hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: diskann-hpa
  namespace: diskann
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: diskann-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: search_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

### 9. Vertical Pod Autoscaler

**autoscaling/diskann-vpa.yaml**:
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: diskann-vpa
  namespace: diskann
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: diskann-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: diskann
      minAllowed:
        cpu: 1
        memory: 2Gi
      maxAllowed:
        cpu: 8
        memory: 16Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

## Database and Cache Deployments

### 10. PostgreSQL Deployment

**deployments/postgres-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-primary
  namespace: diskann
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
      role: primary
  template:
    metadata:
      labels:
        app: postgres
        role: primary
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: diskann
        - name: POSTGRES_USER
          value: diskann
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: diskann-secrets
              key: postgres-password
        - name: POSTGRES_REPLICATION_MODE
          value: master
        - name: POSTGRES_REPLICATION_USER
          value: replicator
        - name: POSTGRES_REPLICATION_PASSWORD
          valueFrom:
            secretKeyRef:
              name: diskann-secrets
              key: postgres-password
        
        ports:
        - containerPort: 5432
          name: postgres
        
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
      
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### 11. Redis Cluster

**deployments/redis-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: diskann
spec:
  serviceName: redis-service
  replicas: 6
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - /etc/redis/redis.conf
        - --cluster-enabled
        - "yes"
        - --cluster-config-file
        - nodes.conf
        - --cluster-node-timeout
        - "5000"
        - --appendonly
        - "yes"
        - --protected-mode
        - "no"
        
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        
        volumeMounts:
        - name: redis-data
          mountPath: /data
        - name: redis-config
          mountPath: /etc/redis
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
  
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 20Gi
```

## Monitoring and Observability

### 12. Prometheus ServiceMonitor

**monitoring/servicemonitor.yaml**:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: diskann-metrics
  namespace: diskann
  labels:
    app: diskann
spec:
  selector:
    matchLabels:
      app: diskann
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    honorLabels: true
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: diskann-alerts
  namespace: diskann
spec:
  groups:
  - name: diskann.rules
    rules:
    - alert: DiskANNHighMemoryUsage
      expr: (container_memory_working_set_bytes{pod=~"diskann-.*"} / container_spec_memory_limit_bytes{pod=~"diskann-.*"}) > 0.9
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "DiskANN pod {{ $labels.pod }} has high memory usage"
        description: "Memory usage is above 90% for 5 minutes"
    
    - alert: DiskANNHighCPUUsage
      expr: rate(container_cpu_usage_seconds_total{pod=~"diskann-.*"}[5m]) > 0.8
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "DiskANN pod {{ $labels.pod }} has high CPU usage"
    
    - alert: DiskANNSearchLatencyHigh
      expr: histogram_quantile(0.95, rate(search_duration_seconds_bucket[5m])) > 1.0
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "DiskANN search latency is high"
        description: "95th percentile search latency is above 1 second"
    
    - alert: DiskANNPodCrashLooping
      expr: increase(kube_pod_container_status_restarts_total{pod=~"diskann-.*"}[1h]) > 5
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "DiskANN pod {{ $labels.pod }} is crash looping"
```

## Deployment Commands

### Development Deployment

```bash
# Create namespace and RBAC
kubectl apply -f namespace.yaml

# Create secrets (update values first)
kubectl apply -f secrets/

# Create storage
kubectl apply -f storage/

# Create config maps
kubectl apply -f configmaps/

# Deploy database and cache
kubectl apply -f deployments/postgres-deployment.yaml
kubectl apply -f deployments/redis-deployment.yaml
kubectl apply -f services/

# Wait for dependencies
kubectl wait --for=condition=ready pod -l app=postgres -n diskann --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n diskann --timeout=300s

# Deploy DiskANN
kubectl apply -f deployments/diskann-deployment.yaml

# Create services and ingress
kubectl apply -f services/diskann-service.yaml
kubectl apply -f ingress/

# Setup autoscaling
kubectl apply -f autoscaling/

# Setup monitoring
kubectl apply -f monitoring/
```

### Production Deployment with Helm

**helm/diskann-chart/Chart.yaml**:
```yaml
apiVersion: v2
name: diskann
description: A Helm chart for DiskANN Rust vector search service
type: application
version: 1.0.0
appVersion: "1.0.0"
dependencies:
- name: postgresql
  version: 11.9.13
  repository: https://charts.bitnami.com/bitnami
- name: redis
  version: 17.3.7
  repository: https://charts.bitnami.com/bitnami
- name: prometheus
  version: 15.5.3
  repository: https://prometheus-community.github.io/helm-charts
```

**helm/diskann-chart/values.yaml**:
```yaml
# Default values for diskann
replicaCount: 3

image:
  repository: diskann-rust
  pullPolicy: IfNotPresent
  tag: "v1.0.0"

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.diskann.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: diskann-tls
      hosts:
        - api.diskann.example.com

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 2
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector:
  kubernetes.io/arch: arm64

persistence:
  vectors:
    enabled: true
    storageClass: shared-storage
    size: 500Gi
  indices:
    enabled: true
    storageClass: fast-ssd
    size: 100Gi

postgresql:
  enabled: true
  auth:
    username: diskann
    database: diskann
  primary:
    persistence:
      size: 100Gi

redis:
  enabled: true
  architecture: replication
  master:
    persistence:
      size: 20Gi

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
```

**Helm deployment commands**:
```bash
# Add required repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install with custom values
helm install diskann ./helm/diskann-chart \
  --namespace diskann \
  --create-namespace \
  --values values-production.yaml

# Upgrade deployment
helm upgrade diskann ./helm/diskann-chart \
  --namespace diskann \
  --values values-production.yaml

# Rollback if needed
helm rollback diskann 1 --namespace diskann
```

## Multi-Region Deployment

### Federation Setup

```yaml
# diskann-federation.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: diskann-federation
  namespace: diskann
data:
  regions.yaml: |
    regions:
      us-east-1:
        endpoint: https://api-us-east.diskann.example.com
        weight: 40
        latency_threshold_ms: 100
      us-west-2:
        endpoint: https://api-us-west.diskann.example.com
        weight: 30
        latency_threshold_ms: 150
      eu-west-1:
        endpoint: https://api-eu.diskann.example.com
        weight: 30
        latency_threshold_ms: 200
    
    routing:
      strategy: "latency_based"
      fallback: "round_robin"
      health_check_interval: 30s
```

### Cross-Region Service Mesh

```yaml
# Using Istio for cross-region communication
apiVersion: networking.istio.io/v1alpha3
kind: ServiceEntry
metadata:
  name: diskann-us-west
  namespace: diskann
spec:
  hosts:
  - api-us-west.diskann.example.com
  ports:
  - number: 443
    name: https
    protocol: HTTPS
  location: MESH_EXTERNAL
  resolution: DNS
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: diskann-global
  namespace: diskann
spec:
  host: "*.diskann.example.com"
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 1000
        maxRequestsPerConnection: 10
    loadBalancer:
      localityLbSetting:
        enabled: true
        failover:
        - from: region/us-east-1/*
          to: region/us-west-2/*
```

## Security and Compliance

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: diskann-network-policy
  namespace: diskann
spec:
  podSelector:
    matchLabels:
      app: diskann
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### Pod Security Standards

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: diskann-pod
  namespace: diskann
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: diskann
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

## Performance Optimization

### Resource Allocation

```yaml
# Performance-optimized resource allocation
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
    hugepages-2Mi: "1Gi"  # For large memory pages
  limits:
    memory: "16Gi"
    cpu: "8"
    hugepages-2Mi: "1Gi"

# CPU pinning for consistent performance
annotations:
  cpu-manager.alpha.kubernetes.io/cpuset: "0-7"
```

### Node Configuration

```yaml
# Node optimization for vector workloads
apiVersion: v1
kind: Node
metadata:
  name: diskann-node
  labels:
    node-type: vector-compute
    cpu-type: arm64-neon
spec:
  taints:
  - key: diskann-dedicated
    value: "true"
    effect: NoSchedule
```

## Disaster Recovery

### Backup Strategy

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: diskann-backup
  namespace: diskann
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: diskann-backup:latest
            command:
            - /bin/sh
            - -c
            - |
              # Backup vector data
              tar czf /backup/vectors-$(date +%Y%m%d).tar.gz /app/data/vectors
              
              # Backup indices
              tar czf /backup/indices-$(date +%Y%m%d).tar.gz /app/data/indices
              
              # Upload to cloud storage
              aws s3 cp /backup/ s3://diskann-backups/ --recursive
            
            volumeMounts:
            - name: vector-storage
              mountPath: /app/data/vectors
              readOnly: true
            - name: index-storage
              mountPath: /app/data/indices
              readOnly: true
            - name: backup-storage
              mountPath: /backup
          
          volumes:
          - name: vector-storage
            persistentVolumeClaim:
              claimName: diskann-vector-storage
          - name: index-storage
            persistentVolumeClaim:
              claimName: diskann-index-storage
          - name: backup-storage
            emptyDir: {}
          
          restartPolicy: OnFailure
```

### Recovery Procedures

```bash
#!/bin/bash
# disaster-recovery.sh

# 1. Restore from backup
kubectl create namespace diskann-recovery
kubectl apply -f recovery-storage.yaml

# 2. Restore data
kubectl run restore-job --image=diskann-backup:latest \
  --namespace=diskann-recovery \
  --command -- /bin/sh -c "
    aws s3 cp s3://diskann-backups/vectors-latest.tar.gz /tmp/
    aws s3 cp s3://diskann-backups/indices-latest.tar.gz /tmp/
    tar xzf /tmp/vectors-latest.tar.gz -C /recovery/vectors/
    tar xzf /tmp/indices-latest.tar.gz -C /recovery/indices/
  "

# 3. Deploy in recovery mode
helm install diskann-recovery ./helm/diskann-chart \
  --namespace diskann-recovery \
  --set recovery.enabled=true \
  --set persistence.vectors.existingClaim=recovery-vectors

# 4. Verify and switch traffic
kubectl patch ingress diskann-ingress \
  --namespace diskann \
  --patch '{"spec":{"rules":[{"host":"api.diskann.example.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"diskann-service","port":{"number":80}}}}]}}]}}'
```

This comprehensive Kubernetes deployment guide provides enterprise-grade scalability, security, and operational capabilities for the DiskANN Rust vector search service.