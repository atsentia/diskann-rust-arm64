# Docker Compose Deployment Guide for DiskANN Rust

This guide covers deploying the DiskANN Rust library as a containerized service using Docker Compose.

## Overview

DiskANN Rust is a high-performance vector search library that can be deployed as a REST API service. This deployment guide assumes you have built a web service wrapper around the core library.

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM (8GB+ recommended for production)
- For ARM64 optimizations: ARM64 host or ARM64 emulation
- For x86-64 optimizations: x86-64 host with AVX2/AVX512 support

## Architecture

The Docker Compose setup includes:

1. **DiskANN Service**: Main vector search API
2. **Redis** (optional): For caching and session management
3. **PostgreSQL** (optional): For metadata and configuration storage
4. **Nginx**: Reverse proxy and load balancer
5. **Prometheus** (optional): Metrics collection
6. **Grafana** (optional): Monitoring dashboard

## Directory Structure

```
diskann-deployment/
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env
├── configs/
│   ├── nginx.conf
│   ├── prometheus.yml
│   └── grafana/
├── data/
│   ├── vectors/
│   ├── indices/
│   └── postgres/
└── Dockerfile
```

## Service Configuration

### 1. Dockerfile

```dockerfile
# Multi-stage build for efficiency
FROM rust:1.75-slim as builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY examples/ ./examples/

# Build with release optimizations
# For ARM64 hosts
RUN cargo build --release --features neon
# For x86-64 hosts with AVX2
# RUN cargo build --release --features avx2
# For maximum compatibility
# RUN cargo build --release --no-default-features

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary and assets
COPY --from=builder /app/target/release/diskann-server .
COPY --from=builder /app/examples/ ./examples/

# Create directories for data
RUN mkdir -p /app/data/{vectors,indices,temp}

# Set ownership and permissions
RUN useradd -r -s /bin/false diskann && \
    chown -R diskann:diskann /app
USER diskann

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./diskann-server"]
```

### 2. Docker Compose Configuration

**docker-compose.yml** (Development):

```yaml
version: '3.8'

services:
  diskann:
    build: .
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - DISKANN_DATA_DIR=/app/data
      - DISKANN_MAX_THREADS=${DISKANN_MAX_THREADS:-4}
      - DISKANN_CACHE_SIZE_GB=${DISKANN_CACHE_SIZE_GB:-2}
    volumes:
      - ./data/vectors:/app/data/vectors
      - ./data/indices:/app/data/indices
      - ./data/temp:/app/data/temp
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    mem_limit: 4g
    cpus: 2.0

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./configs/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - diskann
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    mem_limit: 512m

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: diskann
      POSTGRES_USER: diskann
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    mem_limit: 1g

volumes:
  redis_data:
  postgres_data:
```

**docker-compose.prod.yml** (Production):

```yaml
version: '3.8'

services:
  diskann:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8g
          cpus: '4'
        reservations:
          memory: 4g
          cpus: '2'
    environment:
      - RUST_LOG=warn
      - DISKANN_DATA_DIR=/app/data
      - DISKANN_MAX_THREADS=8
      - DISKANN_CACHE_SIZE_GB=6
      - DISKANN_ENABLE_METRICS=true
    volumes:
      - vector_data:/app/data/vectors:ro
      - index_data:/app/data/indices
      - temp_data:/app/data/temp
    depends_on:
      - redis-cluster
      - postgres-primary
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./configs/nginx.prod.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - nginx_cache:/var/cache/nginx
    depends_on:
      - diskann
    restart: always
    deploy:
      resources:
        limits:
          memory: 1g
          cpus: '1'

  redis-cluster:
    image: redis:7-alpine
    command: redis-server --appendonly yes --cluster-enabled yes
    volumes:
      - redis_cluster_data:/data
    restart: always
    deploy:
      replicas: 3

  postgres-primary:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: diskann
      POSTGRES_USER: diskann
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${POSTGRES_REPL_PASSWORD}
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
    restart: always
    deploy:
      resources:
        limits:
          memory: 2g
          cpus: '2'

  postgres-replica:
    image: postgres:15-alpine
    environment:
      POSTGRES_MASTER_SERVICE: postgres-primary
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${POSTGRES_REPL_PASSWORD}
    depends_on:
      - postgres-primary
    restart: always

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: always

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./configs/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: always

volumes:
  vector_data:
  index_data:
  temp_data:
  nginx_cache:
  redis_cluster_data:
  postgres_primary_data:
  prometheus_data:
  grafana_data:
```

### 3. Environment Configuration

**.env** file:

```env
# Database
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_REPL_PASSWORD=your_replication_password_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here

# DiskANN Configuration
DISKANN_MAX_THREADS=8
DISKANN_CACHE_SIZE_GB=4
DISKANN_INDEX_TYPE=memory
DISKANN_DISTANCE_METRIC=l2

# API Configuration
API_RATE_LIMIT=1000
API_MAX_QUERY_SIZE=100
API_ENABLE_CORS=true

# Production optimizations
RUST_LOG=info
RAYON_NUM_THREADS=8
```

### 4. Nginx Configuration

**configs/nginx.conf**:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream diskann_backend {
        least_conn;
        server diskann:8080 max_fails=3 fail_timeout=30s;
        # Add more instances for production:
        # server diskann-2:8080 max_fails=3 fail_timeout=30s;
        # server diskann-3:8080 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;

    # Caching
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=diskann_cache:10m 
                     max_size=1g inactive=60m use_temp_path=off;

    server {
        listen 80;
        server_name localhost;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://diskann_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts for vector operations
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # Cache search results briefly
            proxy_cache diskann_cache;
            proxy_cache_valid 200 5m;
            proxy_cache_key "$request_method$request_uri$is_args$args";
        }

        # Health check
        location /health {
            proxy_pass http://diskann_backend;
            access_log off;
        }

        # Metrics (protect in production)
        location /metrics {
            proxy_pass http://diskann_backend;
            # allow 10.0.0.0/8;
            # deny all;
        }

        # Static files and documentation
        location / {
            root /usr/share/nginx/html;
            index index.html;
        }
    }

    # HTTPS configuration (production)
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        location /api/ {
            limit_req zone=api burst=50 nodelay;
            proxy_pass http://diskann_backend;
            # ... same proxy settings as above
        }
    }
}
```

## Deployment Commands

### Development Deployment

```bash
# Clone and setup
git clone https://github.com/atsentia/diskann-rust-arm64
cd diskann-rust-arm64

# Create data directories
mkdir -p data/{vectors,indices,temp}
mkdir -p configs ssl

# Setup environment
cp .env.example .env
# Edit .env with your values

# Start services
docker-compose up -d

# View logs
docker-compose logs -f diskann

# Health check
curl http://localhost/health
```

### Production Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale DiskANN service
docker-compose -f docker-compose.prod.yml up -d --scale diskann=5

# Update service
docker-compose -f docker-compose.prod.yml pull diskann
docker-compose -f docker-compose.prod.yml up -d --no-deps diskann
```

## Performance Tuning

### Memory Configuration

```yaml
# In docker-compose.yml, tune based on your data size:
services:
  diskann:
    environment:
      - DISKANN_CACHE_SIZE_GB=8  # Set to 50-70% of available memory
      - DISKANN_MAX_THREADS=8    # Match CPU cores
    mem_limit: 12g              # Leave headroom for OS
```

### ARM64 Optimization

```dockerfile
# In Dockerfile, for ARM64 hosts:
RUN cargo build --release --features neon

# Set CPU affinity for better performance
services:
  diskann:
    cpuset: "0-3"  # Bind to specific cores
```

### Storage Optimization

```yaml
# Use separate volumes for different data types
volumes:
  fast_storage:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=4g
  
services:
  diskann:
    volumes:
      - fast_storage:/app/data/temp  # Use tmpfs for temporary data
      - ./data/indices:/app/data/indices  # Persistent storage for indices
```

## Monitoring and Logging

### Prometheus Metrics

**configs/prometheus.yml**:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'diskann'
    static_configs:
      - targets: ['diskann:8080']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: /nginx_status

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### Log Aggregation

```yaml
# Add to docker-compose.yml
services:
  diskann:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
        labels: "service=diskann"

  # Optional: ELK stack for log aggregation
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: logstash:8.11.0
    volumes:
      - ./configs/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Scaling Strategies

### Horizontal Scaling

```bash
# Scale DiskANN service instances
docker-compose up -d --scale diskann=5

# Use Docker Swarm for multi-node scaling
docker swarm init
docker stack deploy -c docker-compose.prod.yml diskann-stack
```

### Load Balancing

```nginx
# In nginx.conf - weighted load balancing
upstream diskann_backend {
    server diskann-1:8080 weight=3;
    server diskann-2:8080 weight=2;
    server diskann-3:8080 weight=1 backup;
}
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/diskann_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup vector data and indices
docker run --rm -v diskann_index_data:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/indices.tar.gz -C /data .

# Backup database
docker-compose exec postgres pg_dump -U diskann diskann > $BACKUP_DIR/database.sql

# Backup configuration
cp -r configs $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
```

### Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

# Restore indices
docker run --rm -v diskann_index_data:/data -v $BACKUP_DIR:/backup \
  alpine tar xzf /backup/indices.tar.gz -C /data

# Restore database
docker-compose exec -T postgres psql -U diskann diskann < $BACKUP_DIR/database.sql

# Restart services
docker-compose restart diskann
```

## Security Considerations

### Network Security

```yaml
# Create custom network
networks:
  diskann_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  diskann:
    networks:
      - diskann_net
    # Don't expose ports directly to host in production
```

### Secrets Management

```yaml
# Use Docker secrets
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  ssl_cert:
    file: ./secrets/ssl_cert.pem

services:
  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase `DISKANN_CACHE_SIZE_GB` or container memory limits
2. **Slow Search**: Check CPU affinity and SIMD feature compilation
3. **Connection Refused**: Verify service dependencies and health checks
4. **High Latency**: Tune nginx caching and connection pooling

### Debug Commands

```bash
# Service logs
docker-compose logs -f diskann

# Container resource usage
docker stats

# Network connectivity
docker-compose exec diskann curl -f http://redis:6379/ping

# Performance monitoring
docker-compose exec diskann top
docker-compose exec diskann iostat -x 1
```

## Maintenance

### Updates

```bash
# Update base images
docker-compose pull

# Rebuild with latest code
docker-compose build --no-cache diskann

# Rolling update (zero downtime)
docker-compose up -d --scale diskann=3
docker-compose stop diskann_diskann_1
docker-compose rm diskann_diskann_1
docker-compose up -d --scale diskann=3
```

### Health Monitoring

```bash
# Automated health check script
#!/bin/bash
if ! curl -f http://localhost/health; then
  echo "DiskANN service unhealthy, restarting..."
  docker-compose restart diskann
fi
```

This deployment guide provides a comprehensive foundation for running DiskANN Rust in production using Docker Compose, with proper monitoring, security, and scaling considerations.