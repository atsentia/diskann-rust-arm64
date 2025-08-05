# Modal.com Deployment Guide for DiskANN Rust

This guide covers deploying the DiskANN Rust library as a serverless vector search service using Modal.com's cloud platform.

## Overview

Modal.com provides a serverless compute platform that's ideal for deploying machine learning and data processing workloads. This guide shows how to deploy DiskANN Rust with automatic scaling, GPU acceleration, and managed infrastructure.

## Prerequisites

- Modal account (https://modal.com)
- Modal CLI installed (`pip install modal`)
- Modal token configured (`modal token set`)
- Docker for local testing
- Python 3.9+ for Modal integration scripts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Modal.com Platform                     │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │   FastAPI   │ │   Worker    │ │   Worker    │            │
│ │   Endpoint  │ │   Container │ │   Container │            │
│ │             │ │             │ │             │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │   Network   │ │   Volumes   │ │   Secrets   │            │
│ │ File System │ │   Storage   │ │   Manager   │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │ Prometheus  │ │    Logs     │ │    Queue    │            │
│ │  Metrics    │ │ Management  │ │   System    │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
modal-deployment/
├── modal_app.py          # Main Modal application
├── requirements.txt      # Python dependencies
├── Dockerfile.modal      # Custom container image
├── config/
│   ├── modal_config.py   # Modal-specific configuration
│   ├── diskann_config.py # DiskANN configuration
│   └── secrets.py        # Secret management
├── src/
│   ├── api/
│   │   ├── endpoints.py  # FastAPI endpoints
│   │   ├── models.py     # Pydantic models
│   │   └── middleware.py # Custom middleware
│   ├── services/
│   │   ├── vector_service.py  # Vector operations
│   │   ├── index_service.py   # Index management
│   │   └── search_service.py  # Search operations
│   └── utils/
│       ├── monitoring.py # Metrics and logging
│       └── data_loader.py # Data loading utilities
├── data/
│   ├── vectors/          # Vector datasets
│   ├── indices/          # Pre-built indices
│   └── models/           # ML models
├── scripts/
│   ├── deploy.py         # Deployment script
│   ├── data_upload.py    # Data upload utility
│   └── benchmark.py      # Performance testing
└── tests/
    ├── test_api.py       # API tests
    ├── test_performance.py # Performance tests
    └── fixtures/         # Test data
```

## Core Modal Application

### 1. Main Application Definition

**modal_app.py**:
```python
import modal
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("diskann-vector-search")

# Custom container image with DiskANN Rust
diskann_image = (
    modal.Image.from_dockerfile("Dockerfile.modal")
    .pip_install([
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "numpy==1.24.3",
        "scipy==1.11.4",
        "redis==5.0.1",
        "httpx==0.25.2",
        "prometheus-client==0.19.0",
        "structlog==23.2.0",
    ])
    .run_commands([
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "source ~/.cargo/env",
        "rustup target add aarch64-unknown-linux-gnu",
    ])
)

# Network file system for persistent data
nfs = modal.NetworkFileSystem.from_name("diskann-data", create_if_missing=True)

# Shared volumes for different data types
vector_volume = modal.Volume.from_name("diskann-vectors", create_if_missing=True)
index_volume = modal.Volume.from_name("diskann-indices", create_if_missing=True)
temp_volume = modal.Volume.from_name("diskann-temp", create_if_missing=True)

# Secrets for external services
secrets = [
    modal.Secret.from_name("diskann-secrets"),  # API keys, DB passwords
    modal.Secret.from_name("redis-credentials"),
    modal.Secret.from_name("monitoring-tokens"),
]

# GPU configuration for large-scale operations
gpu_config = modal.gpu.A100(size="40GB")

@app.cls(
    image=diskann_image,
    volumes={
        "/data/vectors": vector_volume,
        "/data/indices": index_volume,
        "/data/temp": temp_volume,
    },
    network_file_systems={"/nfs": nfs},
    secrets=secrets,
    cpu=8.0,
    memory=32768,  # 32GB
    timeout=3600,  # 1 hour timeout
    concurrency_limit=10,
    allow_concurrent_inputs=100,
)
class DiskANNService:
    """DiskANN vector search service running on Modal."""
    
    def __init__(self):
        self.index_cache = {}
        self.redis_client = None
        self.metrics_enabled = True
        
    @modal.enter()
    def setup(self):
        """Initialize the service on container startup."""
        import redis
        import subprocess
        import json
        from pathlib import Path
        
        logger.info("Initializing DiskANN service...")
        
        # Build DiskANN Rust binary if not exists
        binary_path = Path("/app/diskann-server")
        if not binary_path.exists():
            logger.info("Building DiskANN Rust binary...")
            result = subprocess.run([
                "cargo", "build", "--release", 
                "--features", "neon,parallel",
                "--target-dir", "/tmp/target"
            ], cwd="/app", capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                raise RuntimeError("Failed to build DiskANN binary")
            
            # Copy binary to expected location
            subprocess.run([
                "cp", "/tmp/target/release/diskann-server", "/app/"
            ])
            
        # Setup Redis connection
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize metrics
        if self.metrics_enabled:
            self._setup_metrics()
        
        # Pre-load common indices
        self._preload_indices()
        
        logger.info("DiskANN service initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load DiskANN configuration."""
        return {
            "max_threads": int(os.environ.get("DISKANN_MAX_THREADS", "8")),
            "cache_size_gb": int(os.environ.get("DISKANN_CACHE_SIZE_GB", "16")),
            "search_list_size": int(os.environ.get("DISKANN_SEARCH_LIST_SIZE", "100")),
            "max_degree": int(os.environ.get("DISKANN_MAX_DEGREE", "32")),
            "alpha": float(os.environ.get("DISKANN_ALPHA", "1.2")),
            "distance_metric": os.environ.get("DISKANN_DISTANCE_METRIC", "l2"),
        }
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        from prometheus_client import Counter, Histogram, Gauge
        
        self.search_requests = Counter(
            'diskann_search_requests_total',
            'Total number of search requests',
            ['index_name', 'status']
        )
        
        self.search_duration = Histogram(
            'diskann_search_duration_seconds',
            'Search request duration',
            ['index_name']
        )
        
        self.index_size = Gauge(
            'diskann_index_size_bytes',
            'Index size in bytes',
            ['index_name']
        )
        
        self.cache_hits = Counter(
            'diskann_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
    
    def _preload_indices(self):
        """Pre-load frequently used indices."""
        import subprocess
        import json
        from pathlib import Path
        
        indices_dir = Path("/data/indices")
        if not indices_dir.exists():
            logger.info("No indices directory found, skipping preload")
            return
        
        # Load index metadata
        for index_file in indices_dir.glob("*.index"):
            index_name = index_file.stem
            try:
                # Load index using DiskANN binary
                cmd = [
                    "/app/diskann-server",
                    "load-index",
                    "--index-path", str(index_file),
                    "--cache-size", str(self.config["cache_size_gb"]),
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.index_cache[index_name] = {
                        "path": str(index_file),
                        "loaded": True,
                        "metadata": json.loads(result.stdout)
                    }
                    logger.info(f"Preloaded index: {index_name}")
                else:
                    logger.warning(f"Failed to preload index {index_name}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error preloading index {index_name}: {e}")
    
    @modal.method()
    def search_vectors(
        self, 
        query: List[float], 
        k: int = 10,
        index_name: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for k nearest neighbors."""
        import time
        import subprocess
        import json
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not query or len(query) == 0:
                raise ValueError("Query vector cannot be empty")
            
            if k <= 0 or k > 1000:
                raise ValueError("k must be between 1 and 1000")
            
            # Check if index is loaded
            if index_name not in self.index_cache:
                raise ValueError(f"Index {index_name} not found")
            
            # Prepare search command
            cmd = [
                "/app/diskann-server",
                "search",
                "--index-name", index_name,
                "--query", json.dumps(query),
                "--k", str(k),
                "--search-list-size", str(self.config["search_list_size"]),
            ]
            
            if filters:
                cmd.extend(["--filters", json.dumps(filters)])
            
            # Execute search
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"Search failed: {result.stderr}")
            
            # Parse results
            search_results = json.loads(result.stdout)
            
            # Cache results if Redis is available
            if self.redis_client:
                cache_key = f"search:{index_name}:{hash(str(query))}:{k}"
                self.redis_client.setex(
                    cache_key, 
                    300,  # 5 minutes TTL
                    json.dumps(search_results)
                )
                self.cache_hits.labels(cache_type="search").inc()
            
            # Update metrics
            duration = time.time() - start_time
            if self.metrics_enabled:
                self.search_requests.labels(
                    index_name=index_name, 
                    status="success"
                ).inc()
                self.search_duration.labels(index_name=index_name).observe(duration)
            
            return {
                "results": search_results,
                "query_time_ms": duration * 1000,
                "index_name": index_name,
                "k": k,
                "total_results": len(search_results)
            }
            
        except Exception as e:
            if self.metrics_enabled:
                self.search_requests.labels(
                    index_name=index_name, 
                    status="error"
                ).inc()
            
            logger.error(f"Search error: {e}")
            raise
    
    @modal.method()
    def build_index(
        self,
        vectors: List[List[float]],
        index_name: str,
        labels: Optional[List[List[int]]] = None,
        distance_metric: str = "l2"
    ) -> Dict[str, Any]:
        """Build a new index from vectors."""
        import tempfile
        import subprocess
        import json
        import shutil
        from pathlib import Path
        
        try:
            if not vectors or len(vectors) == 0:
                raise ValueError("Vectors list cannot be empty")
            
            if len(vectors) < 100:
                raise ValueError("Need at least 100 vectors to build an index")
            
            # Create temporary directory for index building
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write vectors to temporary file
                vector_file = temp_path / "vectors.bin"
                with open(vector_file, "wb") as f:
                    # Write in binary format expected by DiskANN
                    import struct
                    import numpy as np
                    
                    arr = np.array(vectors, dtype=np.float32)
                    f.write(struct.pack("I", len(vectors)))  # Number of vectors
                    f.write(struct.pack("I", len(vectors[0])))  # Dimension
                    arr.tobytes()
                
                # Write labels if provided
                labels_file = None
                if labels:
                    labels_file = temp_path / "labels.txt"
                    with open(labels_file, "w") as f:
                        for i, label_list in enumerate(labels):
                            f.write(f"{i}\t{','.join(map(str, label_list))}\n")
                
                # Build index
                cmd = [
                    "/app/diskann-server",
                    "build-index",
                    "--vectors", str(vector_file),
                    "--output", str(temp_path / f"{index_name}.index"),
                    "--distance", distance_metric,
                    "--max-degree", str(self.config["max_degree"]),
                    "--alpha", str(self.config["alpha"]),
                    "--threads", str(self.config["max_threads"]),
                ]
                
                if labels_file:
                    cmd.extend(["--labels", str(labels_file)])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Index building failed: {result.stderr}")
                
                # Move index to persistent storage
                index_path = Path(f"/data/indices/{index_name}.index")
                shutil.move(str(temp_path / f"{index_name}.index"), str(index_path))
                
                # Update cache
                metadata = json.loads(result.stdout)
                self.index_cache[index_name] = {
                    "path": str(index_path),
                    "loaded": True,
                    "metadata": metadata
                }
                
                # Update metrics
                if self.metrics_enabled:
                    self.index_size.labels(index_name=index_name).set(
                        index_path.stat().st_size
                    )
                
                logger.info(f"Successfully built index: {index_name}")
                
                return {
                    "index_name": index_name,
                    "vector_count": len(vectors),
                    "dimension": len(vectors[0]),
                    "distance_metric": distance_metric,
                    "metadata": metadata,
                    "size_bytes": index_path.stat().st_size,
                }
                
        except Exception as e:
            logger.error(f"Index building error: {e}")
            raise
    
    @modal.method()
    def list_indices(self) -> Dict[str, Any]:
        """List all available indices."""
        return {
            "indices": [
                {
                    "name": name,
                    "loaded": info["loaded"],
                    "metadata": info.get("metadata", {}),
                    "path": info["path"]
                }
                for name, info in self.index_cache.items()
            ],
            "total_count": len(self.index_cache)
        }
    
    @modal.method()
    def get_health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        import psutil
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "indices_loaded": len(self.index_cache),
            "redis_connected": self.redis_client is not None,
            "memory_usage": {
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "percent": psutil.virtual_memory().percent
            },
            "cpu_usage": psutil.cpu_percent(interval=1),
            "config": self.config
        }

# GPU-accelerated operations for large-scale processing
@app.cls(
    image=diskann_image,
    gpu=gpu_config,
    volumes={
        "/data/vectors": vector_volume,
        "/data/indices": index_volume,
        "/data/temp": temp_volume,
    },
    secrets=secrets,
    cpu=16.0,
    memory=65536,  # 64GB
    timeout=7200,  # 2 hours
    concurrency_limit=2,
)
class DiskANNGPUService:
    """GPU-accelerated DiskANN operations for large datasets."""
    
    @modal.enter()
    def setup_gpu(self):
        """Initialize GPU service."""
        import subprocess
        
        # Verify GPU availability
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("GPU not available")
        
        logger.info("GPU service initialized")
        logger.info(result.stdout)
    
    @modal.method()
    def build_large_index(
        self,
        vector_file_path: str,
        index_name: str,
        distance_metric: str = "l2",
        use_gpu: bool = True
    ) -> Dict[str, Any]:
        """Build index for large datasets using GPU acceleration."""
        import subprocess
        import json
        from pathlib import Path
        
        try:
            vector_path = Path(f"/data/vectors/{vector_file_path}")
            if not vector_path.exists():
                raise FileNotFoundError(f"Vector file not found: {vector_file_path}")
            
            output_path = Path(f"/data/indices/{index_name}.index")
            
            cmd = [
                "/app/diskann-server",
                "build-index-gpu",
                "--vectors", str(vector_path),
                "--output", str(output_path),
                "--distance", distance_metric,
                "--max-degree", "64",
                "--alpha", "1.2",
                "--threads", "16",
            ]
            
            if use_gpu:
                cmd.append("--use-gpu")
            
            logger.info(f"Building large index: {index_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode != 0:
                raise RuntimeError(f"GPU index building failed: {result.stderr}")
            
            metadata = json.loads(result.stdout)
            
            return {
                "index_name": index_name,
                "input_file": vector_file_path,
                "output_path": str(output_path),
                "metadata": metadata,
                "gpu_used": use_gpu,
                "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            }
            
        except Exception as e:
            logger.error(f"GPU index building error: {e}")
            raise

# FastAPI web server
@app.function(
    image=diskann_image,
    secrets=secrets,
    allow_concurrent_inputs=1000,
    timeout=900,
)
@modal.asgi_app()
def web_app():
    """FastAPI web application for HTTP API."""
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any, Optional
    import asyncio
    import httpx
    
    # Initialize FastAPI
    api = FastAPI(
        title="DiskANN Vector Search API",
        description="High-performance vector search using DiskANN Rust",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add middleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Pydantic models
    class SearchRequest(BaseModel):
        query: List[float] = Field(..., description="Query vector")
        k: int = Field(10, ge=1, le=1000, description="Number of results")
        index_name: str = Field("default", description="Index name")
        filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    
    class BuildIndexRequest(BaseModel):
        vectors: List[List[float]] = Field(..., description="Vector data")
        index_name: str = Field(..., description="Index name")
        labels: Optional[List[List[int]]] = Field(None, description="Vector labels")
        distance_metric: str = Field("l2", description="Distance metric")
    
    class SearchResponse(BaseModel):
        results: List[Dict[str, Any]]
        query_time_ms: float
        index_name: str
        k: int
        total_results: int
    
    # Initialize service
    service = DiskANNService()
    gpu_service = DiskANNGPUService()
    
    @api.get("/")
    async def root():
        return {"message": "DiskANN Vector Search API", "version": "1.0.0"}
    
    @api.get("/health")
    async def health_check():
        try:
            health = await service.get_health.aio()
            return health
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")
    
    @api.post("/search", response_model=SearchResponse)
    async def search_vectors(request: SearchRequest):
        try:
            result = await service.search_vectors.aio(
                query=request.query,
                k=request.k,
                index_name=request.index_name,
                filters=request.filters
            )
            return SearchResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    
    @api.post("/indices/build")
    async def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
        try:
            # For large datasets, use GPU service
            if len(request.vectors) > 100000:
                background_tasks.add_task(
                    gpu_service.build_large_index.aio,
                    vectors=request.vectors,
                    index_name=request.index_name,
                    distance_metric=request.distance_metric
                )
                return {"message": "Large index build started in background", "index_name": request.index_name}
            else:
                result = await service.build_index.aio(
                    vectors=request.vectors,
                    index_name=request.index_name,
                    labels=request.labels,
                    distance_metric=request.distance_metric
                )
                return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Index building failed: {e}")
    
    @api.get("/indices")
    async def list_indices():
        try:
            result = await service.list_indices.aio()
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list indices: {e}")
    
    @api.get("/metrics")
    async def get_metrics():
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    return api

# Scheduled functions for maintenance
@app.function(
    schedule=modal.Cron("0 2 * * *"),  # Daily at 2 AM UTC
    image=diskann_image,
    volumes={"/data/indices": index_volume},
    secrets=secrets,
    timeout=3600,
)
def cleanup_temp_files():
    """Clean up temporary files daily."""
    import shutil
    from pathlib import Path
    
    temp_dir = Path("/data/temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cleaned up temporary files")

@app.function(
    schedule=modal.Cron("0 3 * * 0"),  # Weekly on Sunday at 3 AM UTC
    image=diskann_image,
    volumes={
        "/data/vectors": vector_volume,
        "/data/indices": index_volume,
    },
    secrets=secrets,
    timeout=7200,
)
def backup_indices():
    """Weekly backup of indices to external storage."""
    import subprocess
    import os
    from pathlib import Path
    
    backup_bucket = os.environ.get("BACKUP_S3_BUCKET")
    if not backup_bucket:
        logger.warning("No backup bucket configured")
        return
    
    indices_dir = Path("/data/indices")
    for index_file in indices_dir.glob("*.index"):
        cmd = [
            "aws", "s3", "cp", 
            str(index_file),
            f"s3://{backup_bucket}/indices/{index_file.name}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Backed up index: {index_file.name}")
        else:
            logger.error(f"Backup failed for {index_file.name}: {result.stderr}")

if __name__ == "__main__":
    # Local development
    modal.run(web_app)
```

### 2. Custom Container Image

**Dockerfile.modal**:
```dockerfile
FROM rust:1.75-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust project
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY examples/ ./examples/

# Build with optimizations for Modal's ARM64 instances
RUN cargo build --release --features neon,parallel

# Runtime stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binary
COPY --from=builder /app/target/release/diskann-server ./diskann-server
COPY --from=builder /app/examples/ ./examples/

# Copy Python API code
COPY src/ ./src/
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p /data/{vectors,indices,temp} && \
    useradd -r -s /bin/false diskann && \
    chown -R diskann:diskann /app /data

USER diskann

EXPOSE 8080

CMD ["./diskann-server"]
```

### 3. Configuration Management

**config/modal_config.py**:
```python
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModalConfig:
    """Modal-specific configuration."""
    
    # Resource allocation
    cpu_cores: int = 8
    memory_gb: int = 32
    gpu_enabled: bool = False
    gpu_type: str = "A100"
    
    # Scaling settings
    concurrency_limit: int = 10
    max_concurrent_inputs: int = 100
    timeout_seconds: int = 3600
    
    # Storage configuration
    vector_volume_size_gb: int = 500
    index_volume_size_gb: int = 100
    temp_volume_size_gb: int = 50
    
    # Secrets and environment
    secrets: list = None
    environment_variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.secrets is None:
            self.secrets = ["diskann-secrets", "redis-credentials"]
        
        if self.environment_variables is None:
            self.environment_variables = {
                "RUST_LOG": "info",
                "RAYON_NUM_THREADS": str(self.cpu_cores),
                "DISKANN_MAX_THREADS": str(self.cpu_cores),
                "DISKANN_CACHE_SIZE_GB": str(min(self.memory_gb // 2, 16)),
            }
    
    @classmethod
    def from_environment(cls) -> "ModalConfig":
        """Load configuration from environment variables."""
        return cls(
            cpu_cores=int(os.environ.get("MODAL_CPU_CORES", "8")),
            memory_gb=int(os.environ.get("MODAL_MEMORY_GB", "32")),
            gpu_enabled=os.environ.get("MODAL_GPU_ENABLED", "false").lower() == "true",
            concurrency_limit=int(os.environ.get("MODAL_CONCURRENCY_LIMIT", "10")),
            max_concurrent_inputs=int(os.environ.get("MODAL_MAX_CONCURRENT_INPUTS", "100")),
            timeout_seconds=int(os.environ.get("MODAL_TIMEOUT_SECONDS", "3600")),
        )

@dataclass
class DiskANNConfig:
    """DiskANN algorithm configuration."""
    
    # Core algorithm parameters
    max_degree: int = 32
    search_list_size: int = 100
    alpha: float = 1.2
    distance_metric: str = "l2"
    
    # Performance settings
    max_threads: int = 8
    cache_size_gb: int = 16
    batch_size: int = 1000
    
    # Index building parameters
    build_threads: int = 8
    build_memory_gb: int = 8
    consolidation_threshold: float = 0.2
    
    # Search parameters
    search_timeout_ms: int = 30000
    max_results: int = 1000
    enable_filtering: bool = True
    
    @classmethod
    def for_modal(cls, modal_config: ModalConfig) -> "DiskANNConfig":
        """Create DiskANN config optimized for Modal environment."""
        return cls(
            max_threads=modal_config.cpu_cores,
            cache_size_gb=min(modal_config.memory_gb // 2, 16),
            build_threads=modal_config.cpu_cores,
            build_memory_gb=modal_config.memory_gb // 4,
        )
```

### 4. Deployment Scripts

**scripts/deploy.py**:
```python
#!/usr/bin/env python3
"""Deployment script for DiskANN on Modal."""

import modal
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_app(environment: str = "production", dry_run: bool = False):
    """Deploy the DiskANN application to Modal."""
    
    logger.info(f"Deploying DiskANN to {environment} environment")
    
    if dry_run:
        logger.info("DRY RUN: No actual deployment will occur")
        return
    
    try:
        # Load the app
        from modal_app import app
        
        # Deploy based on environment
        if environment == "production":
            logger.info("Deploying to production...")
            app.deploy("diskann-prod")
        elif environment == "staging":
            logger.info("Deploying to staging...")
            app.deploy("diskann-staging")
        else:
            logger.info("Deploying to development...")
            app.deploy("diskann-dev")
        
        logger.info("Deployment completed successfully!")
        
        # Print deployment info
        deployment_url = app.web_url
        logger.info(f"Application URL: {deployment_url}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

def upload_data(data_path: str, data_type: str = "vectors"):
    """Upload data to Modal volumes."""
    
    logger.info(f"Uploading {data_type} from {data_path}")
    
    try:
        from modal_app import vector_volume, index_volume
        
        if data_type == "vectors":
            volume = vector_volume
        elif data_type == "indices":
            volume = index_volume
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Upload files
        with volume.batch_upload() as batch:
            data_dir = Path(data_path)
            for file_path in data_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(data_dir)
                    batch.put_file(str(file_path), str(relative_path))
        
        logger.info(f"Upload completed: {data_type}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)

def create_secrets():
    """Create Modal secrets for the application."""
    
    secrets_config = {
        "diskann-secrets": {
            "POSTGRES_PASSWORD": "your_postgres_password",
            "API_KEY": "your_api_key",
            "JWT_SECRET": "your_jwt_secret",
        },
        "redis-credentials": {
            "REDIS_URL": "redis://your-redis-host:6379",
            "REDIS_PASSWORD": "your_redis_password",
        },
        "monitoring-tokens": {
            "PROMETHEUS_TOKEN": "your_prometheus_token",
            "DATADOG_API_KEY": "your_datadog_key",
        },
    }
    
    for secret_name, env_vars in secrets_config.items():
        logger.info(f"Creating secret: {secret_name}")
        
        try:
            secret = modal.Secret.from_dict(env_vars)
            secret.deploy(secret_name)
            logger.info(f"Secret created: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to create secret {secret_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Deploy DiskANN to Modal")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy application")
    deploy_parser.add_argument(
        "--environment", 
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    deploy_parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform a dry run without actual deployment"
    )
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload data")
    upload_parser.add_argument("data_path", help="Path to data directory")
    upload_parser.add_argument(
        "--type", 
        choices=["vectors", "indices"],
        default="vectors",
        help="Type of data to upload"
    )
    
    # Secrets command
    secrets_parser = subparsers.add_parser("secrets", help="Create secrets")
    
    args = parser.parse_args()
    
    if args.command == "deploy":
        deploy_app(args.environment, args.dry_run)
    elif args.command == "upload":
        upload_data(args.data_path, args.type)
    elif args.command == "secrets":
        create_secrets()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### 5. Data Upload Utility

**scripts/data_upload.py**:
```python
#!/usr/bin/env python3
"""Data upload utility for Modal volumes."""

import modal
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
import struct
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_to_binary(
    vectors: np.ndarray, 
    output_path: str,
    labels: List[List[int]] = None
) -> None:
    """Convert numpy vectors to DiskANN binary format."""
    
    logger.info(f"Converting {vectors.shape[0]} vectors to binary format")
    
    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack("I", vectors.shape[0]))  # Number of vectors
        f.write(struct.pack("I", vectors.shape[1]))  # Dimension
        
        # Write vectors
        vectors.astype(np.float32).tobytes()
    
    # Write labels if provided
    if labels:
        labels_path = Path(output_path).with_suffix(".labels")
        with open(labels_path, "w") as f:
            for i, label_list in enumerate(labels):
                f.write(f"{i}\t{','.join(map(str, label_list))}\n")
        
        logger.info(f"Wrote labels to {labels_path}")
    
    logger.info(f"Binary vectors written to {output_path}")

def upload_vectors_from_numpy(
    vectors: np.ndarray,
    filename: str,
    labels: List[List[int]] = None
) -> None:
    """Upload vectors from numpy array to Modal volume."""
    
    # Convert to binary format
    temp_path = f"/tmp/{filename}"
    convert_numpy_to_binary(vectors, temp_path, labels)
    
    # Upload to Modal
    from modal_app import vector_volume
    
    with vector_volume.batch_upload() as batch:
        batch.put_file(temp_path, filename)
        
        if labels:
            labels_path = Path(temp_path).with_suffix(".labels")
            batch.put_file(str(labels_path), Path(filename).with_suffix(".labels").name)
    
    logger.info(f"Uploaded {filename} to Modal volume")

def upload_directory(
    local_path: str,
    volume_type: str = "vectors",
    file_patterns: List[str] = None
) -> None:
    """Upload entire directory to Modal volume."""
    
    from modal_app import vector_volume, index_volume
    
    volume = vector_volume if volume_type == "vectors" else index_volume
    
    if file_patterns is None:
        file_patterns = ["*.bin", "*.index", "*.labels", "*.fvecs", "*.bvecs"]
    
    local_dir = Path(local_path)
    files_to_upload = []
    
    for pattern in file_patterns:
        files_to_upload.extend(local_dir.glob(pattern))
    
    logger.info(f"Found {len(files_to_upload)} files to upload")
    
    with volume.batch_upload() as batch:
        for file_path in files_to_upload:
            relative_path = file_path.relative_to(local_dir)
            batch.put_file(str(file_path), str(relative_path))
            logger.info(f"Uploading: {relative_path}")
    
    logger.info(f"Upload completed: {len(files_to_upload)} files")

def download_sample_data() -> Tuple[np.ndarray, List[List[int]]]:
    """Download and prepare sample data for testing."""
    
    # Generate random vectors for demonstration
    np.random.seed(42)
    vectors = np.random.randn(10000, 128).astype(np.float32)
    
    # Generate random labels
    labels = []
    for i in range(len(vectors)):
        num_labels = np.random.randint(1, 4)
        label_set = np.random.choice(100, num_labels, replace=False).tolist()
        labels.append(label_set)
    
    logger.info(f"Generated {len(vectors)} sample vectors with labels")
    return vectors, labels

def main():
    parser = argparse.ArgumentParser(description="Upload data to Modal volumes")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Upload directory
    upload_parser = subparsers.add_parser("upload", help="Upload directory")
    upload_parser.add_argument("path", help="Local directory path")
    upload_parser.add_argument(
        "--type", 
        choices=["vectors", "indices"],
        default="vectors",
        help="Volume type"
    )
    upload_parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.bin", "*.index", "*.labels"],
        help="File patterns to match"
    )
    
    # Generate sample data
    sample_parser = subparsers.add_parser("sample", help="Generate and upload sample data")
    sample_parser.add_argument("--filename", default="sample_vectors.bin", help="Output filename")
    sample_parser.add_argument("--size", type=int, default=10000, help="Number of vectors")
    sample_parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        upload_directory(args.path, args.type, args.patterns)
    elif args.command == "sample":
        vectors, labels = download_sample_data()
        if hasattr(args, 'size') and hasattr(args, 'dim'):
            vectors = np.random.randn(args.size, args.dim).astype(np.float32)
            labels = [[i % 10] for i in range(args.size)]  # Simple label assignment
        
        upload_vectors_from_numpy(vectors, args.filename, labels)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

## Performance Optimization

### 6. Optimized Configuration

**config/performance.py**:
```python
"""Performance optimization configurations for Modal deployment."""

import modal
from typing import Dict, Any

# ARM64-optimized configuration
ARM64_CONFIG = modal.Image.debian_slim().run_commands([
    "apt-get update && apt-get install -y build-essential curl",
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    "source ~/.cargo/env && rustup target add aarch64-unknown-linux-gnu",
    "echo 'export RUSTFLAGS=\"-C target-cpu=native\"' >> ~/.bashrc",
]).pip_install([
    "numpy==1.24.3",
    "scipy==1.11.4", 
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
])

# GPU-optimized configuration
GPU_CONFIG = modal.Image.from_registry(
    "nvidia/cuda:12.1-devel-ubuntu22.04"
).run_commands([
    "apt-get update && apt-get install -y python3 python3-pip curl build-essential",
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    "source ~/.cargo/env && rustup target add x86_64-unknown-linux-gnu",
    "echo 'export RUSTFLAGS=\"-C target-cpu=native\"' >> ~/.bashrc",
]).pip_install([
    "cupy-cuda12x==12.3.0",
    "numpy==1.24.3",
    "scipy==1.11.4",
    "fastapi==0.104.1",
])

# High-memory configuration for large indices
HIGH_MEMORY_CONFIG = {
    "cpu": 32.0,
    "memory": 131072,  # 128GB
    "timeout": 14400,  # 4 hours
    "concurrency_limit": 1,
}

# High-throughput configuration for search workloads
HIGH_THROUGHPUT_CONFIG = {
    "cpu": 16.0,
    "memory": 32768,  # 32GB
    "timeout": 600,    # 10 minutes
    "concurrency_limit": 50,
    "allow_concurrent_inputs": 1000,
}

def get_optimal_config(workload_type: str) -> Dict[str, Any]:
    """Get optimal configuration for specific workload types."""
    
    configs = {
        "search": HIGH_THROUGHPUT_CONFIG,
        "index_build": HIGH_MEMORY_CONFIG,
        "gpu_accelerated": {**HIGH_MEMORY_CONFIG, "gpu": modal.gpu.A100()},
        "development": {
            "cpu": 4.0,
            "memory": 8192,
            "timeout": 1800,
            "concurrency_limit": 5,
        }
    }
    
    return configs.get(workload_type, HIGH_THROUGHPUT_CONFIG)
```

## Monitoring and Observability

### 7. Metrics and Logging

**src/utils/monitoring.py**:
```python
"""Monitoring utilities for Modal deployment."""

import time
import logging
import functools
from typing import Dict, Any, Callable
from contextlib import contextmanager
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class PerformanceMonitor:
    """Performance monitoring for DiskANN operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    @contextmanager
    def measure(self, operation: str, **context):
        """Context manager for measuring operation duration."""
        start = time.time()
        context["operation"] = operation
        
        try:
            logger.info("Operation started", **context)
            yield
            duration = time.time() - start
            logger.info("Operation completed", duration_ms=duration * 1000, **context)
            
            self._record_metric(operation, duration, "success")
            
        except Exception as e:
            duration = time.time() - start
            logger.error("Operation failed", 
                        duration_ms=duration * 1000, 
                        error=str(e), 
                        **context)
            
            self._record_metric(operation, duration, "error")
            raise
    
    def _record_metric(self, operation: str, duration: float, status: str):
        """Record metric for operation."""
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_duration": 0,
                "success_count": 0,
                "error_count": 0,
                "min_duration": float('inf'),
                "max_duration": 0,
            }
        
        metric = self.metrics[operation]
        metric["count"] += 1
        metric["total_duration"] += duration
        metric["min_duration"] = min(metric["min_duration"], duration)
        metric["max_duration"] = max(metric["max_duration"], duration)
        
        if status == "success":
            metric["success_count"] += 1
        else:
            metric["error_count"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, metric in self.metrics.items():
            if metric["count"] > 0:
                stats[operation] = {
                    "count": metric["count"],
                    "avg_duration_ms": (metric["total_duration"] / metric["count"]) * 1000,
                    "min_duration_ms": metric["min_duration"] * 1000,
                    "max_duration_ms": metric["max_duration"] * 1000,
                    "success_rate": metric["success_count"] / metric["count"],
                    "error_rate": metric["error_count"] / metric["count"],
                }
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "operations": stats,
        }

def async_timer(operation_name: str):
    """Decorator for timing async operations."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                logger.info("Async operation completed",
                           operation=operation_name,
                           duration_ms=duration * 1000,
                           status="success")
                
                return result
                
            except Exception as e:
                duration = time.time() - start
                
                logger.error("Async operation failed",
                            operation=operation_name,
                            duration_ms=duration * 1000,
                            status="error",
                            error=str(e))
                raise
        
        return wrapper
    return decorator

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
```

## Deployment Commands

### Quick Start

```bash
# Install Modal CLI
pip install modal

# Set up authentication
modal token set

# Clone and setup project
git clone https://github.com/atsentia/diskann-rust-arm64
cd diskann-rust-arm64/modal-deployment

# Create secrets
python scripts/deploy.py secrets

# Upload sample data
python scripts/data_upload.py sample --size 10000 --dim 128

# Deploy to development
python scripts/deploy.py deploy --environment development

# Test the deployment
curl -X POST "https://your-modal-app.modal.run/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, ...],
    "k": 10,
    "index_name": "default"
  }'
```

### Production Deployment

```bash
# Deploy to production with optimized settings
modal deploy modal_app.py --name diskann-prod

# Monitor deployment
modal app logs diskann-prod

# Scale based on demand (automatic with Modal)
# Modal handles scaling automatically based on request volume

# Update deployment
git pull origin main
modal deploy modal_app.py --name diskann-prod
```

### Data Management

```bash
# Upload large vector datasets
python scripts/data_upload.py upload ./large_vectors/ --type vectors

# Upload pre-built indices
python scripts/data_upload.py upload ./indices/ --type indices

# Monitor storage usage
modal volume list
modal volume ls diskann-vectors
```

## Cost Optimization

### Pricing Considerations

1. **Compute Costs**: 
   - ARM64 instances: ~$0.40/hour for 8 CPU, 32GB RAM
   - GPU instances: ~$4.00/hour for A100 40GB

2. **Storage Costs**:
   - Volume storage: ~$0.10/GB/month
   - Network file system: ~$0.30/GB/month

3. **Data Transfer**:
   - Ingress: Free
   - Egress: ~$0.09/GB

### Cost Optimization Strategies

```python
# Use spot instances for batch processing
@app.function(
    image=diskann_image,
    schedule=modal.Cron("0 2 * * *"),
    cpu=16.0,
    memory=65536,
    timeout=7200,
    # Enable spot pricing for cost savings
    cloud="aws",
    region="us-east-1",
)
def batch_index_building():
    """Build indices during off-peak hours with spot pricing."""
    pass

# Auto-scale down during low traffic
@app.function(
    image=diskann_image,
    cpu=4.0,
    memory=8192,
    # Scale to zero when no requests
    keep_warm=0,
    timeout=300,
)
def search_service_small():
    """Small search service that scales to zero."""
    pass
```

## Security Best Practices

### API Security

```python
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token for API access."""
    try:
        secret_key = os.environ["JWT_SECRET"]
        payload = jwt.decode(credentials.credentials, secret_key, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@api.post("/search")
async def secure_search(
    request: SearchRequest,
    token_payload: dict = Depends(verify_token)
):
    """Secured search endpoint."""
    # Rate limiting based on user
    user_id = token_payload.get("user_id")
    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return await service.search_vectors.aio(**request.dict())
```

### Data Protection

```python
# Encrypt sensitive data at rest
@app.function(
    image=diskann_image,
    secrets=[modal.Secret.from_name("encryption-keys")],
)
def encrypt_vector_data(data: bytes) -> bytes:
    """Encrypt vector data before storage."""
    from cryptography.fernet import Fernet
    
    key = os.environ["ENCRYPTION_KEY"].encode()
    f = Fernet(key)
    return f.encrypt(data)
```

This comprehensive Modal.com deployment guide provides a serverless, scalable solution for deploying DiskANN Rust with automatic scaling, cost optimization, and production-ready security features.