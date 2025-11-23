# GPU Requirements for Qwen Models on PLI H100 Cluster

## H100 GPU Specifications
- **Memory per GPU**: 80 GB HBM2e
- **Cluster**: PLI cluster

## Model GPU Requirements

### 14B Model
- **Memory needed**: ~50-70 GB minimum, 80-96 GB recommended
- **GPUs required**: **1 GPU** (80 GB available)
- **Status**: Should fit comfortably on 1 GPU

### 32B Model
- **Memory needed**: ~100-120 GB minimum, 128-160 GB recommended
- **GPUs required**: **2 GPUs** (160 GB total)
- **Status**: Needs 2 GPUs for safe operation

### 72B Model
- **Memory needed**: ~190-240 GB minimum, 256 GB recommended
- **GPUs required**: **4 GPUs** (320 GB total) for recommended, **3 GPUs** (240 GB) for minimum
- **Status**: Needs 3-4 GPUs (using 4 for safety margin)

## Notes

- Models use `device_map="auto"` which automatically distributes the model across available GPUs
- Memory estimates include model weights, KV cache, activations, and system overhead
- Actual usage may vary based on sequence length and batch size
- Using more GPUs than strictly necessary provides headroom for longer sequences

