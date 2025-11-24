# cs2821r-project

## Run

### Setup

Create submodule HippoRAG (don't need to run this)

```sh
# First fork HippoRAG on its GitHub repo page

git submodule add https://github.com/<USERNAME>/HippoRAG HippoRAG
git commit -m "Add HippoRAG submodule"
git submodule update --init --recursive  # Get all submodules ready
```

Python environment (run this)

```sh
mamba create -n hipporag python=3.10
pip install -r HippoRAG/requirements.txt
```

### Test

Run VLLM server on GPU.

```sh
salloc --partition gpu_test --gpus 1 --time 00:30:00 --mem=32G --cpus-per-task=4
module load python
source activate hipporag
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B   --dtype bfloat16   --max-model-len 12288   --gpu-memory-utilization 0.94
```

Running `test_vllm.py` requires more than 20GB of GPU memory, so use CPU partition or local.

Setup python environment again on CPU only device.

```sh
python test_vllm.py
```

If you're running on local device, forward VLLM server's port.

```sh
ssh -L 8000:localhost:8000 <USERNAME>@login.rc.fas.harvard.edu \
  -t "ssh -L 8000:localhost:8000 <USERNAME>@<gpu_node_id>.rc.fas.harvard.edu"
```