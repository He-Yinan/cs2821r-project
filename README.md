# cs2821r-project

## Run

### Setup

Create submodule HippoRAG (don't need to run this)

```sh
# First fork hipporag on its GitHub repo page

git submodule add https://github.com/<USERNAME>/hipporag hipporag
git commit -m "Add HippoRAG submodule"
git submodule update --init --recursive  # Get all submodules ready
```

Python environment (run this)

```sh
mamba create -n hipporag python=3.10
pip install -r hipporag/requirements.txt
```

### Test

Run VLLM server on GPU.

```sh
salloc --partition gpu_test --gpus 1 --time 02:00:00 --mem=32G --cpus-per-task=4
module load python
source activate hipporag
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B   --dtype bfloat16   --max-model-len 12288   --gpu-memory-utilization 0.94
```

Start another GPU session and run index and query.

```sh
python test_vllm.py  # Run basic test
```

If you're running on local device, forward VLLM server's port. Not recommended for experiments because getting embedding will be very slow.

```sh
ssh -L 8000:localhost:8000 <USERNAME>@login.rc.fas.harvard.edu \
  -t "ssh -L 8000:localhost:8000 <USERNAME>@<gpu_node_id>.rc.fas.harvard.edu"
```

### Experiments

Run VLLM server on GPU.

Start another GPU session and run index and query.

```sh
# Run sample dataset
python main.py --dataset sample --llm_base_url http://holygpu7c26105.rc.fas.harvard.edu:8000/v1 --llm_name Qwen/Qwen3-8B --embedding_name facebook/contriever
# Run musique dataset, estimated time more than 4 hours
python main.py --dataset musique --llm_base_url http://holygpu7c26105.rc.fas.harvard.edu:8000/v1 --llm_name Qwen/Qwen3-8B --embedding_name facebook/contriever
# Run 2wikimultihopqa dataset, estimated time around 4 hours
python main.py --dataset 2wikimultihopqa --llm_base_url http://holygpu7c26105.rc.fas.harvard.edu:8000/v1 --llm_name Qwen/Qwen3-8B --embedding_name facebook/contriever
# Run hotpotqa dataset, estimated time several hours
python main.py --dataset hotpotqa --llm_base_url http://holygpu7c26105.rc.fas.harvard.edu:8000/v1 --llm_name Qwen/Qwen3-8B --embedding_name facebook/contriever
```