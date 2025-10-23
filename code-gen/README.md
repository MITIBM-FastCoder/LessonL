# LessonL for Code Generation

The code in this directory runs the HumanEval, MBPP, HumanEval+, and MBPP+ benchmarks using [EvalPlus](https://evalplus.github.io).

## 1. Set Up Environment

```bash
conda create -n code-gen python=3.12 -y
pip install -r requirements
```

## 2. Run and Evaluate

1. Initialize models using `vllm serve` and API keys.
```bash
# Start server for deepseek-coder-7b-instruct-v1.5 on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-coder-7b-instruct-v1.5 --api-key <token-0> --host 0.0.0.0 --port 8001 > server1.log 2>&1 &

# Start server for Qwen2.5-Coder-7B-Instruct on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --api-key <token-1> --host 0.0.0.0 --port 8002 > server2.log 2>&1 &

# Start server for Qwen2.5-Coder-14B-Instruct on GPU 2
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --api-key <token-2> --host 0.0.0.0 --port 8003 > server3.log 2>&1 &
```

2. Run `LessonL` or other frameworks. For instance, for `LessonL`, do:
```bash
# Start VLLM servers first on ports 8001, 8002, 8003
export HF_HOME=</path/to/cache>
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --run_name "test_multi_lessons_opensource" \
  --root_dir "root" \
  --dataset_path humaneval \
  --strategy "lesson-opensource" \
  --language "py" \
  --model "3-opensource-models" \
  --pass_at_k "1" \
  --max_iters "4" \
  --verbose
```
Check `run_lessons_opensource.sh` (LessonL) and `run_*.sh` (other multi-agent frameworks) for details of the command line arguments.

3. Copy the `.jsonl` file generated at step 2 to the `evalplus_folder`. For instance,
```bash
mkdir -p evalplus_folder
cp ./root/test_multi_lessons_opensource/humaneval_lesson-opensource_4_3-opensource-models_pass_at_k_1_py.jsonl ./evalplus_folder
```

4. Go to the `evalplus_folder`, sanitize the `.jsonl` file and then evaluate.
```bash
cd evalplus_folder

# For simplicity, we use the name `samples.jsonl` in place of `humaneval_lesson-opensource_4_3-opensource-models_pass_at_k_1_py.jsonl`
evalplus.sanitize --samples samples.jsonl
# Sanitized code will be produced to `samples-sanitized.jsonl`
evalplus.evaluate --dataset humaneval --samples samples-sanitized.jsonl
```
For more details about the usage of `evalplus`, see the official github [repo](https://github.com/evalplus/evalplus/blob/master/docs/cli.md#code-post-processing).

## Acknowledgment

This repository is adapted from [LATS](https://github.com/lapisrocks/LanguageAgentTreeSearch/tree/main/programming) and [Reflexion](https://github.com/noahshinn/reflexion/tree/main/programming_runs).

