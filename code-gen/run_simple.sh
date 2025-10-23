export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" # if needed


python main.py \
  --run_name "test_simple" \
  --root_dir "root" \
  --dataset_path humaneval  \
  --strategy "simple" \
  --language "py" \
  --model "Qwen/Qwen2.5-Coder-14B-Instruct" \
  --pass_at_k "1" \
  --max_iters "10" \
  --verbose


python main.py \
  --run_name "test_simple" \
  --root_dir "root" \
  --dataset_path mbpp \
  --strategy "simple" \
  --language "py" \
  --model "Qwen/Qwen2.5-Coder-14B-Instruct" \
  --pass_at_k "1" \
  --max_iters "10" \
  --verbose



