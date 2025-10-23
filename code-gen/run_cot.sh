python main.py \
  --run_name "test_cot" \
  --root_dir "root" \
  --dataset_path humaneval \
  --strategy "cot" \
  --language "py" \
  --model "Qwen/Qwen2.5-Coder-14B-Instruct" \
  --pass_at_k "1" \
  --max_iters "10" \
  --verbose

# python main.py \
#   --run_name "test_cot" \
#   --root_dir "root" \
#   --dataset_path mbpp \
#   --strategy "cot" \
#   --language "py" \
#   --model "Qwen/Qwen2.5-Coder-14B-Instruct" \
#   --pass_at_k "1" \
#   --max_iters "10" \
#   --verbose
