export HF_HOME="YOUR_HF_HOME_PATH"
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

# python main.py \
#   --run_name "test_multi_lessons_opensource" \
#   --root_dir "root" \
#   --dataset_path mbpp \
#   --strategy "lesson-opensource" \
#   --language "py" \
#   --model "3-opensource-models" \
#   --pass_at_k "1" \
#   --max_iters "4" \
#   --verbose
