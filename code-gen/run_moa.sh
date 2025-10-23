export HF_HOME="YOUR_HF_HOME_PATH"
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
#python main.py \
#  --run_name "test_moa" \
#  --root_dir "root" \
#  --dataset_path humaneval \
#  --strategy "moa" \
#  --language "py" \
#  --model "3_models_+_gpt4o" \
#  --pass_at_k "1" \
#  --max_iters "4" \
#  --verbose

python main.py \
  --run_name "test_moa" \
  --root_dir "root" \
  --dataset_path mbpp \
  --strategy "moa" \
  --language "py" \
  --model "3_models_+_gpt4o" \
  --pass_at_k "1" \
  --max_iters "4" \
  --verbose
