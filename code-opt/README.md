# LessonL for Code Optimization

The code in this directory runs the ParEval and PolyBench benchmarks. These benchmarks were modified from the original versions to suite the purpose of code optimization (see Appendix F of the [LessonL paper](https://arxiv.org/pdf/2505.23946)). The modified benchmarks are hosted under the `code-opt/ParEval-PolyBench-Code-Opt/drivers/cpp/benchmarks` directory.

## 1. Set Up Environment
```bash
conda create -n code-opt python=3.12 -y
conda activate code-opt
conda install -c conda-forge gcc=14.2.0 gxx=14.2.0 -y # install gcc with 14.2.0
pip install -r requirements
```

Next, compile the C++ benchmarking codes.

```bash
cd ParEval-PolyBench-Code-Opt/drivers/cpp
make
```

## 2. Host vLLM Servers

In the paper's setup, we host LLM agents on an Amazon EC2 instance (g6e12xlarge) through vLLM. This instance contains 4x GPUs, sufficient for hosting three (even four) agents. To request an g6e12xlarge instance, do:
```bash
srun --partition=queue-g6e12xlarge --exclusive --pty bash -i
```

Then, start LLM servers using API keys:
```bash
export HF_HOME=<YOUR/HUGGINGFACE/PATH>

# Start server for deepseek-coder-7b-instruct-v1.5 on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-coder-7b-instruct-v1.5 --api-key <token-0> --host 0.0.0.0 --port 8001 > server1.log 2>&1 &

# Start server for Qwen2.5-Coder-7B-Instruct on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --api-key <token-1> --host 0.0.0.0 --port 8002 > server2.log 2>&1 &

# Start server for Qwen2.5-Coder-14B-Instruct on GPU 2
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --api-key <token-2> --host 0.0.0.0 --port 8003 > server3.log 2>&1 &
```

## 3. Run Experiments

First, get the hostname:
```bash
hostname
```

Then, run an experiment:
```bash
python main.py strategy@_global_=<strategy> benchmark=<benchmark> mode=<mode> localhost=<hostname> <strategy_params>=<strategy_params>
```

For instance, to run the LessonL strategy on the ParEval benchmark in the serial mode, do:
```bash
python main.py strategy@_global_=lesson benchmark=ParEval mode=serial localhost=<hostname> Rounds=4 k=4 temperature=0.2 reason_temperature=0.2
```

Note: Because code timing is sensitive to the system load, we request a c5.4xlarge instance with exclusive access to ensure resource isolation from the g6e.12xlarge instance hosting the LLMs. The experiment is done on the c5.4xlarge instance in a batch mode. For this, we prepare a script `submit-job.sh` and use the following slurm command to submit the experiment as a job:
```bash
sbatch submit-job.sh
```

### General Arguments

We use [hydra](https://hydra.cc) to manage configurations. See `code-opt/config/base.yaml` for default values.

| Argument    | Explanation | Choices | 
| -------- | ---------- | ---------- | 
| strategy@\_global\_  | Multi-agent strategy | lesson, simple, cot, reflexion, moa, mapcoder, openai. Default: lesson |
| benchmark  | Benchmark | ParEval, PolyBench Default: ParEval |
| mode  | Benchmark mode | serial, OpenMP. Default: serial |
| localhost | Host name | Default: queue-g6e12xlarge-dy-g6e12xlarge-1 |
| logging.loggers.main_logger.level | Logging verbosity of the experiment | INFO, DEBUG. Default: INFO|
| logging.loggers.class_logger.level | Logging verbosity of the agents | INFO, DEBUG. Default: INFO|

### Strategy-Specific Arguments

For **LessonL**, see `code-opt/config/strategies/lesson.yaml` for default values.

| Argument    | Explanation | Values | 
| -------- | ---------- | ---------- | 
| Rounds | Number of collaboration rounds | Default: 4 |
| k | Number of selected lessons in each round | Default: 4 |
| temperature | LLM temperature for code generation | Default: 0.2 |
| reason_temperature | LLM temperature for lesson generation | Default: 0.2 |

For other strategies, see `submit-job.sh` for the choice of strategy-specific values.

