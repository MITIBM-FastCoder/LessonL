#!/bin/bash

#SBATCH -J lessonl
##SBATCH -o slurm-logs/lessonl_serial_pareval_%j.out 
##SBATCH -e slurm-logs/lessonl_serial_pareval_%j.err

#SBATCH --nodes=1
##SBATCH -t 12:00:00
#SBATCH --exclusive
#SBATCH --partition=queue-c54xlarge

HOME2=/PATH/TO/YOUR/HOME
PYTHON_VIRTUAL_ENVIRONMENT=code-opt
CONDA_ROOT=$HOME2/miniconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
${CONDA_ROOT}/bin/conda init
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

#export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
python main.py strategy@_global_=lesson benchmark=ParEval mode=serial localhost=queue-g6e12xlarge-dy-g6e12xlarge-1 Rounds=4 k=4 temperature=0.2 reason_temperature=0.2

# python main.py strategy@_global_=cot benchmark=ParEval mode=serial localhost=localhost temperature=0.2

# python main.py strategy@_global_=mapcoder benchmark=ParEval mode=serial localhost=localhost temperature=0.2 k=3 t=5

# python main.py strategy@_global_=moa benchmark=ParEval mode=serial localhost=localhost Rounds=4 temperature=0.2 aggregator_temperature=0.7

# python main.py strategy@_global_=reflexion benchmark=ParEval mode=serial localhost=localhost max_iter=10 temperature=0.2 pass_at_k=1

# python main.py strategy@_global_=simple benchmark=ParEval mode=serial localhost=localhost temperature=0.2

# python main.py strategy@_global_=openai benchmark=ParEval mode=serial localhost=localhost temperature=0.2 model=gpt-4o trial=0
 

