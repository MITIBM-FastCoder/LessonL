# ParEval-PolyBench-Code-Opt


This directory contains the (modified) ParEval and PolyBench code optimization benchmarks. It started from a fork of the Parallel Code Evaluation (ParEval) Benchmark for evaluating the ability of Large Language Models to write parallel code. We modified the benchmark such that it suites the purpose of code optimization. We also added a modified version of the PolyBench benchmark in this directory.

The original ParEval repo is located at https://github.com/parallelcodefoundry/ParEval and the original PolyBench repo is located at https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.

## Overview

The organization of the repo is as follows.

- `prompts/` -- the prompts in ParEval alongside some utility scripts
- `generate/` -- scripts for generating LLM outputs
- `drivers/` -- scripts to evaluate LLM outputs
- `analysis/` -- scripts to analyze driver results and compute metrics

Each subdirectory has further documentation on its contents. The general
workflow is to use `generate/generate.py` to generate LLM outputs, run
`drivers/run-all.py` to evaluate outputs, and `analysis/metrics.py` to
post-process the results.
