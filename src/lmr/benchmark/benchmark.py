import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional
import torch
import traceback
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Benchmark:
    def __init__(self, benchmark_config, model, tokenizer, checkpointing, output_dir, device=None):
        self.benchmark_config = benchmark_config

        self.model = model
        
        self.tokenizer = tokenizer
        self.checkpointing = checkpointing
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = Path(f"{benchmark_config.prefix if benchmark_config.prefix is not None else ''}{output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lm_eval_dir = Path(__file__).resolve().parent / "lm-evaluation-harness"
        self.results_dir = self.output_dir / "results"
        self.summary_path = self.output_dir / "summary.json"
        
        self.checkpointing.load_model_states(benchmark_config.checkpoint_mode)
        
        self._init_repo()

    def _init_repo(self):
        if not self.lm_eval_dir.exists():
            print("Cloning lm-evaluation-harness...")
            subprocess.run([
                "git", "clone", "https://github.com/EleutherAI/lm-evaluation-harness", str(self.lm_eval_dir)
            ], check=True)

            print("Installing lm-evaluation-harness...")
            subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=str(self.lm_eval_dir),
                check=True
            )
        else:
            print("lm-evaluation-harness already installed.")
            

    def run_benchmarks(self):
        sys.path.insert(0, str(self.lm_eval_dir))

        from lm_eval.evaluator import simple_evaluate
        from .benchmark_wrapper import BenchmarkWrapper

        
        summary = self._load_existing_summary()
        status_report = {}

        model = self.model
        if self.benchmark_config.compile:
            print(f"Compiling model with mode: {self.benchmark_config.compile_mode}")
            model = torch.compile(model, mode=self.benchmark_config.compile_mode)

        wrapped_model = BenchmarkWrapper(
            model,
            self.tokenizer,
            device=self.device,
            batch_size=self.benchmark_config.batch_size,
            dtype=getattr(torch, self.benchmark_config.precision)
        )

        new_results = {}
        for task in self.benchmark_config.tasks:
            try:
                if task in summary:
                    print(f"Skipping {task}, already benchmarked.")
                    status_report[task] = {"status": "cached", "result": summary[task]}
                    continue

                print(f"Running benchmark for task: {task}")
                result = simple_evaluate(
                    model=wrapped_model,
                    tasks=[task],
                    num_fewshot=self.benchmark_config.num_fewshot,
                    batch_size=self.benchmark_config.batch_size,
                    limit=self.benchmark_config.limit if hasattr(self.benchmark_config, "limit") else None,
                    bootstrap_iters=self.benchmark_config.bootstrap_iters,
                    device=self.device,
                    log_samples=False
                )
                
                self.results_dir.mkdir(parents=True, exist_ok=True)
                
                def make_json_safe(obj):
                    if isinstance(obj, torch.device):
                        return str(obj)
                    elif isinstance(obj, dict):
                        return {k: make_json_safe(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_json_safe(v) for v in obj]
                    elif isinstance(obj, tuple):
                        return tuple(make_json_safe(v) for v in obj)
                    else:
                        return obj
                
                with open(self.results_dir / f"{task}.json", "w") as f:
                    json.dump(make_json_safe(result), f, indent=2) 

                task_result = result["results"].get(task, {})
                summary[task] = task_result
                new_results[task] = task_result
                status_report[task] = {"status": "completed", "result": task_result}

            except Exception as e:
                print(f"⚠️  Warning: Benchmark for task '{task}' failed with error: {e}")
                traceback.print_exc()
                status_report[task] = {"status": "failed", "error": str(e)}

        self._save_summary(summary)
        self._print_status_report(status_report)

    def _load_existing_summary(self) -> dict:
        if self.summary_path.exists():
            with open(self.summary_path, "r") as f:
                return json.load(f)
        return {}

    def _save_summary(self, summary: dict):
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def _print_status_report(self, report):
        def is_metric_key(k):
            return any(key in k.lower() for key in ["acc", "f1", "loss", "perplexity", "exact", "score"])

        def format_metric(k, v):
            
            clean_key = k.replace(",none", "")
            
            if "norm" in k:
                return None
            
            if isinstance(v, dict) and "value" in v:
                val = v["value"]
                stderr = v.get("stderr", None)
                if stderr is not None:
                    return f"{clean_key}: {val:.5f} ± {stderr:.5f}"
                else:
                    return f"{clean_key}: {val:.5f}"
            elif isinstance(v, (float, int)):
                return f"{clean_key}: {v:.5f}"
            return None

        print("\n=== Benchmark Summary Report ===")
        for task, info in report.items():
            status = info["status"]
            if status in {"completed", "cached"}:
                label = "✅" if status == "completed" else "📄"
                metrics = info["result"]
                filtered_metrics = {
                    k: v for k, v in metrics.items()
                    if is_metric_key(k)
                }
                formatted = [
                    format_metric(k, v)
                    for k, v in filtered_metrics.items()
                    if format_metric(k, v) is not None
                ]
                metric_str = ", ".join(formatted)
                print(f"{label} {task}: {status} | {metric_str}")
            elif status == "failed":
                print(f"❌ {task}: {status} | Error: {info['error']}")