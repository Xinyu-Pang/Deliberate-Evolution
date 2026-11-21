import os
import sys

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf

from bench.datamodules import get_datamodule

from dotenv import load_dotenv
load_dotenv()


parser = ArgumentParser()
parser.add_argument('--searcher_config', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ds_root_folder', type=str, default=None)
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--problem_name', type=str, default=None)
parser.add_argument('--local_llm_port', type=int, default=None)
args = parser.parse_args()

now = datetime.now()
now_str = now.strftime("%m%d_%H%M")

dm = get_datamodule(name=args.dataset, root_folder=args.ds_root_folder)
dm.setup()

# Load searcher configuration from yaml file
def _apply_base_config(cfg, config_path: str):
    base_ref = cfg.pop("base_config", None)
    if not base_ref:
        return cfg
    base_cfg = OmegaConf.load(str((Path(config_path).parent / base_ref).resolve())) if isinstance(base_ref, str) else OmegaConf.create(base_ref)
    return OmegaConf.merge(base_cfg, cfg)

searcher_cfg = OmegaConf.load(args.searcher_config)
searcher_cfg = _apply_base_config(searcher_cfg, args.searcher_config)

def get_cfg_value(cfg, key: str, default=None):
    if OmegaConf.is_config(cfg):
        value = OmegaConf.select(cfg, key)
        return value if value is not None else default
    else:
        return getattr(cfg, key, default)
# Set up output directories for logging results
# If not resuming from previous run:
#   - Create new timestamped output directory under logs/{dataset}/{searcher}/
# If resuming:
#   - Use existing directory specified by resume_from argument
# Creates search_logs subdirectory and sets searcher's log path
if args.resume_from is None:
    cfg_name = get_cfg_value(searcher_cfg, "name")
    output_base_dir = get_cfg_value(searcher_cfg, "output_base_dir", "logs")
    output_path = Path(f"{output_base_dir}/{dm.name}/{cfg_name}/{now_str}")
    output_path.mkdir(parents=True, exist_ok=True)
else:
    output_path = Path(args.resume_from)
searcher_log_path = output_path / "search_logs"
searcher_log_path.mkdir(exist_ok=True, parents=True)


# Set API key based on the searcher configuration type
# vllm: Local VLLM server
# openai: OpenAI API
api_type = get_cfg_value(searcher_cfg, "api_type", "")
if api_type == "vllm":
    api_key = os.environ['VLLM_API_KEY']
    searcher_cfg.api_key = api_key
    api_url = get_cfg_value(searcher_cfg, "api_url", "")
    searcher_cfg.api_url = api_url.format(args.local_llm_port or 10005)
elif api_type == "openai":
    api_key = os.environ['OPENAI_API_KEY']
    searcher_cfg.api_key = api_key
    searcher_cfg.api_url = os.environ['OPENAI_API_URL']
else:
    api_key = None


class_name = get_cfg_value(searcher_cfg, "class_name", "")
if class_name != "SESearcher":
    raise ValueError(f"Only SESearcher is supported in this project. Got: {class_name}")

# Filter problems based on command line arguments:
# - If problem_name is specified, only keep that specific problem
# - Otherwise use all problems from the dataset
problems = dm.problems
if args.problem_name is not None:
    problems = list(filter(lambda p: p.equation_idx == args.problem_name, problems))

# Set original_idx for each problem based on the order in the problems list
# This index will remain fixed even if some problems are skipped later
for idx, problem in enumerate(problems):
    problem.original_idx = idx

print(f"Total number of problems: {len(problems)}")


    # Evaluate problems with the configured searcher and save results.
verbose_timing = False

# Determine if we should use parallel processing
parallel_workers = int(get_cfg_value(searcher_cfg, "parallel_workers", 1))
use_parallel = parallel_workers > 1
print(f"SE method - use_parallel: {use_parallel}, parallel_workers: {parallel_workers}")

# Get target_original_idxes from config if specified
target_original_idxes = get_cfg_value(searcher_cfg, "target_original_idxes")
if target_original_idxes is not None:
    print(f"SE method - target_original_idxes specified: {target_original_idxes}")

# Import run_se_evaluation and create_se_searcher
sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
from methods.se.evaluation import run_se_evaluation, create_se_searcher

def create_se_searcher_factory():
    """Create a new SESearcher instance."""
    return create_se_searcher(searcher_cfg)

if use_parallel:
    run_se_evaluation(problems,
                     searcher_cfg,
                     output_path,
                     verbose_timing=verbose_timing,
                     parallel_workers=parallel_workers,
                     searcher_factory=create_se_searcher_factory,
                     target_original_idxes=target_original_idxes)
else:
    run_se_evaluation(problems,
                     searcher_cfg,
                     output_path,
                     verbose_timing=verbose_timing,
                     parallel_workers=1,
                     searcher_factory=create_se_searcher_factory,
                     target_original_idxes=target_original_idxes)
