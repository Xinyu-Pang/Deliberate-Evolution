# Deliberate Evolution

Implementation of **Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs**.

Deliberate Evolution (DE) is an agentic framework for sample-efficient symbolic regression with large language models. It separates candidate expression generation from search guidance, using adaptive evolutionary operators, data and residual analysis, and reflective memory to guide the search.

## Code Structure

```text
.
├── configs/                # Experiment and model settings
├── bench/                  # Benchmark data and metrics
├── methods/se/
│   ├── method.py           # Deliberate Evolution search
│   ├── evaluation.py       # Search evaluation and results
│   └── client.py           # Model API client
├── scripts/
│   ├── run_se.sh           # Experiment launcher
│   └── vllm_serve.sh       # Optional local model server
└── eval.py                 # Python entry point
```

## 1. Install the method

Run these commands from the repository root:

```bash
conda create -n sebench python=3.11 -y
conda activate sebench
python -m pip install -r requirements.txt
cp .env.example .env
```

After copying the template, **edit `.env` and fill in the values for the connection you choose below**. `eval.py` loads this file automatically. You only need the variables for your chosen option.

`requirements.txt` contains the experiment dependencies, including LaTeX parsing
and logging. Model serving is configured separately.

## 2. Configure the model connection

Choose an existing API or a local model service. The client uses OpenAI-compatible
Chat Completions, including `n`, `temperature`, and `stop`; other protocols require an adapter.

### Option A: Existing API

- Set `OPENAI_API_KEY` and `OPENAI_API_URL` in `.env`. Use the API base URL
  (usually ending in `/v1`, without `/chat/completions`).
- In `configs/se_api.yaml`, set `api_model` to your model ID and `name` to an
  output label. This config inherits the method settings from `configs/se.yaml`.

### Option B: Local vLLM (optional)

Prepare and activate a serving environment following the
[vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/),
then start the example service:

```bash
CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 bash scripts/vllm_serve.sh
```

This serves `Qwen/Qwen3-4B-Instruct-2507` at `http://localhost:10005/v1`,
downloading weights from Hugging Face if needed. Select GPU IDs with
`CUDA_VISIBLE_DEVICES` and GPU count with `TP_SIZE` (defaults to 4 if omitted).
Keep the service running. In `configs/se.yaml`, `api_model` must match the served
model; `VLLM_API_KEY` in `.env` must match the server key (example: `token-abc123`).

<details>
<summary>Optional: check the local model API</summary>

Run in another terminal and confirm the response contains `choices[0].message.content`:

```bash
curl --max-time 120 -fsS http://localhost:10005/v1/chat/completions -H "Authorization: Bearer token-abc123" -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"Reply with OK only."}],"max_tokens":32,"temperature":0}'
```

</details>

## 3. Run an experiment

Run these commands from the repository root in the experiment environment. For local deployment, keep the model server running in a separate terminal.

```bash
conda activate sebench
```

Choose the command matching the model connection configured in step 2:

| Connection | Command |
| --- | --- |
| Existing API | `CONFIG=configs/se_api.yaml bash scripts/run_se.sh` |
| Local vLLM example | `bash scripts/run_se.sh` |

Both commands evaluate all problems in LSR-Transform by default, with 100 search generations per problem.

The shell script changes to the repository root and invokes `eval.py`. To use the Python entry point directly, the default local command is equivalent to:

```bash
python eval.py --dataset lsrtransform --searcher_config configs/se.yaml
```

For an existing API, replace the config path with `configs/se_api.yaml`. Use `python eval.py --help` to see all CLI options.

## Experiment settings

Choose the YAML file with `CONFIG` and the sub-benchmark with `DATASET`. For example:

```bash
DATASET=phys_osc CONFIG=configs/se_api.yaml bash scripts/run_se.sh
```

Edit the selected YAML file to change experiment settings. `configs/se_api.yaml` inherits fields from `configs/se.yaml`; fields added to the API config override the inherited values.

| Setting | Where to set it | Meaning |
| --- | --- | --- |
| `api_model` | YAML | Model ID accepted by the configured service. |
| `max_generations` | YAML | Search generations per problem; defaults to `100`. |
| `target_original_idxes` | YAML | Problem indices to evaluate; defaults to `null` (all problems in the chosen sub-benchmark). |
| `parallel_workers` | YAML | Number of problems evaluated concurrently. |
| `name` | YAML | Label used in the output path, such as `qwen` or `api`. |
| `output_base_dir` | YAML | Output root; defaults to `logs` when omitted. |
| API credentials and URL | `.env` | Fill the variables for your chosen connection as described in step 2. |

Selecting another sub-benchmark changes the problems being evaluated; it keeps the search settings from the chosen YAML file.

For local vLLM, the client port defaults to `10005`. Set `VLLM_PORT` when using the shell script, or `--local_llm_port` when invoking `eval.py` directly, to match a server on another port. Set the server's `--port` separately. Existing API mode takes its base URL from `OPENAI_API_URL` in `.env`.

## Datasets

Dataset files are not bundled with this repository. On the first run, `bench/datamodules.py` automatically downloads the benchmark from [nnheui/llm-srbench](https://huggingface.co/datasets/nnheui/llm-srbench), including `lsr_bench_data.hdf5`, and stores it in the Hugging Face cache.

Ensure internet access to Hugging Face and sufficient disk space for the initial download. Subsequent runs reuse cached data, but the loader may still access Hugging Face to check for updates.

### Select a sub-benchmark

Set `DATASET` to choose one sub-benchmark for a run. The default is `lsrtransform`.
Use the same value with `--dataset` when running `eval.py` directly.

| Sub-benchmark | `DATASET` / `--dataset` value |
| --- | --- |
| LSR-Transform (default) | `lsrtransform` |
| Physics Oscillator | `phys_osc` |
| Materials Science | `matsci` |
| Biological Population Growth | `bio_pop_growth` |
| Chemical Reaction Kinetics | `chem_react` |

For example, select Physics Oscillator with the local vLLM configuration:

```bash
DATASET=phys_osc CONFIG=configs/se.yaml bash scripts/run_se.sh
```

For an existing API, use `CONFIG=configs/se_api.yaml` in the same command.
Changing `DATASET` only selects the data; the selected configuration still controls
the search settings and which problems are evaluated (`target_original_idxes`).
See [Experiment settings](#experiment-settings) for these options.

## Results

For a new run, results are written under the project root when using
`scripts/run_se.sh`:

```text
logs/<dataset_name>/<config_name>/<MMDD_HHMM>/
```

With the default LSR-Transform command and config name `qwen`, this is
`logs/LSR_Transform/qwen/<MMDD_HHMM>/`. The timestamp contains month, day, hour,
and minute. Avoid starting runs with the same dataset and config name within
the same minute, because they will use the same directory.

With the existing API example and its default name, results are saved to `logs/LSR_Transform/api/<MMDD_HHMM>/`.

The run directory contains:

- `experiment_config.json`: the saved experiment configuration.
- `results.jsonl`: results recorded as problems are processed.
- `all_results.json`: the aggregate results written at the end of evaluation.
- `search_logs/`: detailed search logs.

Set `output_base_dir` in `configs/se.yaml` to change the output root (default:
`logs`). Relative output paths for direct `python eval.py` commands are resolved
from the current working directory.

## Reference environment

Reference versions for local model serving: **Qwen3-4B-Instruct-2507**, **vLLM 0.8.5**,
and **Transformers 4.51.1**.
These versions are for reference; other compatible models and services can be used.
`requirements-vllm.txt` is an optional version-pinned serving example, not a method dependency.

## Acknowledgments

We thank [LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench)
for its benchmark and evaluation infrastructure. See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for attribution and license terms.

## v1 status

This is the v1 release. Actual dimensional analysis is not implemented yet.
The `dimensional_analysis` tool is a placeholder that always returns `Valid.`
to preserve the original tool interaction. This response does not certify
that an expression is dimensionally consistent. We plan to add actual
dimensional analysis soon.
