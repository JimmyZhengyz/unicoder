# UniCoder

This repository hosts public resources for **UniCoder: Unified Visual-to-Code Generation via Symbolic Rewards and Reference-Guided Code Optimization**.

## Authors

Yaozhi Zheng\*, Yilei Jiang\*, Manyuan Zhang, Yuxuan Wan, Kaituo Feng, Tianshuo Peng, Bo Zhang, Michael R. Lyu, Xiangyu Yue†

(\* Equal contribution, † Corresponding author)

The Chinese University of Hong Kong &nbsp;|&nbsp; Shanghai Artificial Intelligence Laboratory

## Overview

The current release focuses on evaluation scripts for visual-to-code benchmarks. UniCoder model weights, training data processing, and full UniCoder training scripts are planned for a later release.

## Release Status

- Evaluation pipeline: available for ChartMimic and UniSVG.
- Model weights: not included in this release.
- UniCoder training scripts: not part of the current release scope.
- Benchmark data: not redistributed here; please download it from the official benchmark sources.

## Repository Structure

- `eval/`: Evaluation release for ChartMimic and UniSVG.
  - `simple_eval/`: Unified evaluation entry point and model wrappers.
    - `run_pipeline.py`: Runs ChartMimic and UniSVG evaluation pipelines.
    - `models.py`: Local/API model wrappers for Qwen, OpenAI, Gemini, Claude, and DashScope Qwen APIs.
    - `requirements.txt`: Python dependencies for the evaluation wrapper.
- `eval/ChartMimic/`: Modified ChartMimic benchmark resources and evaluators.
- `eval/unisvg.github.io/`: Modified UniSVG benchmark resources and evaluation scripts.
- `train/`: Reserved for future UniCoder training-code release.

Benchmark-specific README files remain inside `eval/` because they document the adapted benchmark components. The root README provides the project-level release instructions.

## Requirements

- Python 3.9+
- Linux or macOS for API-based evaluation. The UniSVG evaluation uses `signal.SIGALRM`, so Windows is not currently supported.
- CUDA-capable GPUs are recommended for local Qwen model evaluation.

## Installation

```bash
pip install -r eval/simple_eval/requirements.txt
```

For macOS users, SVG rendering may additionally require Cairo and Pango:

```bash
brew install cairo pango
```

## API Keys

API-based models read credentials from environment variables. Do not hard-code keys in source files.

Set only the credentials required by the model APIs you plan to use:

```bash
export DASHSCOPE_API_KEY=your_dashscope_key
export OPENAI_API_KEY=your_openai_key
export OPENAI_BASE_URL=your_openai_base_url_optional
export GOOGLE_API_KEY=your_google_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

## Data Preparation

Download benchmark data from the official sources and place it under the expected paths:

- ChartMimic data: `eval/ChartMimic/dataset/`
- UniSVG data: `eval/unisvg.github.io/dataset/`

Please follow the original benchmark licenses and data-use requirements.

## Quick Start

Run commands from the repository root.

### ChartMimic

```bash
python eval/simple_eval/run_pipeline.py \
  --task chartmimic \
  --model qwen_api \
  --model_path qwen-vl-max \
  --judge_model qwen_api \
  --judge_path qwen-vl-plus
```

### UniSVG

```bash
python eval/simple_eval/run_pipeline.py \
  --task unisvg \
  --model qwen_api \
  --model_path qwen-vl-max \
  --data_file eval/unisvg.github.io/dataset/test.json \
  --image_dir eval/unisvg.github.io/dataset/png
```

Use `--num` for a small smoke test:

```bash
python eval/simple_eval/run_pipeline.py \
  --task unisvg \
  --model qwen_api \
  --model_path qwen-vl-max \
  --data_file eval/unisvg.github.io/dataset/test.json \
  --image_dir eval/unisvg.github.io/dataset/png \
  --num 5
```

## Citation

```bibtex
@article{zheng2026unicoder,
  title={Unified Visual-to-Code Generation via Symbolic Rewards and Reference-Guided Code Optimization},
  author={Zheng, Yaozhi and Jiang, Yilei and Zhang, Manyuan and Wan, Yuxuan and Feng, Kaituo and Peng, Tianshuo and Zhang, Bo and Lyu, Michael R. and Yue, Xiangyu},
  journal={arXiv preprint},
  year={2026},
  note={Yaozhi Zheng and Yilei Jiang contributed equally; Xiangyu Yue is the corresponding author.}
}
```

## Acknowledgements

This evaluation release builds on ChartMimic and UniSVG. Please also cite and follow the licenses of the original benchmark repositories when using their code or data.