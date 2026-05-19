# ChartMimic Evaluation Components

This directory contains evaluation components adapted from [ChartMimic](https://github.com/ChartMimic/ChartMimic) (Apache-2.0 License).

Only the evaluator modules used by UniCoder's unified evaluation pipeline are included here. For the full ChartMimic benchmark, dataset, and original evaluation scripts, please visit the [official repository](https://github.com/ChartMimic/ChartMimic).

## Included Components

- `chart2code/utils/evaluator/` — Low-level evaluators (text, color, chart type, layout)
- `chart2code/prompts/` — Generation and evaluation prompt templates
- `eval_configs/` — Configuration files for the evaluation pipeline

## Data

Download ChartMimic evaluation data from the official source:

```bash
mkdir dataset
wget https://huggingface.co/datasets/ChartMimic/ChartMimic/resolve/main/dataset-iclr.tar.gz
tar -xzvf dataset-iclr.tar.gz -C dataset
```

## Citation

If you use ChartMimic evaluation components, please cite the original work:

```bibtex
@inproceedings{yang2024chartmimic,
  title={ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation},
  author={Yang, Cheng and Shi, Chufan and Liu, Yaxin and Shui, Bo and Wang, Junjie and Jing, Mohan and Xu, Linran and Zhu, Xinyu and Li, Siheng and Zhang, Yuxiang and others},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
