# Third-party notices

## Giga Embeddings `0826`

This project is an independent MLX port derived from the architecture and
model artifacts published by `ai-sage`:

| Model | Pinned source revision |
|---|---|
| [480M](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-480M-0826) | `2d0c1a92716eef0e5b6972df85b5883eb5b4f57a` |
| [3B](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-3B-0826) | `ed7db5c91b900b39381b27b6e9c0a3d31137cd29` |
| [10B-A1.8B](https://huggingface.co/ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826) | `1cb3ad3374dbf0eb9130546ca38b262de5f60287` |

All three upstream model cards declare the MIT License. The converted Q8
weights remain derived model artifacts and are distributed separately on
Hugging Face. Model weights are not included in the Python package or GitHub
repository.

## Runtime libraries

The implementation uses [MLX](https://github.com/ml-explore/mlx),
[MLX-LM](https://github.com/ml-explore/mlx-lm),
[Transformers](https://github.com/huggingface/transformers),
[huggingface_hub](https://github.com/huggingface/huggingface_hub), and
[safetensors](https://github.com/huggingface/safetensors). Their packages and
licenses are distributed by their respective authors.

## Public evaluation datasets

Only pinned identifiers, revisions, selection rules, hashes, and compact
aggregate results are stored in this repository. Dataset contents remain
outside Git.

| Dataset | Recorded revision | License recorded by source card |
|---|---|---|
| [RuBQ Retrieval](https://huggingface.co/datasets/ai-forever/rubq-retrieval) | `e19b6ffa60b3bc248e0b41f4cc37c26a55c2a67b` | CC BY-SA 4.0 |
| [SciFact](https://huggingface.co/datasets/mteb/scifact) | `d56462d0e63a25450459c4f213e49ffdb866f7f9` | CC BY-NC 4.0 |
| [CosQA](https://huggingface.co/datasets/CoIR-Retrieval/cosqa) | `bc5efb7e9d437246ce393ed19d772e08e4a79535` | MIT |
| [Multilingual NanoNQ Retrieval](https://huggingface.co/datasets/mteb/MultilingualNanoNQRetrieval) | `4104e3376fe243f0bd4845e204b01c0fc3f7d1d7` | CC BY 4.0 |
| [RuSTS Benchmark](https://huggingface.co/datasets/ai-forever/ru-stsbenchmark-sts) | `7cf24f325c6da6195df55bef3d86b5e0616f3018` | CC BY-SA 4.0 |
| [RuToxic OKMLCUP](https://huggingface.co/datasets/mteb/ru_toxic_okmlcup) | `729025d2cfa68fcbc587ea80014a42d569cd9048` | not specified by the recorded source card |

The final row is used only for the documented local classification probe; this
repository does not redistribute its contents or infer a license.
