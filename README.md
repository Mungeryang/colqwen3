# ColQwen3: Efficient Document Retrieval with Qwen3-VL-2B-Instruct 👀

### This is the v0.1 version trained with batch_size 32 for 1 epoch and with the updated pad token

> Welcome to follow and star! ⭐️⭐️⭐️

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/ViDoRe_Benchmark-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/goodman2001/colqwen3-v0.1)

## Associated Paper

This repository contains the code used for training the **ColQwen3**, which is a vision retriever based on the ColBERT architecture and the **Qwen3-VL-2B** model.

## Introduction

**ColQwen3** is a model based on a novel model architecture and training strategy based on Vision Language Models (VLMs) to efficiently index documents from their visual features.
It is a [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) extension that generates [ColBERT](https://arxiv.org/abs/2004.12832)- style multi-vector representations of text and images. 
It was introduced in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) and first released in [my repository](https://github.com/Mungeryang/colqwen3)

![ColPali Architecture](assets/colpali_architecture.webp)

## Version specificity

This model takes dynamic image resolutions in input and does not resize them, changing their aspect ratio as in ColPali.
Maximal resolution is set so that 768 image patches are created at most. Experiments show clear improvements with larger amounts of image patches, at the cost of memory requirements.

This version is trained with `colpali-engine==0.3.14`.

Data is the same as the ColPali data described in the paper.

## Setup

All models are trained for **only 1 epoch** on the train set. Unless specified otherwise, we train models in `bfloat16` format, use low-rank adapters ([LoRA](https://arxiv.org/abs/2106.09685)) 
with `alpha=32`  and `r=32` on the transformer layers from the language model, 
as well as the final randomly initialized projection layer, and use a `paged_adamw_8bit` optimizer. 
We train on 2*NVIDIA A100 80GB GPUs setup with data parallelism, a learning rate of 5e-5 with linear decay with 2.5% warmup steps, and a batch size of 32.

We used Python 3.10 and PyTorch 2.4 to train and test our models, but the codebase is compatible with Python >=3.9 and recent PyTorch versions. To install the package, run:

```bash
pip install colpali-engine # from PyPi
pip install git+https://github.com/illuin-tech/colpali # from source
```

Mac users using MPS with the ColQwen models have reported errors with torch 2.6.0. These errors are fixed by downgrading to torch 2.5.1.

> [!WARNING]
> For ColPali versions above v1.0, make sure to install the `colpali-engine` package from source or with a version above v0.2.0.


## Usage 🤗

Make sure `colpali-engine` is installed from source or with a version superior to 0.3.4.
`transformers` version must be >= **4.57.1**.(compatible with Qwen3-VL interface)

```bash
pip install git+https://github.com/Mungeryang/colqwen3
```

```python
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen3, ColQwen3Processor

model = ColQwen3.from_pretrained(
    "goodman2001/colqwen3-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen3Processor.from_pretrained("goodman2001/colqwen3-v0.1")

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```


### Benchmarking

To benchmark ColQwen3 on the [ViDoRe leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard), use the [`mteb`](https://github.com/illuin-tech/vidore-benchmark) package.


### Training

To keep a lightweight repository, only the essential packages were installed. In particular, you must specify the dependencies to use the training script for ColPali. You can do this using the following command:

```bash
pip install -r requirements.txt

pip install mteb==1.39.7

pip install "colpali-engine[train]"
```

All the model configs used can be found in `scripts/configs/` and rely on the [configue](https://github.com/illuin-tech/configue) package for straightforward configuration. They should be used with the `train_colbert.py` script.

<details>
<summary><strong>🔽 Example : Local training</strong></summary>


```bash
accelerate launch --multi-gpu scripts/configs/qwen3/train_colqwen3_model.py
```

</details>



## License

ColQwen3's vision language backbone model ([Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) is under `apache2.0` license. The adapters attached to the model are under MIT license.

## Contact

- Mungeryang: mungerygm@gmail.com/yangguimiao@iie.ac.cn

## Acknowledgments

❤️❤️❤️

> Thanks to the **Colpali team** and **Qwen team** for their excellent open-source works!
> I accomplished this work by **standing on the shoulders of giants~**

👆👍

<p align="center">
    <img src="https://cdn.mos.cms.futurecdn.net/pqHroHNqYyQoJvEPrYkbcj-1200-80.jpg" width="80%"/>
<p>


## Citation

**ColPali: Efficient Document Retrieval with Vision Language Models**  

Authors: **Manuel Faysse**\*, **Hugues Sibille**\*, **Tony Wu**\*, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (\* denotes equal contribution)

```latex
@misc{faysse2024colpaliefficientdocumentretrieval,
      title={ColPali: Efficient Document Retrieval with Vision Language Models}, 
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}, 
}

@misc{macé2025vidorebenchmarkv2raising,
      title={ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval}, 
      author={Quentin Macé and António Loison and Manuel Faysse},
      year={2025},
      eprint={2505.17166},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2505.17166}, 
}
```
