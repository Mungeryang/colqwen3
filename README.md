# ColQwen3: Efficient Document Retrieval with Qwen3-VL-2B-Instruct 👀

### This is the v0.1 version trained with batch_size 32 for 1 epoch and with the updated pad token

> Welcome to follow and star! ⭐️⭐️⭐️

[![arXiv](https://img.shields.io/badge/arXiv-2407.01449-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2407.01449)
[![GitHub](https://img.shields.io/badge/ViDoRe_Benchmark-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/illuin-tech/vidore-benchmark)
[![Hugging Face](https://img.shields.io/badge/Vidore_Hf_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/goodman2001/colqwen3-v0.1)

## 📜 News

**[2025.12.10]** 🎉🎉 I have released the [ColQwen3-v0.2](https://huggingface.co/goodman2001/colqwen3-v0.2) model based on ColQwen3-Base

**[2025.12.02]** 🎉🎉 I have released the [ColQwen3-v0.1](https://huggingface.co/goodman2001/colqwen3-v0.1) model based on ColQwen3-Base

**[2025.12.02]** 🎉🎉 I have released the [ColQwen3-Base](https://huggingface.co/goodman2001/colqwen3-base) model based on Qwen3-VL-2B-Instruct


## Related Work

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

## ⚙️ Setup

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

## ColQwen3 Test Results


🎉🎉 **[2025.12.08]** I used [mteb](https://github.com/illuin-tech/vidore-benchmark) to evaluate(NDCG@5) my ColQwen3-v0.1 retriever on the **ViDoRe benchmark v2**.

| model          | BioMedicalLectures-french | BioMedicalLectures-spanish | BioMedicalLectures-english | BioMedicalLectures-german | EconomicsReports-french | EconomicsReports-spanish | EconomicsReports-english | EconomicsReports-german | ESGReports-french | ESGReports-spanish | ESGReports-english | ESGReports-german | ESGReportsHL |
|----------------|---------------------------|----------------------------|----------------------------|---------------------------|-------------------------|---------------------------|---------------------------|--------------------------|-------------------|--------------------|--------------------|-------------------|-------------|
| **colqwen3-v0.1**  | 55.32                     | 56.35                      | 58.87                      | 51.73                     | 40.77                   | 41.38                     | 57.22                     | 44.38                    | 50.51             | 47.12              | 51.34              | 48.08             | 55.75       |
| **colqwen3-v0.2**  | 57.40                     | 58.67                      | 62.37                      | 56.45                     | 50.18                   | 52.90                     | 63.24                     | 53.58                    | 52.97             | 50.89              | 50.81              | 52.61             | 52.87       |


⚙️I used [mteb](https://github.com/illuin-tech/vidore-benchmark) to evaluate my ColQwen3-v0.1 retriever on the ViDoRe benchmark.


| Model                                  | ArxivQ              | DocQ                | InfoQ               | TabF               | TATQ               | Shift              | AI                 | Energy             | Gov.               | Health.            | Avg.               |
|----------------------------------------|---------------------|---------------------|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| **`Unstructured` (text-only)**         |                     |                     |                     |                    |                    |                    |                    |                    |                    |                    |                    |
| - BM25                                 | -                   | 34.1                | -                   | -                  | 44.0               | 59.6               | 90.4               | 78.3               | 78.8               | 82.6               | -                  |
| - BGE-M3                               | -                   | 28.4 (↓5.7)         | -                   | -                  | 36.1 (↓7.9)        | 68.5 (↑8.9)        | 88.4 (↓2.0)        | 76.8 (↓1.5)        | 77.7 (↓1.1)        | 84.6 (↑2.0)        | -                  |
| **`Unstructured` + OCR**               |                     |                     |                     |                    |                    |                    |                    |                    |                    |                    |                    |
| - BM25                                 | 31.6                | 36.8                | 62.9                | 46.5               | 62.7               | 64.3               | 92.8               | 85.9               | 83.9               | 87.2               | 65.5               |
| - BGE-M3                               | 31.4 (↓0.2)         | 25.7 (↓11.1)        | 60.1 (↓2.8)         | 70.8 (↑24.3)       | 50.5 (↓12.2)       | **73.2 (↑8.9)**    | 90.2 (↓2.6)        | 83.6 (↓2.3)        | 84.9 (↑1.0)        | 91.1 (↑3.9)        | 66.1 (↑0.6)        |
| **`Unstructured` + Captioning**        |                     |                     |                     |                    |                    |                    |                    |                    |                    |                    |                    |
| - BM25                                 | 40.1                | 38.4                | 70.0                | 35.4               | 61.5               | 60.9               | 88.0               | 84.7               | 82.7               | 89.2               | 65.1               |
| - BGE-M3                               | 35.7 (↓4.4)         | 32.9 (↓5.4)         | 71.9 (↑1.9)         | 69.1 (↑33.7)       | 43.8 (↓17.7)       | 73.1 (↑12.2)       | 88.8 (↑0.8)        | 83.3 (↓1.4)        | 80.4 (↓2.3)        | 91.3 (↑2.1)        | 67.0 (↑1.9)        |
| **Contrastive VLMs**                   |                     |                     |                     |                    |                    |                    |                    |                    |                    |                    |                    |
| Jina-CLIP                              | 25.4                | 11.9                | 35.5                | 20.2               | 3.3                | 3.8                | 15.2               | 19.7               | 21.4               | 20.8               | 17.7               |
| Nomic-vision                           | 17.1                | 10.7                | 30.1                | 16.3               | 2.7                | 1.1                | 12.9               | 10.9               | 11.4               | 15.7               | 12.9               |
| SigLIP (Vanilla)                       | 43.2                | 30.3                | 64.1                | 58.1               | 26.2               | 18.7               | 62.5               | 65.7               | 66.1               | 79.1               | 51.4               |
| SigLIP (Vanilla)                       | 43.2                | 30.3                | 64.1                | 58.1               | 26.2               | 18.7               | 62.5               | 65.7               | 66.1               | 79.1               | 51.4               |
| BiSigLIP (+fine-tuning)                | 58.5 (↑15.3)        | 32.9 (↑2.6)         | 70.5 (↑6.4)         | 62.7 (↑4.6)        | 30.5 (↑4.3)        | 26.5 (↑7.8)        | 74.3 (↑11.8)       | 73.7 (↑8.0)        | 74.2 (↑8.1)        | 82.3 (↑3.2)        | 58.6 (↑7.2)        |
| BiPali (+LLM)                          | 56.5 (↓2.0)         | 30.0 (↓2.9)         | 67.4 (↓3.1)         | 76.9 (↑14.2)       | 33.4 (↑2.9)        | 43.7 (↑17.2)       | 71.2 (↓3.1)        | 61.9 (↓11.7)       | 73.8 (↓0.4)        | 73.6 (↓8.8)        | 58.8 (↑0.2)        |
| *ColPali* (+Late Inter.)               | 79.1 (↑22.6)        | 54.4 (↑24.5)        | 81.8 (↑14.4)        | 83.9 (↑7.0)        | 65.8 (↑32.4)       | 73.2 (↑29.5)       | 96.2 (↑25.0)       | 91.0 (↑29.1)       | 92.7 (↑18.9)       | 94.4 (↑20.8)       | 81.3 (↑22.5)       |
| **Ours**                               |                     |                     |                     |                    |                    |                    |                    |                    |                    |                    |                    |
| ***Colqwen3* (+Late Inter.)**          | **80.1 (↑1.0)**     | **55.8 (↑1.4)**     | **86.7 (↑5.9)**     | 82.1 (↓1.8)        | **70.8 (↑5.0)**    | **75.9 (↑2.7)**    | **99.1 (↑2.9)**    | **95.6 (↑4.6)**    | **96.1 (↑3.4)**    | **96.8 (↑2.4)**    | **83.9 (↑2.6)**    |



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
