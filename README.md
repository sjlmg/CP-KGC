# CP-KGC
The paper is available at: [Can Text-based Knowledge Graph Completion Benefit From Zero-Shot Large Language Models?](https://arxiv.org/pdf/2310.08279.pdf) 

In this paper, we found that (1) without fine-tuning, LLMs have the capability to further improve the quality of entity text descriptions. We validated this through experiments on the FB15K-237 and WN18RR datasets. (2) LLMs exhibit text generation hallucination issues and selectively output words with multiple meanings. This was mitigated by contextualizing prompts to constrain LLM outputs. (3) Larger model sizes do not necessarily guarantee better performance; even the 7B model can achieve optimized results in this comparative task. These findings underscore the untapped potential of large models in text-based KGC, which is a promising direction for further research in KGC.

![Alt text](./model.png)

CP-KGC data augmentation strategy evaluation framework. Design simple prompt cues to regenerate or supplement text content using entities and text descriptions from the WN18RR and FB15k-237 datasets. This enhances the expressive capacity of existing data to explore the hypothesis that the limited text descriptions inherent in the data restrict the performance ceiling of text-based KGC models.

# Requirements

* python>=3.8
* torch>=1.8 (for mixed precision training)
* transformers>=4.15

All experiments are run with 1 A100(80GB) GPU.

If you want to reproduce our best experimental results, you need to download the model weights [here](https://drive.google.com/drive/my-drive), and replace the file path.

The graphics required to reproduce the experiment is about 78GB.

CP-KGC used [SimKGC](https://github.com/intfloat/SimKGC) as the basic model in the paper. 

```
pip install transformers
```

Just replace the generated latest data of FB15K-237 and WN18RR in SimKGC.


* In FB15k-237, you need to replace FB15k_mid2description.txt with FB15k_mid2description_llama2_13B_chat.txt.
* In WN18RR，you need to replace wordnet-mlj12-definitions.txt with wordnet-mlj12-definitions_llama2_7B_chat.txt.



For filtering synonyms, please refer to the **Synonyms_WN18RR** folder.


If you **don't have the computing resources**, you can use the **Qwen-7B-Chat** and **LLaMA2-7B/13B-Chat** inference tests [here](https://modelscope.cn/topic/dfefe5be778b49fba8c44646023b57ba/pub/summary). ModelScope provides sufficient computing resources for inference testing of the 13B model. You can also use quantized models for inference.

# Contribute together

Apart from SimKGC, we have also tested the KG-S2S model, all of which are text-based knowledge graph completion models.

Would you like to proceed with further testing using a One-Shot or Few-Shot approach, or experiment with prompts that you find effective?

General-domain testing is meant to demonstrate the feasibility of this approach, while further exploration would require experimentation on domain-specific data. Considering the distribution of model training data, fine-tuning the model is necessary for vertical domains.

|       | | WN18RR|     |      |  |FB15K237 |    |     |
|-------|--------|-----|-----|------|----------|----|----|-----|
|       | MRR    | H@1 | H@3 | H@10 | MRR      | H@1| H@3| H@10|
|SimKGC |66.6|58.7|71.7|80.0|33.6|24.9|36.2|51.1|
|SimKGC+CP-KGC|67.3|59.9|72.1|80.4|33.8|25.1|36.5|51.6|
|KG-S2S |57.4|53.1|59.5|66.1|33.6|25.7|37.3|49.8|
|KG-S2S+CP-KGC |57.9|53.3|60.3|66.7|33.4|25.6|36.7|49.8|




# Citation
If you find our paper or code repository helpful, please consider citing as follows:
```
@article{yang2023cp,
  title={CP-KGC: Constrained-Prompt Knowledge Graph Completion with Large Language Models},
  author={Yang, Rui and Fang, Li and Zhou, Yi},
  journal={arXiv preprint arXiv:2310.08279},
  year={2023}
}
```
