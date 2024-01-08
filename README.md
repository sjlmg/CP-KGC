# CP-KGC
The paper is available at: [Can Text-based Knowledge Graph Completion Benefit From Zero-Shot Large Language Models?](https://arxiv.org/pdf/2310.08279.pdf)

In this paper, we found that (1) without fine-tuning, LLMs have the capability to further improve the quality of entity text descriptions. We validated this through experiments on the FB15K-237 and WN18RR datasets. (2) LLMs exhibit text generation hallucination issues and selectively output words with multiple meanings. This was mitigated by contextualizing prompts to constrain LLM outputs. (3) Larger model sizes do not necessarily guarantee better performance; even the 7B model can achieve optimized results in this comparative task. These findings underscore the untapped potential of large models in text-based KGC, which is a promising direction for further research in KGC.

# Requirements

If you want to reproduce our best experimental results, you need to download the model weights [here](https://drive.google.com/drive/my-drive), and replace the file path.

The graphics required to reproduce the experiment is about 78GB.

CP-KGC used [SimKGC](https://github.com/intfloat/SimKGC) as the basic model in the paper. 

```
pip install transformers
```

Just replace the generated latest data of FB15K-237 and WN18RR in SimKGC.

For filtering synonyms, please refer to the **Synonyms_WN18RR** folder.


# Results

|model|wn18rr| | | |fb15k237| | | |
|     |MRR|H@1|H@3|H@10|MRR|H@1|H@3|H@10|
|-----|---|---|---|----|---|---|---|----|
|SimKGC|  |   |   |    |   |   |   |    |



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
