# CP-KGC
[Can Text-based Knowledge Graph Completion Benefit From Zero-Shot Large Language Models?](https://arxiv.org/pdf/2310.08279.pdf)

**Abstract**：Text-based knowledge graph completion (KGC) methods, leveraging textual entity descriptions are at the research forefront. The efficacy of these models hinges on the quality of the textual data. This study explores whether enriched or more efficient textual descriptions can amplify model performance. Recently, Large Language Models (LLMs) have shown remarkable improvements in NLP tasks, attributed to their sophisticated text generation and conversational capabilities. LLMs assimilate linguistic patterns and integrate knowledge from their training data. Compared to traditional databases like Wikipedia, LLMs provide several advantages, facilitating broader information querying and content augmentation. We hypothesize that LLMs, without fine-tuning, can refine entity descriptions, serving as an auxiliary knowledge source. An in-depth analysis was conducted to verify this hypothesis. We found that (1) without fine-tuning, LLMs have the capability to further improve the quality of entity text descriptions. We validated this through experiments on the FB15K-237 and WN18RR datasets. (2) LLMs exhibit text generation hallucination issues and selectively output words with multiple meanings. This was mitigated by contextualizing prompts to constrain LLM outputs. (3) Larger model sizes do not necessarily guarantee better performance; even the 7B model can achieve optimized results in this comparative task. These findings underscore the untapped potential of large models in text-based KGC, which is a promising direction for further research in KGC.


CP-KGC used [SimKGC](https://github.com/intfloat/SimKGC) as the basic model in the paper. 

Just replace the generated latest data of FB15K-237 and WN18RR in SimKGC.

For filtering synonyms, please refer to the **Synonyms_WN18RR** folder.


```
@article{yang2023cp,
  title={CP-KGC: Constrained-Prompt Knowledge Graph Completion with Large Language Models},
  author={Yang, Rui and Fang, Li and Zhou, Yi},
  journal={arXiv preprint arXiv:2310.08279},
  year={2023}
}
```
