# Textual Style Classification & Transfer on Multiple Personas

Poster: https://drive.google.com/open?id=1oqbIPAuq1PdARAmyVPtiymH-t8rEK3ie

Data and baselines taken from ["(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas"](https://arxiv.org/abs/1909.00098) by Dongyeop Kang, Varun Gangal, and Eduard Hovy, EMNLP 2019

## The PASTEL dataset
PASTEL is a parallelly annotated stylistic language dataset.
The dataset consists of ~41K parallel sentences and 8.3K parallel stories annotated across different personas.


## Approach
Implemented copy-pointer based Seq2Seq models for the task of multi-persona style transfer. Achieved 18% gain in the BLEU score for stylized sentences while retaining the original meaning of the source text.

Achieved 15-20% improvement in F1 scores using BERT-based finetuning models for style classification task.
