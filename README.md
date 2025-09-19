# Investigating the interaction of linguistic and mathematical reasoning in language models using multilingual number puzzles
This repository contains code and data for the EMNLP 2025 paper "Investigating the interaction of linguistic and mathematical reasoning in language models using multilingual number puzzles" by Antara Raaghavi Bhattacharya, Isabel Papadimitriou, Kathryn Davidson, and David Alvarez-Melis. A preprint is available on [ArXiv](https://arxiv.org/abs/2506.13886). 

## Results

Our results are under `result_files`. 

**Experiment 1:**

**o1-mini**
[single-character](result_files/ALL_STD_singletok)
[multi-character](result_files/ALL_STD_multitok)

**DeepSeek-R1-distill-Qwen-7B** 
[single-character](result_files/DEEPSEEK_STD_sgtok)
[multi-character](result_files/DEEPSEEK_STD_multitok)

**Experiment 2:**

**single-character**
[base](result_files/base_prompt_STD_sgtok)
[language](result_files/context_prompt_STD_sgtok)
[implicit operations](result_files/implctxt_prompt_STD_sgtok)

**multi-character**
[base](result_files/base_prompt_STD_multitok)
[language](result_files/context_prompt_STD_multitok)
[implicit operations](result_files/implctxt_prompt_STD_multitok)

**Experiment 3:**
[minimal pair problems](result_files/minpair_paradigms)

**Appendix**
[base experiments](result_files/base_expts)



## Experiments

(will be updated with details!)
The main code file to query o1-mini can be run at [`query_o1.py`](code/query_o1.py). 
