# Benchmark summary

500 prompts x 3 seeds per checkpoint. Brackets are 95% bootstrap CIs over prompts.
'Clean' = exactly one face, det score >= 0.7, face area >= 3% of the image.

| metric | base | sft | grpo |
|---|---|---|---|
| clean-face rate | 93.1% [91.6, 94.4] | 89.2% [87.4, 91.1] | 99.5% [99.1, 99.9] |
| exactly one face | 99.6% [99.3, 99.9] | 99.7% [99.3, 99.9] | 100.0% [100.0, 100.0] |
| no face | 0.3% [0.1, 0.7] | 0.2% [0.0, 0.5] | 0.0% [0.0, 0.0] |
| multiple faces | 0.1% [0.0, 0.2] | 0.1% [0.0, 0.3] | 0.0% [0.0, 0.0] |
| mean det score | 0.792 [0.787, 0.797] | 0.781 [0.777, 0.786] | 0.844 [0.841, 0.847] |
| ImageReward | -0.528 [-0.595, -0.463] | -0.234 [-0.302, -0.163] | 0.229 [0.157, 0.302] |

## Paired differences (next checkpoint minus previous)

- clean: sft - base = -3.8 pts [-5.9, -1.7]
- clean: grpo - sft = +10.3 pts [8.5, 12.2]
- ir: sft - base = +0.294 [0.236, 0.352]
- ir: grpo - sft = +0.462 [0.425, 0.498]

## ImageReward win rate (same prompt + seed)

- sft beats base: 61.7% of 1500 pairs
- grpo beats sft: 81.1% of 1500 pairs

## Clean-face rate by attribute (groups with >= 10 prompts)

**age**

| age     |   n_prompts |   base |   sft |   grpo |
|:--------|------------:|-------:|------:|-------:|
| 30-49   |          98 |  0.918 | 0.901 |  0.993 |
| 50-69   |          66 |  0.939 | 0.924 |  0.995 |
| 70+     |         112 |  0.923 | 0.872 |  0.997 |
| <30     |         132 |  0.942 | 0.891 |  1     |
| unknown |          92 |  0.931 | 0.884 |  0.989 |

**gender**

| gender            |   n_prompts |   base |   sft |   grpo |
|:------------------|------------:|-------:|------:|-------:|
| androgynous       |          75 |  0.916 | 0.884 |  0.991 |
| man               |          83 |  0.916 | 0.867 |  1     |
| nonbinary         |          82 |  0.943 | 0.866 |  0.996 |
| transgender man   |          82 |  0.955 | 0.927 |  0.996 |
| transgender woman |         100 |  0.923 | 0.907 |  0.993 |
| woman             |          78 |  0.932 | 0.897 |  0.996 |

**glasses**

| glasses   |   n_prompts |   base |   sft |   grpo |
|:----------|------------:|-------:|------:|-------:|
| False     |         378 |  0.926 | 0.892 |  0.994 |
| True      |         122 |  0.945 | 0.891 |  1     |

**headwear**

| headwear   |   n_prompts |   base |   sft |   grpo |
|:-----------|------------:|-------:|------:|-------:|
| False      |         326 |  0.941 | 0.9   |  0.998 |
| True       |         174 |  0.912 | 0.877 |  0.99  |

**angle**

| angle         |   n_prompts |   base |   sft |   grpo |
|:--------------|------------:|-------:|------:|-------:|
| candid        |         104 |  0.955 | 0.942 |  1     |
| off-center    |          85 |  0.937 | 0.902 |  0.992 |
| other         |         175 |  0.899 | 0.876 |  0.99  |
| straight-on   |          31 |  0.968 | 0.968 |  1     |
| three-quarter |          62 |  0.957 | 0.823 |  1     |
| tilted        |          43 |  0.922 | 0.86  |  1     |

**lighting**

| lighting         |   n_prompts |   base |   sft |   grpo |
|:-----------------|------------:|-------:|------:|-------:|
| dramatic/low-key |          93 |  0.918 | 0.871 |  0.996 |
| soft/other       |         407 |  0.934 | 0.897 |  0.995 |
