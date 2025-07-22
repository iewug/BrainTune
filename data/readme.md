You can download the data in the official website https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/. In this paper, only the original `cnt` files of SEED, SEED-IV and SEED-V are used and we recalculated the DE features, because the DE features offered in the official website are smoothed by LDS which do not meet the requirement of real-time.

Put these `cnt` files in `data` folder like:
```
data
├── seed3 # 1_1 ~ 15_3
│   ├── 1_1
│   │   └── 1_1.cnt
│   ├── 1_2
│   │   └── 1_2.cnt
│   ├── 1_3
│   │   └── 1_3.cnt
│   └── 2_1 ~ 15_3 # the remaining folders and files
├── seed4 # 1_1 ~ 15_3
│   └── ...
└── seed5 # 1_1 ~ 16_3
    └── ...
```