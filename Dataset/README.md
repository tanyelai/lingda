# Dataset Summary

This dataset is enhanced version of existing offensive language studies. Existing studies are highly imbalanced, and solving this problem is too costly. To solve this, we proposed contextual data mining method for dataset augmentation. Our method is basically prevent us from retrieving random tweets and label individually. We can directly access almost exact hate related tweets and label them directly without any further human interaction in order to solve imbalanced label problem.

In addition, existing studies *(can be found at Reference section)* are merged to create even more comprehensive and robust dataset for Turkish offensive language detection task. 

The file turkish_off_lang_detection.zip includes; train.csv, test.csv and valid.csv with 42.398, 8.851, and 1.756 samples respectively.

# Dataset Structure

A binary dataset with (0) Not Offensive and (1) Offensive tweets.

### Task and Labels
Offensive language identification:
- (0) Not Offensive - Tweet does not contain offense or profanity.
- (1) Offensive - Tweet contains offensive language or a targeted (veiled or direct) offense

### Data Splits
| | train | test | dev |
|------:|:------|:-----|:-----|
| 0 (Not Offensive) | 22,589 | 4,436 | 1,402 |
| 1 (Offensive) | 19,809 | 4,415 | 354 |

### Alternative Link (Huggingface)
- https://huggingface.co/datasets/Toygar/turkish-offensive-language-detection
 
### Citation Information
```
BibTeX will be provided after publication.
UBMK 2022 Paper: "Linguistic-based Data Augmentation Approach for Offensive Language Detection"
```

# References
We merged open-source offensive language dataset studies in Turkish to increase contextuality with existing data even more, before our method is applied.
- https://huggingface.co/datasets/offenseval2020_tr
- https://github.com/imayda/turkish-hate-speech-dataset-2
- https://www.kaggle.com/datasets/kbulutozler/5k-turkish-tweets-with-incivil-content

