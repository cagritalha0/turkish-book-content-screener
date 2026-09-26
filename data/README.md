# Data

Nothing in `data/raw/` is committed.

## Offensive-language dataset (training)

Turkish Offensive Language Detection by Toygar Tanyel et al., built from several earlier Turkish
offensive-language corpora (tweets labelled `0` = not offensive, `1` = offensive).

| Split | Rows   |
|-------|-------:|
| train | 42,398 |
| valid |  1,756 |
| test  |  8,851 |

Download from [Kaggle](https://www.kaggle.com/datasets/toygarr/turkish-offensive-language-detection)
or [Hugging Face](https://huggingface.co/datasets/Toygar/turkish-offensive-language-detection) and place
`train.csv`, `valid.csv` and `test.csv` (columns `id,text,label`) in `data/raw/`.

> Tanyel, T., Alkurdi, B., & Ayvaz, S. (2022). *Linguistic-based Data Augmentation Approach for
> Offensive Language Detection.* 7th International Conference on Computer Science and Engineering (UBMK).

## Books (inference only)

The system is meant to be run on books you have the right to use. No books are bundled with this
repository.
