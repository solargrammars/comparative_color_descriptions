
# README

Pytorch implementation of the paper 'Lighter' Can Still Be Dark: Modeling Comparative Color Descriptions (https://aclweb.org/anthology/P18-2125) by Winn et al. 

## Data

Download the comparative dataset from https://bitbucket.org/o_winn/comparative_colors.git
and modify the value of `DATA_PATH` in `settings.py` accordingly.

Download Google pretrained vectors from https://code.google.com/archive/p/word2vec/
and modify the value of `GOOGLE_W2V` in `settings.py` accordingly.

Run `run_get_w2v_sample.py`

## Training

Run `python run_train.py`

## Testing 

If you just need the test loss, run `python run_test --model_path PATH`

If you want to visualize the results, please check `experiments/explore_inference.ipynb`

## TODO
- Paramater search
- Delta-E computation

## Acknowledgement

Thanks to Olivia Winn for providing the`data_maker.py` script.
