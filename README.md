MDS Clique
=========
Generates cliques using MDS-Clique from topics extracted by LDA from a corpus.

## Prerequisites
Uses Python 3 and various Python libraries (`gensim`, `networkx`, `scikit-learn`, etc.)

## Installation
1. Use virtualenv to create a virtual Python 3 environment within the git repository (`virtualenv -p python3 venv`)
2. Activate your Python [virtualenv](https://virtualenv.pypa.io) `source venv/bin/activate` (you should now see something like `(venv)` in your console)
3. Install the required Python libraries by running `pip install -r requirements.txt`

### Installing Relative MDS
1. Clone https://github.com/hateno/scikit-learn.git and go to branch `v0.19.0`
2. Ensure you are still within the same virtualenv from the previous section and run `python setup.py build` and then `python setup.py install` (this may take a while)

## Quick Start
1. Configure `config.ini`
2. `python corpus.py`
3. `python gen_topic.py`
4. `python sim.py -dim 2 -data corpus`

### Detailed Explanation
Run `corpus.py` with `config.ini` filled out (see *config.ini* section) which reads a corpus (a directory of text documents) for pre-processing (e.g. stemming and tokenization). Then execute `gen_topic.py` which uses the output artifacts of `corpus.py` in order to perform LDA topic modeling over the pre-processed corpus. Then execute `sim.py` which will either perform a specific experiment or simply execute MDS-Clique (see sections below).

## Running `sim.py`
Execute `python sim.py --help` (make sure you are in your virtualenv) and go through the commands

### Sample `sim.py`
Generate a random pre-computed dissimilarity matrix, run MDS-Clique using the standard deviation (named `stress`) measure, it will write cliques to `out/cliques_<num>`
`python sim.py -dim 2 -data random --matrix -clique stress`

Use extracted LDA topics and run MDS-Clique using the distance measure, write cliques to `out/cliques_<num>`, note that there `-clusters <num>` needs to be higher than the number of topics extracted or an error will be thrown
`python sim.py -dim 2 -data corpus -clusters 3 -clique distance`

Run the RMDS experiment, set `-data none` since each sample will generate its own random data set
`python sim.py -dim 2 -data none --matrix -clique stress --rmds`

### Running Experiments
Each experiment is denoted with a flag `--<experiment_codename>`, by default an experiment will run 8 samples, you can manually specify number of samples with `-e <num_samples>`, and utilize 1/4 of the max cores available on the system, you can manually specify number of cores with `-c <num_cores>`

Relative MDS experiment (`k`-values are hard-coded)
`python sim.py -dim 2 -data random --matrix --relative`

MDS-Clique RMDS experiment
`python sim.py -dim 2 -data none --matrix --rmds`

MDS-Clique experiment
`python sim.py -dim 2 -data none --matrix --rclique`

Relative Online experiment
`python sim.py -dim 2 -data none --matrix --relativeonline`

Online Clique experiment
`python sim.py -dim 2 -data none --matrix -clique stress --onlineclique`

Online experiment
`python sim.py -dim 2 -data none --matrix --online`

### `config.ini`
* `corpus`: directory to the corpus (text documents)
	* A directory of plain text document(s) that will be pre-processed by `corpus.py`
* `mds_seed`: set MDS `random_state`
	* Optional, if blank the MDS algorithm (SMACOF) will start with a random configuration therefore most likely a different final result, set the seed value if you want a deterministic solution (useful for debugging), used by `sim.py`

#### Sample `config.ini`
```
[Global]
corpus = /path/to/sample-corpus/
mds_seed = 7
```

## Misc
Run in interactive debug mode
`ipython -i -c "%run -dim 2 -data corpus" --pdb`

You may need to manually create `store`, `out/final`, `out/experiment`, and `out/ident` directories
