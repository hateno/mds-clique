MDS Clique
=========
Run MDS-Clique, LDA, and experiments

## Prerequisites
Uses Python 3 and various Python libraries (`gensim`, `networkx`, `scikit-learn`, etc.)

## General workflow
1. `corpus.py`
2. `gen_topic.py`
3. `sim.py`

Run `corpus.py` (with `config.ini` filled out, see the *config.ini* section) which reads a corpus (a directory of text documents) for pre-processing (e.g. stemming and tokenization). Then execute `gen_topic.py` which uses the output artifacts of `corpus.py` in order to perform LDA topic modeling over the pre-processed corpus. Then execute `sim.py` which will either perform a specific experiment or simply execute MDS-Clique (see sections below).

### Quick Start

`python sim.py -dim 2 -data corpus`

Generates cliques from MDS-Clique using topics extracted by LDA from a corpus specified by *config.ini*

*NOTE:* you may get runtime errors, see installation procedure below, fill out *config.ini*, and try running this command again

## Installation
1. Install Python 3
2. Use virtualenv to create a virtual Python 3 environment within the git repository `virtualenv -p python3 venv`
3. Activate your Python virtualenv `source venv/bin/activate` (you should now see something like `(venv)` in your console)
4. Install the required Python libraries using, run `python sim.py` and it will tell you which libraries to install, install the Python library using `pip install <library-name>`

### Installing Relative MDS
1. Ensure you have installed the `scikit-learn` library in your virtualenv
2. Clone the git repository https://github.com/akbaylor/scikit-learn/tree/v0.19.0 in some other directory, checkout branch `v0.19.0`
3. Go to that directory (ensure you are still within the same virtualenv from the previous section) and run `python setup.py build` and then `python setup.py install`, you may get build errors to ensure you install whatever Python libraries or packages it asks you to install

## Running `sim.py`
Execute `python sim.py --help` (make sure you are in your virtualenv) and go through the commands

### Sample `sim.py`
Generate a random pre-computed dissimilarity matrix, run MDS-Clique using the standard deviation (named `stress`) measure, it will write cliques to `out/cliques_<num>`
`python sim.py -dim 2 -data random --matrix -clique stress`

Use extracted LDA topics and run MDS-Clique using the distance measure, write cliques to `out/cliques_<num>`, note that there `-clusters <num>` needs to be higher than the number of topics extracted or an error will be thrown
`python sim.py -dim 2 -data corpus -clusters 3 -clique distance`

Run the RMDS experiment, set `-data none` since each sample will generate its own random data set
`python sim.py -dim 2 -data none --matrix -clique stress --rmds`

### Data `text`
Specify your own dissimilarity matrix in a text file. For now, specify it as a triangular dissimilarity matrix or a symmetric matrix.

`python sim.py -dim 2 -data text -textfile /path/to/matrix/textfile`

Note, cluster support is not yet present.

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
	* A directory of plain text documents that will be pre-processed by `corpus.py`
* `stopwords`: path to stopwords file
	* Stopwords filters out certain words from the corpus so it doesn't show in the final vocabulary set, typically you will filter out common words like 'and' and 'this', used only by `corpus.py`
* `mds_seed`: set MDS `random_state`
	* Optional, if blank the MDS algorithm (SMACOF) will start with a random configuration therefore most likely a different final result, set the seed value if you want a deterministic solution (useful for debugging), used by `sim.py`

#### Sample `config.ini`
```
[Global]
corpus = /path/to/sample-corpus/
stopwords = /path/to/stopwords-file
mds_seed = 7
```

## Misc
Run in interactive debug mode
`ipython -i -c "%run -dim 2 -data corpus" --pdb`

You may need to manually create `store`, `out/final`, `out/experiment`, and `out/ident` directories
