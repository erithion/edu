import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser
import os.path


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


parser = ArgumentParser(description="Correlation matrix for CSV")
parser.add_argument("-i", dest="filename", required=True,
                    help="CSV file", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
args = parser.parse_args()


df = pd.read_csv(args.filename, encoding="utf8")

num_cols = len(list(df))
rng = range(1, num_cols + 1)
new_cols = ['C' + str(i) for i in rng]
df.columns = new_cols[:num_cols]


corrmat = df.corr()

plt.subplots()
plt.title('Correlation matrix')

sns.heatmap(corrmat, vmax=.9, square=True, annot=True, linewidths=.5)

plt.show()