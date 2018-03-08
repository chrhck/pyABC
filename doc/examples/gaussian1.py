# import sys
# sys.path.remove('')
# sys.path.insert(0, '../../../pyabc')

import pyabc.visualization

import scipy
import tempfile
import os
import matplotlib.pyplot as pyplot

import logging
import sys

logging.basicConfig(level=logging.DEBUG)

df_logger = logging.getLogger('DistanceFunction')
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
df_logger.addHandler(console)
df_logger.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console)


df_logger.info('df_Hoho')
df_logger.debug('debuggggg')

logging.debug('Hoho')


def model(p):
    return {'ss1': p['theta'] + 1 + 0.1*scipy.randn(),
            'ss2': 2 + scipy.randn()}
# ss1 is informative, ss2 is uninformative about theta


prior = pyabc.Distribution(theta=pyabc.RV('uniform', 0, 10))

distance = pyabc.WeightedPNormDistance(p=2, adaptive=True)
# distance = pyabc.PNormDistance(2)
# sampler = pyabc.SingleCoreSampler();
abc = pyabc.ABCSMC(model, prior, distance, 100, lambda x: x, None, None, None,
                   None)
db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))
observation1 = 4
observation2 = 2
# desirable: optimal theta=3
abc.new(db_path, {'ss1': observation1, 'ss2': observation2})

# run
history = abc.run(minimum_epsilon=.1, max_nr_populations=10)

# output
fig, ax = pyplot.subplots()
for t in range(history.max_t + 1):
    df, w = history.get_distribution(m=0, t=t)
    pyabc.visualization.plot_kde_1d(df, w,
                xmin=0, xmax=10,
                x='theta', ax=ax,
                label="PDF t={}".format(t))
ax.axvline(observation1-1, color="k", linestyle="dashed")
ax.legend()
pyplot.show()

print("done test1")
