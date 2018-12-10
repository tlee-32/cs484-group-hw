import matplotlib.pyplot as plt
from .stats import toxicCount, nonToxicCount
from keras.utils.vis_utils import plot_model

def plotToxicAndNonToxicDistribution():
  classNames = ['non_toxic', 'toxic','severe_toxic','obscene','threat','insult','identity_hate']
  values = nonToxicCount() + toxicCount() 
  plt.bar(classNames, values)
  for xy in zip(classNames, values):
    plt.annotate('%s' % xy[1], xy=xy, textcoords='data',ha='center')
  plt.suptitle('Number of toxic and non-toxic records', fontsize=16)
  plt.xlabel('Type', fontsize=14)
  plt.ylabel('Count', fontsize=14)


def plotModel(model):
  plot_model(model.model, to_file='./plots/model_plot.png', show_shapes=True, show_layer_names=True)


# plotToxicAndNonToxicDistribution()
# plt.show()