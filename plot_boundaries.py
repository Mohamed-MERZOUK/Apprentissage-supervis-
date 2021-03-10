import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


palette = sns.color_palette("Set2")

def plot_boundaries(X, y, clf, title, xLabel, yLabel, pathToSave):

	nbClasses = pd.DataFrame(y)[0].value_counts().count()

	cmap_bold = palette[0:nbClasses]
	cmap_light = ListedColormap(palette[0:nbClasses])

	h = .02  # step size in the mesh

	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(figsize=(12, 6))
	plt.contourf(xx, yy, Z, cmap=cmap_light)

	# Plot also the training points
	df = pd.DataFrame(data=y, columns=["class"])
	sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.flatten().astype(int), palette=cmap_bold, alpha=1.0, edgecolor="black")
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title(title)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	plt.savefig(pathToSave)
	plt.show()