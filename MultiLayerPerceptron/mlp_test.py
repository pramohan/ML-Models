import datasets
import mlp
import plotting

# start with a totally balanced split in 3 dimensions
N = 100
M = 100
dims = 3
gaus_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims).get_dataset()

X = gaus_dataset_points[:, :-1]
y = gaus_dataset_points[:, -1].astype(int)

model = mlp.MLP([3, 2])

model.train(X, y)
model.predict(X, y)
print(model.test_loss)
title = f'losses_gaussian_{dims}d_mlp'
plotting.plot_losses(title, model.get_losses(), save=True)
title = f'accuracies_gaussian_{dims}d_mlp'
plotting.plot_accuracies(title, model.get_accuracy(), save=True)
