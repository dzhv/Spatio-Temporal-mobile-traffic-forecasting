import os
import os.path
from experiments.storage_utils import load_statistics
import numpy as np

results_folder = "results/lstm"


best_mse_losses = []
best_nrmse_losses = []

for dirpath, dirnames, filenames in os.walk(results_folder):
	for filename in [f for f in filenames if f == 'summary.csv']:
		summary_path = os.path.join(dirpath, filename)
		stats = load_statistics(summary_path)

		mse_losses = np.array(stats['val_loss']).astype(np.float)
		best_mse_loss = np.min(mse_losses)
		best_mse_model_idx = np.argmin(mse_losses)
		best_mse_losses.append((best_mse_loss, best_mse_model_idx, summary_path))

		nrmse_losses = np.array(stats['val_nrmse_loss']).astype(np.float)
		best_nrmse_loss = np.min(nrmse_losses)
		best_nrmse_model_idx = np.argmin(nrmse_losses)
		best_nrmse_losses.append((best_nrmse_loss, best_nrmse_model_idx, summary_path))

best_mse_losses.sort(key = lambda touple: touple[0])
best_nrmse_losses.sort(key = lambda touple: touple[0])

print("top 5 mse losses:")
for i in range(5):
	print(best_mse_losses[i])

print("top 5 nrmse losses:")
for i in range(5):
	print(best_nrmse_losses[i])