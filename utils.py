import matplotlib.pyplot as plt

def visualize_batch(batch, classes, dataset_type):
	batch_size = len(batch[0])
	for i in range(0, batch_size):
		# create a subplot
		plt.subplot(batch_size//8, 8, i + 1)
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		idx = batch[1][i]
		label = classes[idx]
		# show the image along with the label
		plt.imshow(image)
		plt.title(label)
		plt.axis("off")
	# show the plot
	# plt.tight_layout()
	plt.show()
	
def plot_metrics(train_metrics, test_metrics, xlabel, ylabel, title, save_path):
	epochs = len(train_metrics)
	plt.plot(range(epochs), train_metrics, label="train metric")
	plt.plot(range(epochs), test_metrics, label="test metric")
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.legend()
	plt.savefig(save_path)

def plot_saliency_map(saliency_map, image, target, prediction):
	fig, axes = plt.subplots(1, 2, figsize=(6, 3))
	if not target.item():
		idc = "IDC (-)"
	elif target.item() == 1:
		idc = "IDC (+)"
	if prediction == 1:
		idc_pred = "IDC (+)"
	elif prediction == 0:
		idc_pred = "IDC (-)"

    # Plot the original image
	axes[0].imshow(image)
	axes[0].set_title(f'Original Image, {idc}')
	axes[0].axis('off')

    # Plot the saliency map
	axes[1].imshow(saliency_map, cmap='hot')
	axes[1].set_title(f'Saliency Map, predicted {idc_pred}')
	axes[1].axis('off')

	plt.show()

def plotHistogram(data, label):
	print(data)
	print(label)
	plt.figure(figsize=(10,5))
	plt.subplot(1,2,1)
	plt.imshow(data)
	plt.axis('off')
	plt.title('IDC(+)' if label else 'IDC(-)')
	histo = plt.subplot(1,2,2)
	histo.set_ylabel('Count')
	histo.set_xlabel('Pixel Intensity')
	n_bins = 30
	plt.hist(data[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5)
	plt.hist(data[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5)
	plt.hist(data[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5)
	plt.show()