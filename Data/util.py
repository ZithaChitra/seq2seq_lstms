import tensorflow as tf
from zipfile import ZipFile
import os
import pandas as pd


def download_data(download_dir, filename, url, unzip=True):
	# download file from given url
	# download dir is an absolute path
	file_path = os.path.join(download_dir, filename)
	_ = tf.keras.utils.get_file(
		file_path,
		url,
	)

	if unzip:
		with ZipFile(file_path) as zip:
			zip.printdir()
			print("Extracting files from: {file_path}...")
			zip.extractall(download_dir)
			print("File extraction done !")
	


