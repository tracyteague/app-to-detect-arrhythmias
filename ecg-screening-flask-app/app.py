from __future__ import print_function, division
from functools import partial

import csv
import os
import numpy as np
import flask
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, flash, render_template, request, redirect, jsonify, url_for, session
from werkzeug.utils import secure_filename

from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import *

from keras.utils import * 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.metrics import *
from keras.callbacks import *
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model

# Define and configure Flask app
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"
app.config['SECRET_KEY'] = 'super secret key'

# Set up model
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
MODEL_PATH = 'models/mc-final-model.h5'
model = load_model(MODEL_PATH)

model._make_predict_function()        
print('Model loaded. Start serving...')

# Functions

## Model Preprocessing
### Upsample
def upsample(y, new_len=2200):
    
    x = np.arange(0, len(y), 1)

    stop = (len(y)/360)
    start = 0
    step = (len(y)/360)/len(y)

    t = np.arange(start, stop, step)
    t_new = np.arange(start, stop, stop/new_len)
    
    if (len(t) != len(y)):
        t = t[:-1]
    
    y_new = np.interp(t_new, t, y)
    return y_new

### Normalize
def find_and_subtract_min_reading(readings_array):
    min_val = round(readings_array.min(), 4)
    readings_array = np.array([(reading-min_val) for i, reading in enumerate(readings_array)])
    return readings_array
    
def find_and_divide_max_reading(readings_array):
    max_val = round(readings_array.max(), 4)
    readings_array = np.array([(reading/max_val) for i, reading in enumerate(readings_array)])
    return readings_array
    
def normalize_readings(readings_array):
    return find_and_divide_max_reading(find_and_subtract_min_reading(readings_array))

## CSV

def read_csv_file_(file, filename):
	csv_df = pd.read_csv(file, sep=',', engine='python', quoting=csv.QUOTE_NONE)
	csv_df.columns = ['Sample #', 'MLII', 'beat']
	csv_df['Sample #'] = csv_df['Sample #'].apply(lambda x: x.replace('"', ''))
	csv_df['beat'] = csv_df['beat'].apply(lambda x: x.replace('"', ''))

	# Convert csv file into dataframe
	beats = list(csv_df.index[csv_df['beat'] == 'T'])
	readings = list(csv_df['MLII'])

	readings_per_beat = []
	prev_a = 0

	for a in beats:
		r = readings[prev_a:a+1]
		readings_per_beat.append(r)
		prev_a = a+1

	ecg_df = pd.DataFrame(list(zip(beats, readings_per_beat)), columns=['Sample #', 'Readings'])

	num_samp_readings = len(ecg_df['Readings'])

	# Preprocess data for modeling
	num_readings = len(ecg_df['Readings'])
	ecg_df['Readings'] = ecg_df['Readings'].apply(lambda v: upsample(v))
	ecg_df['Readings'] = ecg_df['Readings'].apply(lambda v: normalize_readings(v))

	ecg_input = ecg_df['Readings']
	ecg_input = np.concatenate(ecg_input.as_matrix(), axis=0).reshape(num_readings, 2200, 1)

	global sess
	global graph
	with graph.as_default():
		set_session(sess)
		preds = model.predict(ecg_input)
		preds_class = preds.argmax(axis=-1)

	preds_class = list(preds_class)
	preds_class_labels_dict = {0: 'APB', 1: 'LBB', 2: 'Normal', 3: 'RBB', 4: 'PVC'}
	preds_class_labels = [ preds_class_labels_dict[k] for k in preds_class ]

	# Create and save plot
	fig = plt.figure(figsize=(25, 4))
	ax = plt.axes()

	MLII = csv_df['MLII']
	plt.plot(MLII, color='black')

	plt.grid(color='#D3D4D9', linestyle='-', linewidth=1)
	plt.xticks(np.arange(0, len(MLII), 100))

	for beat in beats:
		plt.axvline(x=beat, color='r', linestyle='--', linewidth=2.5)

	annot_placement = csv_df.loc[:, 'MLII'].max()

	i = 0
	for beat in beats:
		plt.annotate(preds_class_labels[i],
					(beat, annot_placement),
					textcoords='offset points',
					xytext=(0, 15),
					ha='center',
					size=30)
		i += 1

	plt.savefig(f'static/images/plot-{filename}.png')
	fig_route = f'/static/images/plot-{filename}.png'
	# Identify whether individual has "normal" or "abnormal" beats detected
	
	abnormal_beats = [0, 1, 3, 4]
	is_normal = 0
	for beat in abnormal_beats:
		if beat in preds_class:
			is_normal = 1

	preds  = [ ["{:.2%}".format(x) for x in y] for y in preds ]

	## Create dataframe for chart with  sample #, main class, % breakdown
	classification_probs_df = pd.DataFrame(columns = ['Sample #', 'Classified Type', 'Probability Normal', 'Probability APB', 'Probability LBB', 'Probability RBB', 'Probability PVC'])
	# 
	for i in range(0, len(beats)):
		row = [beats[i], preds_class_labels[i], preds[i][0], preds[i][1], preds[i][2], preds[i][3], preds[i][4]]
		classification_probs_df = classification_probs_df.append(pd.Series((row), index=classification_probs_df.columns), ignore_index=True)

	return render_template('result.html', classification_probs_df=classification_probs_df, preds=preds, beats=beats, preds_class_labels=preds_class_labels, is_normal=is_normal, fig_route=fig_route, tables=[classification_probs_df.to_html(index=False, classes='data', header="true")])

#########################

### ROUTES

@app.route('/')
def index():
	return render_template('index.html') 

@app.route('/result', methods=['GET', 'POST'])
def result():
	if request.method == 'POST':
		# Check if post has file part
		error = None
		f = request.files['file']
		try:
			# Check if user does not select file
			basepath = os.path.dirname(__file__)
			file_path = os.path.join(
				basepath, 'uploads', secure_filename(f.filename))
			f.save(file_path)
			filename = str(f.filename).replace('.csv', '')
			try:
				return read_csv_file(f'uploads/{f.filename}', filename)
			except: 
				error = 'Incorrect file format. Please follow instructions and try again.'
				return render_template('/index.html', error=error)
		except:
			if f.filename == '':
				error = 'File not selected. Please try again.'
				return render_template('/index.html', error=error)
	return None


@app.route('/contact')
def contact():
	return render_template('contact.html')

if __name__ == '__main__':
	app.run(port=5000, debug=True)

