#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:58:23 2020

@author: rkaratt
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.summary import FileWriter






# Converting a SavedModel.
saved_model_dir="/home/rkaratt/TeamProject/resnet50-tensorflow/training/epoch0"
converter = tf.contrib.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)




