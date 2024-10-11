# AIAA-hw1

## get_bof.py

```python
#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster import KMeans
import sys
import time
import collections
import csv
import argparse
from tqdm import tqdm
# Generate MFCC-Bag-of-Word features for videos
# each video is represented by a single vector

parser = argparse.ArgumentParser()
parser.add_argument("kmeans_model")
parser.add_argument("cluster_num", type=int)
parser.add_argument("file_list")
parser.add_argument("--mfcc_path", default="mfcc")
parser.add_argument("--output_path", default="bof")

if __name__ == '__main__':
  args = parser.parse_args()

  # 1. load the kmeans model
  kmeans = pickle.load(open(args.kmeans_model, "rb"))

  # 2. iterate over each video and
  start = time.time()
  fread = open(args.file_list, "r")
  for line in tqdm(fread.readlines()):
    mfcc_path = os.path.join(args.mfcc_path, line.strip() + ".mfcc.csv")
    bof_path = os.path.join(args.output_path, line.strip() + ".csv")

    if not os.path.exists(mfcc_path):
      continue
    # (num_frames, d)
    array = numpy.genfromtxt(mfcc_path, delimiter=";")

    #  ************************************TA: Your code starts here

    ## use kmeans.predict(mfcc_features_of_video)
    # (num_frames,), each row is an integer for the closest cluster center
    prediction = kmeans.predict(array)

    # create dict containing frequencies of each "code word"
    # {0: count_for_0, 1: count_for_1, ...}
    frequency_dict = collections.Counter(prediction)

    # normalize the frequency by dividing with frame number
    num_frames = array.shape[0]
    if num_frames > 0:
      list_freq = [frequency_dict[i] / num_frames for i in range(args.cluster_num)]
    else:
      list_freq = [0] * args.cluster_num

    #  ************************************TA: Your code ends here

    numpy.savetxt(bof_path, list_freq)

  end = time.time()
  print("K-means features generated successfully!")
  print("Time for computation: ", (end - start))
```

## train_kmeans.py

```python
#!/bin/python 
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle 
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print("mfcc_csv_file -- path to the mfcc csv file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    mfcc_csv_file = sys.argv[1]; 
    output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
    # 1. load all mfcc features in one array
    selection = pd.read_csv(mfcc_csv_file, sep=';', dtype='float')

    # TA: perform kmeans clustering here. get a model file variable kmeans
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(selection)

    # 2. Save trained model
    pickle.dump(kmeans, open(output_file, 'wb'))

    print("K-means trained successfully!")
```


## train_svm_multiclass.py

```python
#!/bin/python

import numpy as np
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import argparse
import sys
import pdb

# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  # pass array for svm training
  # not one-versus-rest multiclass strategy
  
  #param_grid = {'C': [0.1, 1, 5, 10, 50, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1]}
  #grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
  #grid_search.fit(X, y)
  #print('Best parameters:', grid_search.best_params_)

  svm_model = SVC(C=30, kernel='rbf', gamma=1)
  bagging_model = BaggingClassifier(estimator=svm_model, n_estimators=10, random_state=42)
  bagging_model.fit(X, y)
  accuracies = cross_val_score(bagging_model, X, y, cv=5, scoring='accuracy')
  print("Cross-validation accuracies: ", accuracies)
  print("Average accuracy: ", np.mean(accuracies))

  # save trained SVM in output_file
  pickle.dump(bagging_model, open(args.output_file, 'wb'))
  print('One-versus-rest multi-class SVM trained successfully')
```


## train_mlp.py

```python
#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pickle
import argparse
import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  # TA: write your code here 
  mlp_model = MLPClassifier(hidden_layer_sizes=(200, ), max_iter=500, random_state=1)
  mlp_model.fit(X, y)

  accuracies = cross_val_score(mlp_model, X, y, cv=5, scoring='accuracy')
  print("Cross-validation accuracies: ", accuracies)
  print("Average accuracy: ", np.mean(accuracies))
  
  # save trained MLP in output_file
  pickle.dump(mlp_model, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
```


## train_LR.py

```python
import numpy as np
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import argparse
import sys
import pdb

# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  # pass array for LR training
  # one-versus-rest multiclass strategy
  lr_model = LogisticRegression(max_iter=2000, multi_class='ovr')
  bagging_model = BaggingClassifier(estimator=lr_model, n_estimators=10, random_state=42)
  bagging_model.fit(X, y)

  # save trained SVM in output_file
  pickle.dump(bagging_model, open(args.output_file, 'wb'))
  print('LR trained successfully')
```


# test_svm_multiclass.py

```python
#!/bin/python

import argparse
import numpy as np
import os
from sklearn.svm import SVC
import pickle
import sys
import numpy as np

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load svm model
  svm = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

  X = np.array(feat_list)

  # 3. Get scores with trained svm model
  # (num_samples, num_class)
  scoress = svm.decision_function(X)

  # 4. save the argmax decisions for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, scores in enumerate(scoress):
      predicted_class = np.argmax(scores)
      f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
```


## test_mlp.py

```python
#!/bin/python

import argparse
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import sys
import numpy as np

# Apply the MLP model to the testing videos;
# Output prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

  X = np.array(feat_list)

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))
```


## test_LR.py

```python
#!/bin/python

import argparse
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import numpy as np

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load svm model
  LR = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

  X = np.array(feat_list)

  # 3. Get scores with trained LR model
  # (num_samples, num_class)
  scoress = LR.predict_proba(X)

  # 4. save the argmax decisions for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, scores in enumerate(scoress):
      predicted_class = np.argmax(scores)
      f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
```
