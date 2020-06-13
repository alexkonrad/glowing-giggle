import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LoadData():
  def __init__(self, directory='images/', limit=np.inf, test_size=0.33, verbose=False):
    self.directory = directory
    self.limit = limit
    self.verbose = verbose
    self.test_size = test_size
    self.image_path = sorted(list(Path(self.directory).glob('*_s.bmp')))
    self.gt_path = sorted(list(Path(self.directory).glob('*_s_GT.bmp')))
    self.N = min(self.limit, len(self.image_path))
    self.X = np.ndarray((self.N, 213, 320, 3))
    self.Y = np.ndarray((self.N, 213, 320, 3))
    self.load_images()
    self.load_ground_truth()

  def load_images(self):
    for i in range(self.N):
      filename = str(self.image_path[i])
      img = cv2.imread(filename)
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
      self.X[i,:,:,:] = np.asarray(img).reshape((213, 320, 3))

    self.verbose and print(f"Loaded {self.X.shape[0]} images.")

  def load_ground_truth(self, limit=-1):
    for i in range(self.N):
      filename = str(self.gt_path[i])
      img = cv2.imread(filename)
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
      self.Y[i,:,:,:] = np.asarray(img).reshape((213, 320, 3))

    self.verbose and print(f"Loaded {self.Y.shape[0]} ground truth images.")

  def train_test_split(self):
    return train_test_split(self.X, self.Y, test_size=self.test_size, random_state=42)

  def display(self, i):
    fig, (x,y) = plt.subplots(1, 2)

    X = self.X[i,:,:,:]
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    X = cv2.normalize(X.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x.imshow(self.X[i,:,:,:])
    x.axis('off')

    Y = self.Y[i,:,:,:]
    Y = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)
    Y = cv2.normalize(Y.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    y.imshow(Y[i,:,:,:])
    y.axis('off')

