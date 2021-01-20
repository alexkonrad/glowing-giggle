import cv2
import pickle

from scene.data import *
from scene.sift import SIFTDescriptor
from sklearn.cluster import KMeans
from sklearn.naive_bayes import CategoricalNB

class SceneClassifier():
  def __init__(self, verbose=True, limit=np.inf, patch_dim=20, height=200, width=320):
    self.verbose = verbose
    self.patch_dim = patch_dim
    self.height = height
    self.width = width
    self.num_patches = int((height * width) / (patch_dim * patch_dim))
    self.ld = LoadData(verbose=verbose, limit=limit)
    self.X_train, self.X_test, self.y_train, self.y_test = self.ld.train_test_split()
    self.crop()
    self.X_train_patches = self.patches(self.X_train)
    self.X_test_patches = self.patches(self.X_test)
    self.y_train_patches = self.patches(self.y_train)
    self.y_test_patches = self.patches(self.y_test)
    N = self.X_train.shape[0]
    total_patches = N * self.num_patches
    self.X_sift_bank = np.zeros((total_patches, 128))
    self.X_hue_bank = np.zeros((total_patches, 36))
    self.X_pos_bank = np.zeros((total_patches, 1)) # Should this be n_patches
    self.X_train_sift = np.zeros((N, self.num_patches))
    M = self.X_test.shape[0]
    self.X_test_features = np.zeros((M, self.num_patches))
    self.X_test_sift = np.zeros((M, self.num_patches))
    self.X_test_hue = np.zeros((M, self.num_patches))
    self.X_test_pos = np.zeros((M, self.num_patches))

  def fit(self):
    """ Fit Naive Bayes Classifier. """
    self.clf = CategoricalNB()
    self.clf.fit(self.X_train_feat, self.y_train_labels.reshape(-1))

  def predict(self, i):
    """ Predict range from test set """
    feat = self.X_test_feat.reshape((self.X_test_patches.shape[0], self.num_patches, 1260))
    pred = self.clf.predict(feat[i,:,:])
    return pred


  def error(self):
    """ Measure error on test set. """
    accs = []
    for i in range(self.X_test.shape[0]):
      pred = self.predict(i)
      acc = np.count_nonzero(pred == self.y_test_labels[i]) / self.num_patches
      accs.append(acc)
    return np.mean(acc)

  def compute_features(self, test=False, save=True):
    """ Compute features. Flag to include test features too, and save to file. """
    if not test:
      self.verbose and print("Computing training features.")
      self.sift_descriptors()
      self.learn_sift_dictionary()

      if save:
        with open('sift.pickle', 'wb') as f:
          pickle.dump(self.sift_dictionary, f)

      self.hue_descriptors()
      self.learn_hue_dictionary()
      if save:
        with open('hue.pickle', 'wb') as f:
          pickle.dump(self.hue_dictionary, f)

      self.position_descriptors()

      self.bitwise_features()

      if save:
        with open('features.pickle', 'wb') as f:
          pickle.dump(self.X_train_feat, f)

      self.encode_y()
      print("Finished computing training features.")
    if test:
      print("Computing test features.")
      self.sift_descriptors(test=True)
      self.hue_descriptors(test=True)
      self.position_descriptors(test=True)
      self.bitwise_features(test=True)
      # self.encode_y(test=True)

  def load(self):
    """ Load classifier data and features from files. """
    with open('sift.pickle', 'rb') as f:
      self.sift_dictionary = pickle.load(f)

    with open('hue.pickle', 'rb') as f:
      self.hue_dictionary = pickle.load(f)

    with open('features.pickle', 'rb') as f:
      self.X_train_feat = pickle.load(f)

    with open('clf.pickle', 'rb') as f:
      self.clf = pickle.load(f)

    with open('features-test.pickle', 'rb') as f:
      self.X_test_feat = pickle.load(f)

    self.encode_y()

  def save(self, test=False):
    """ Save computed model parameters to disk. """
    if not test:
      with open('sift.pickle', 'wb') as f:
        pickle.dump(self.sift_dictionary, f)

      with open('hue.pickle', 'wb') as f:
        pickle.dump(self.hue_dictionary, f)

      with open('features.pickle', 'wb') as f:
        pickle.dump(self.X_train_feat, f)

      with open('clf.pickle', 'wb') as f:
        pickle.dump(self.clf, f)

    if test:
      with open('features-test.pickle', 'wb') as f:
        pickle.dump(self.X_test_feat, f)

  def crop(self):
    """ Crop images from bottom and right to a convenient size for building patches. """
    self.X_train = self.X_train[:, :self.height, :self.width, :]
    self.X_test = self.X_test[:, :self.height, :self.width, :]
    self.y_train = self.y_train[:, :self.height, :self.width, :]
    self.y_test = self.y_test[:, :self.height, :self.width, :]

  def patches(self, images):
    """ Divide an image into patches. """
    patch_size = self.patch_dim * self.patch_dim
    n, h, w, d = images.shape
    thresh = h // self.patch_dim * self.patch_dim
    S = (n, self.num_patches, patch_size, d)
    patches = images[:,:thresh,:,:].reshape(S)
    if thresh < h:
      patches = np.vstack((
        patches,
        images[:, thresh + 1:, :, :].reshape(S)
      ))
    remaining_rows = self.num_patches - patches.shape[1]
    if remaining_rows > 0:
      patches = np.vstack((
        patches,
        np.zeros((n, remaining_rows, patch_size, d))
      ))
    return patches

  def predict_and_display(self, idx):
    fig, ax = plt.subplots(3, 3)
    for i in range(3):
      pred = self.predict(idx+i)
      img = self.decode_y(pred).reshape(10, 16, 3)
      ax[0,i].imshow(self.process(self.X_test[idx+i]))
      ax[1,i].imshow(self.process(self.y_test[idx+i]))
      ax[2,i].imshow(self.process(img))
    [a.axis('off') for a in np.ravel(ax)]
    # plt.show()
    plt.savefig(f'report/images/result{idx}.png')

  def process(self, image):
    """ Convert image to format for display. """
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return image

  def encode_y(self):
    """ Turn y labels from 3-D RGB values to ordinals. """
    n, n_patches, patch_size, d = self.y_train_patches.shape
    m = self.y_test_patches.shape[0]
    labels = {
      (0, 0, 0): 'void',
      (128, 0, 0): 'building',
      (0, 128, 0): 'grass',
      (128, 128, 0): 'tree',
      (0, 0, 128): 'cow',
      (128, 0, 128): 'horse',
      (0, 128, 128): 'sheep',
      (128, 128, 128): 'sky',
      (64, 0, 0): 'mountain',
      (192, 0, 0): 'aeroplane',
      (64, 128, 0): 'water',
      (192, 128, 0): 'face',
      (64, 0, 128): 'car',
      (192, 0, 128): 'bicycle'
    }

    labels = {
      (0, 0, 0): 0,
      (128, 0, 0): 1,
      (0, 128, 0): 2,
      (128, 128, 0): 3,
      (0, 0, 128): 4,
      (128, 0, 128): 5,
      (0, 128, 128): 6,
      (128, 128, 128): 7,
      (64, 0, 0): 8,
      (192, 0, 0): 9,
      (64, 128, 0): 10,
      (192, 128, 0): 11,
      (64, 0, 128): 12,
      (192, 0, 128): 13
    }

    self.y_train_labels = np.zeros((n, n_patches))
    self.y_test_labels = np.zeros((m, n_patches))
    for i in range(n):
      for j in range(n_patches):
        patch = self.y_train_patches[i,j,:,:]
        vals, counts = np.unique(patch, axis=0, return_counts=True)
        self.y_train_labels[i,j] = labels[tuple(reversed(vals[-1]))]
    for i in range(m):
      for j in range(n_patches):
        patch = self.y_test_patches[i,j,:,:]
        vals, counts = np.unique(patch, axis=0, return_counts=True)
        self.y_test_labels[i,j] = labels[tuple(reversed(vals[-1]))]

  def decode_y(self, pred):
    """ Turn class labels into pixel space. """
    labels = np.array([
      (0, 0, 0),
      (128, 0, 0),
      (0, 128, 0),
      (128, 128, 0),
      (0, 0, 128),
      (128, 0, 128),
      (0, 128, 128),
      (128, 128, 128),
      (64, 0, 0),
      (192, 0, 0),
      (64, 128, 0),
      (192, 128, 0),
      (64, 0, 128),
      (192, 0, 128),
    ])
    return labels[pred.astype('uint8')]

  def get_x(self, i):
    """ Get training X image by index. """
    return self.process(
      self.X_train[i,:,:,:]
    )

  def get_y(self, i):
    """ Get training Y image by index. """
    return self.process(
      self.y_train[i,:,:,:]
    )

  def display(self, i):
    fig, (x,y) = plt.subplots(1, 2)
    x.imshow(self.get_x(i))
    x.axis('off')
    y.imshow(self.get_y(i))
    y.axis('off')
    plt.show()

  def sift_descriptors(self, test=False):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1, contrastThreshold=1e-12, edgeThreshold=1e6)
    X = self.X_test_patches if test else self.X_train_patches
    sd = SIFTDescriptor(patchSize=self.patch_dim)

    N, n_patches, patch_size, d = X.shape
    for n in range(N):
      for p in range(n_patches):
        patch = X[n,p,:,:].reshape((self.patch_dim, self.patch_dim, 3))
        patch = patch.astype('uint8')
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        desc = sd.describe(gray_patch)
        if test:
          self.X_test_sift[n,p] = self.sift_dictionary.predict(desc.reshape(1,-1))
        else:
          feat_bank_row = p * n + p;
          self.X_train_sift[n,p] = feat_bank_row
          self.X_sift_bank[feat_bank_row,:] = desc

      self.verbose and print(f'Computing SIFT for image {n+1}')

    self.verbose and print("Computed SIFT descriptors.")

  def learn_sift_dictionary(self):
    """ Learn keypoint dictionary by using kmeans clustering with 1000 clusters. """
    self.verbose and print("Learning SIFT dictionary...")
    self.sift_dictionary = KMeans(1000, random_state=0).fit(self.X_sift_bank)
    self.X_train_sift = self.sift_dictionary.labels_
    self.verbose and print("Learned SIFT dictionary.")

  def learn_hue_dictionary(self):
    """ Learn keypoint dictionary by using kmeans clustering with 1000 clusters. """
    self.verbose and print("Learning hue dictionary...")
    self.hue_dictionary = KMeans(100, random_state=0).fit(self.X_hue_bank)
    self.X_train_hue = self.hue_dictionary.labels_
    self.verbose and print("Computed hue dictionary.")

  def hue_descriptors(self, test=False):
    """ Compute hue descriptors. """
    X = self.X_test_patches if test else self.X_train_patches

    N, n_patches, patch_size, d = X.shape
    for n in range(N):
      for p in range(n_patches):
        patch = X[n, p, :, :].reshape((self.patch_dim, self.patch_dim, 3))
        desc = self.hue_descriptor(patch)
        if test:
          self.X_test_hue[n,p] = self.hue_dictionary.predict(desc.reshape(1,-1))
        else:
          feat_bank_row = p * n + p;
          self.X_hue_bank[feat_bank_row,:] = desc

      self.verbose and print(f'Computing hue for image {n+1}')

    self.verbose and print("Computed hue descriptors.")

  def hue_descriptor(self, patch, number_of_bins=36):
    """ Calculate 36-dimensional hue descriptor. """
    pr = patch[:,:,0]
    pg = patch[:,:,1]
    pb = patch[:,:,2]
    lo, hi = -2.5, 2.5
    xx, yy = np.mgrid[lo:hi,lo:hi]
    spatial_weights = np.exp((-(xx**2) - (yy**2)) / 50)

    out = np.zeros((number_of_bins, pr.shape[0]))
    H = np.arctan2(pr+pg-2*pb, np.sqrt(3)*(pr-pg)) + np.pi
    H[np.isnan(H)] = 0

    saturation=(2/3*(pr**2+pg**2+pb**2-pr*(pg+pb)-pg*pb)+0.01)
    weights = np.ravel(spatial_weights)
    if weights.shape[0] > pr.shape[1]:
      weights = weights[:pr.shape[1]]
    RGB_energy=(np.sum((pr**2+pg**2+pb**2)*(weights*np.ones((1,pr.shape[1])))))

    H = np.floor(H/(2*np.pi)*(number_of_bins))
    for jj in range(number_of_bins):
      term1 = saturation
      term2 = weights * np.ones((1, pr.shape[1]))
      term3 = (H==jj)
      termsum = np.sum(term1 * term2 * term3, axis=1)
      out[jj,:] = termsum

    eps = 1e-3
    out = np.sqrt(out.T / (np.ones((out.shape[1],1)) * RGB_energy + eps)).T

    return np.sum(out, axis=1)

  def position_descriptors(self, test=False):
    """ Assign every patch a row and column position in an 8x8 grid. """
    if test:
      self.X_test_pos = np.repeat(np.arange(160), self.X_test.shape[0])
    else:
      self.X_train_pos = np.repeat(np.arange(160), self.X_train.shape[0])

    self.verbose and print("Computed position descriptors.")


  def bitwise_features(self, test=False):
    """ Generate bitwise features for SIFT, hue and position. """
    N = self.X_test.shape[0] if test else self.X_train.shape[0]
    M = 1000 + 100 + 160
    if test:
      self.X_test_feat = np.full((N * self.num_patches, M), 0, dtype=bool)
      for i in range(self.X_test_feat.shape[0]):
        sift = int(self.X_test_sift.reshape(-1)[i])
        hue = int(self.X_test_hue.reshape(-1)[i])
        pos = self.X_test_pos[i]
        self.X_test_feat[i,sift] = True
        self.X_test_feat[i,1000+hue] = True
        self.X_test_feat[i,1100+pos] = True
      self.X_test_feat.reshape((N, self.num_patches, M))
    else:
      self.X_train_feat = np.full((N * self.num_patches, M), 0, dtype=bool)
      for i in range(self.X_train_feat.shape[0]):
        sift = self.X_train_sift[i]
        hue = self.X_train_hue[i]
        pos = self.X_train_pos[i]

        self.X_train_feat[i,sift] = True
        self.X_train_feat[i,1000+hue] = True
        self.X_train_feat[i,1100+pos] = True
      self.X_train_feat.reshape((N, self.num_patches, M))


