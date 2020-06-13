from scene.data import *
from scene.sift import SIFTDescriptor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import cv2

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

  def process(self, image):
    """ Convert image to format for display. """
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB)
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return image

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

  def compute_features(self):
    pass

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
          # self.X_test_sift[n,p] = self.sift_dictionary.predict(desc)
          pass
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
          # self.X_test_hue[n,p] = self.hue_dictionary.predict(desc)
          pass
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

  def position_feature(self, X, n_patches=266, n_cells=8):
    """ Assign every patch to a cell in a grid of the image. """
    h, w, d = X.shape
    quo, rem = divmod(n_patches, n_cells)
    cell_labels = np.repeat(np.arange(1, n_cells+1), quo)
    extra_cell_labels = np.repeat(1, rem)
    cell_labels = np.sort(np.concatenate((cell_labels, extra_cell_labels)))
    return cell_labels


  # def keypoints(self, i):
  #   """ Generate SIFT descriptors. """
  #   sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=.01, edgeThreshold=20)
  #   img = self.X_train[i,:,:,:].astype('uint8')
  #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #   # kp = sift.detect(gray, None)
  #   kp, des = sift.detectAndCompute(gray,None)
  #   img = cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), kp, img)
  #   return kp, des, img



