# TODO: encode position vector
# TODO: encode three binary feature vectors
# TODO: train on all training data and save dictionaries to disk
# TODO: create naive bayes predictor
# TODO: visualize result
# TODO: measure error
# TODO: write report


#%% Setup scene classifier object
from scene.classifier import *
sc = SceneClassifier(limit=12)

#%% Compute SIFT descriptors
sc.sift_descriptors()
sc.learn_sift_dictionary()
# TODO: plot sc.X_train_sift

#%% Compute hue descriptors
sc.hue_descriptors()
sc.learn_hue_dictionary()
# TODO: plot sc.X_train_hue

#%% Compute position descriptors
sc.position_descriptors()
sc.learn_position_dictionary()


#%% Display X and Y training exampless
sc.display(0)
sc.display(1)

#%% Get keypoints and features
_, keypoint_descriptor, _ = sc.keypoints(2)
hue_descriptor = sc.hue_feature(sc.X_train[0])
position_descriptor = sc.position_feature(sc.X_train[0])
print(
  keypoint_descriptor.shape,
  position_descriptor.shape,
  hue_descriptor.shape
)

#%% oK

img = sc.X_train[0]
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
plt.imshow(img)
plt.show()

#%% OK

print("HEL")