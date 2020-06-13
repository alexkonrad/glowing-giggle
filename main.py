# TODO: encode position vector
# TODO: encode three binary feature vectors
# TODO: train on all training data and save dictionaries to disk
# TODO: create naive bayes predictor
# TODO: visualize result
# TODO: measure error
# TODO: write report
import pickle


#%% Setup scene classifier object
from scene.classifier import *
sc = SceneClassifier()

#%% Load classifier data from file
sc.load()

#%% Compute test features
sc.compute_features(test=True)

#%% Make predictions

pred = sc.predict(6)
img = sc.decode_y(pred).reshape(16, 10, 3)

plt.figure;
plt.imshow(img)
plt.show();

# pred = sc.clf.predict(sc.X_test_features[0].reshape(-1))

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

#%% Compute SIFT descriptors
sc.sift_descriptors()
sc.learn_sift_dictionary()
sift_dict = sc.sift_dictionary
with open('sift.pickle', 'wb') as f:
    pickle.dump(sift_dict, f)
# clf2 = pickle.loads(s)
# TODO: plot sc.X_train_sift

#%% Compute hue descriptors
sc.hue_descriptors()
sc.learn_hue_dictionary()
hue_dict = sc.hue_dictionary
with open('hue.pickle', 'wb') as f:
  pickle.dump(hue_dict, f)
# TODO: plot sc.X_train_hue

#%% Compute position descriptors
sc.position_descriptors()

#%% Compute bitwise feature vectors
sc.bitwise_features()

with open('features.pickle', 'wb') as f:
  pickle.dump(sc.X_train_feat, f)

#%% Setup Y vectors

sc.encode_y()

#%% Fit classifier

sc.fit()

#%% Save classifier

with open('clf.pickle', 'wb') as f:
  pickle.dump(sc.clf, f)
