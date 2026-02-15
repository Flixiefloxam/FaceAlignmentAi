import numpy as np
import os
import sys
from skimage.transform import resize
from skimage.feature import hog
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import time

def loadData():
    #Change working directory to the script's directory so the data files can be found(For some reason it sometimes wasn't there already for me)
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    # Load the data using np.load
    data = np.load('face_alignment_training_images.npz', allow_pickle=True)

    # Extract the images
    images = data['images']
    # and the data points
    pts = data['points']

    print(images.shape, pts.shape)

    test_data = np.load('face_alignment_test_images.npz', allow_pickle=True)
    test_images = test_data['images']
    print(test_images.shape)

    return images, pts, test_images

def visualise_pts(img, pts):
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+r')
  plt.show()

def show_predictions(images, predicted_points, n=3):#n is how many predicted images to show
    for i in range(n):
        idx = np.random.randint(0, images.shape[0])
        img = images[idx]
        pts = predicted_points[idx]

        plt.imshow(img, cmap='gray')
        plt.scatter(pts[:, 0], pts[:, 1], c='red', marker='+')
        plt.title(f'Predicted landmarks for test image {idx}')
        plt.axis('off')
        plt.show()

def extractHogFeatures(images):
    return np.array([
        hog(rgb2gray(img) if img.ndim == 3 else img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        for img in images
    ])

def calculateValError(yPred, yVal):
  #Reshape predictions and ground truth
  yValPred = yPred.reshape(-1, 5, 2)
  yValTrue = yVal.reshape(-1, 5, 2)

  #Compute per-image mean Euclidean distance
  valErrors = np.array([np.mean(euclid_dist(p, g)) for p, g in zip(yValPred, yValTrue)])

  #Report average error
  print(f'Mean validation Euclidean error: {np.mean(valErrors):.2f} pixels')

  #plot error distribution
  plt.hist(valErrors, bins=20)
  plt.title('Validation Euclidean Distance per Image')
  plt.xlabel('Mean Error (pixels)')
  plt.ylabel('Frequency')
  plt.show()

  return valErrors

def displayWorstPredictions(valErrors, yPred, yVal, resizedImages, n=3):
  worstIndices = np.argsort(valErrors)[-3:]  # highest error
  yValPred_reshaped = yPred.reshape(-1, 5, 2)
  yValTrue_reshaped = yVal.reshape(-1, 5, 2)

  for idx in worstIndices:
      print(f"Image {idx} - Error: {valErrors[idx]:.2f}")
      
      plt.imshow(resizedImages[idx], cmap='gray')
      plt.scatter(yValTrue_reshaped[idx][:, 0], yValTrue_reshaped[idx][:, 1], c='green', label='True')
      plt.scatter(yValPred_reshaped[idx][:, 0], yValPred_reshaped[idx][:, 1], c='red', label='Predicted')
      plt.legend()
      plt.title(f"Validation Sample {idx}")
      plt.axis('off')
      plt.show()
  thresholds = np.linspace(0, 20, 100)
  cumulativeAccuracy = [(valErrors < t).mean() for t in thresholds]

  plt.plot(thresholds, cumulativeAccuracy)
  plt.xlabel('Error threshold (pixels)')
  plt.ylabel('Cumulative accuracy')
  plt.title('Cumulative Error Distribution on Validation Set')
  plt.grid(True)
  plt.show()

def evaluate_models(XTrain, XVal, yTrain, yVal):
    models = {
        'Ridge (alpha=1.0)': Ridge(alpha=1.0),
        'Lasso (alpha=0.01)': Lasso(alpha=0.01, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, n_jobs=-1),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    }

    results = []

    for name, model in models.items():
        print(f'\nTraining {name}...')
        startTime = time.time()
        model.fit(XTrain, yTrain)
        endTime = time.time()
        print(f'Training time: {endTime - startTime:.2f} seconds')
        yPred = model.predict(XVal)

        mse = np.mean((yPred - yVal) ** 2)
        eucl_errors = np.array([
            np.mean(euclid_dist(p, g)) for p, g in zip(yPred.reshape(-1, 5, 2), yVal.reshape(-1, 5, 2))
        ])
        mean_eucl = np.mean(eucl_errors)

        results.append((name, mse, mean_eucl))
        print(f'{name} - MSE: {mse:.4f}, Mean Euclidean Error: {mean_eucl:.2f} px')

    return results

def compareModels(results):
    names = [r[0] for r in results]
    mse_vals = [r[1] for r in results]
    eucl_vals = [r[2] for r in results]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, mse_vals, width, label='MSE')
    plt.bar(x + width/2, eucl_vals, width, label='Euclidean Error')
    plt.xticks(x, names, rotation=15, ha='right')
    plt.title('Model Comparison on Validation Set')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.show()
   

def euclid_dist(pred_pts, gt_pts):
  """
  Calculate the euclidean distance between pairs of points

  :param pred_pts: The predicted points
  :param gt_pts: The ground truth points
  :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
  """
  import numpy as np
  pred_pts = np.reshape(pred_pts, (-1, 2))
  gt_pts = np.reshape(gt_pts, (-1, 2))
  return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))

def save_as_csv(points, location = '.'):
    """
    Save the points out as a .csv file

    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==5*2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

images, pts, testImages = loadData()

for i in range(0):
  idx = np.random.randint(0, images.shape[0])
  visualise_pts(images[idx, ...], pts[idx, ...])

#resize images to 64x64
resizedImages = np.array([resize(img, (64, 64)) for img in images])
scaleFactor = 64 / images.shape[1]  # assuming square images
scaledPoints = pts * scaleFactor

X = extractHogFeatures(resizedImages)
y = scaledPoints.reshape((scaledPoints.shape[0], -1))  # Flatten landmarks

XTrain, XVal, yTrain, yVal = train_test_split(X, y, test_size=0.1)

results = evaluate_models(XTrain, XVal, yTrain, yVal)
compareModels(results)


testImagesResized = np.array([resize(img, (64, 64)) for img in testImages])
XTest = extractHogFeatures(testImagesResized)

best_model = Ridge(alpha=1.0)
best_model.fit(XTrain, yTrain)
yTestPred = best_model.predict(XTest)
yTestPred = yTestPred.reshape((-1, pts.shape[1], 2)) / scaleFactor  # Rescale to original

show_predictions(testImages, yTestPred)

# Test performance
yPred = best_model.predict(XVal)
valErrors = calculateValError(yPred, yVal)
displayWorstPredictions(valErrors, yPred, yVal, resizedImages)

# Save to CSV
save_as_csv(yTestPred)