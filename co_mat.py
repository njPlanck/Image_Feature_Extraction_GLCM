# importing libraries 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#load image
img = Image.open('images.jpg').convert('L')
img_arr = np.array(img)

'''
#printing thing oout the the image and the image array
print(img_arr)
plt.imshow(img)
plt.show()

'''

#calculating the co-occurence matrix with a function
def cal_comat(img,distance,angle):
  img = np.pad(img,1,mode='constant')
  rows, cols = img.shape
  glcm = np.zeros((256,256)) #initialising the gray level co-occurence matrix glcm to zeros

  for i in range(1,rows-1):
    for j in range(1,cols-1):
      if angle == 0:
        neighbour_i, neighbour_j = i,j+distance
      elif angle == 45:
        neighbour_i,neighbour_j = i+distance,j+distance
      elif angle == 90:
        neighbour_i,neighbour_j = i+distance,j
      elif angle == 135:
        neighbour_i,neighbour_j = i+distance,j-distance

      if 1 <= neighbour_i<rows-1 and 1 <= neighbour_j<cols-1:
        glcm[img[i,j],img[neighbour_i,neighbour_j]] += 1
  glcm /= np.sum(glcm)
  return glcm

#calculate the co-occurence matrix for different angles

distance = 1
angles = [0,45,90,135]
co_occurence_matrices = {}

for angle in angles:
  co_occurence_matrices[angle] = cal_comat(img_arr,distance,angle)


#calculate matrix features from the co-occurence matrix

def cal_feat(glcm):
  contrast = np.sum(glcm*np.sqrt(np.sum(glcm,axis=1)))
  dissimilarity = np.sum(glcm*np.abs(np.arange(256)-np.arange(256)[:,None]))
  homogeneity = np.sum(glcm/(1+np.abs(np.arange(256)-np.arange(256)[:,None])))
  asm = np.sum(glcm**2)
  energy = np.sqrt(np.sum(glcm**2))
  correlation = np.sum(glcm*(np.arange(256)*np.arange(256)[:,None]))

  return {'contrast':contrast,'dissimilarity':dissimilarity,'homogeneity':homogeneity,'asm':asm,'energy':energy,
          'correlation':correlation}

features = {}

for angle, glcm, in co_occurence_matrices.items():
  features[angle] = cal_feat(glcm)


'''
#print features
for angle, feat_values in features.items():
  print(f"Angle: {angle}")
  for feature_name, value  in feat_values.items():
    print(f"{feature_name} :{value}")
  print("---------------------------------")


plt.imshow(img_arr,cmap="gray")
plt.show()

'''


