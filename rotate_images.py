import os
import cv2

# Set the directory containing the images
image_dir = '/home/asl-student/Desktop/Smazzucco/ssn-pytorch/BSR/BSDS500/data/images/train'

# Iterate through all the files in the directory
for file in os.listdir(image_dir):
  # Check if the file is a valid image
  if file.endswith(".jpg") or file.endswith(".png"):
    # Load the image
    image = cv2.imread(os.path.join(image_dir, file))
    # Check if the image needs to be rotated
    print(image.shape)
    if image.shape[0] > image.shape[1]:
      # Rotate the image by 90 degrees
      image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
      # Save the rotated image
      cv2.imwrite(os.path.join(image_dir, file), image)
    
print("HERE")
