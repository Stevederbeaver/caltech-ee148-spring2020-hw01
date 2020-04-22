import os
import numpy as np
import json
from PIL import Image

# Read the json file to get the boxes
with open('../data/hw01_preds/preds.json') as f:
    bounding_boxes_list = json.load(f)

# Get sorted list of files:
file_names = sorted(os.listdir(data_path))

# Remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# Visualizing the boxes on each image
for i in range(len(file_names)):#len(file_names)):
    # Tracking the loop
    print(i)

    # Read the matrices and corresponding boxes:
    I = np.asarray(Image.open(os.path.join(data_path,file_names[i])))
    I_matrix = np.copy(I)
    bounding_boxes = bounding_boxes_list[file_names[i]]

    # Display the bounding boxes
    for box in bounding_boxes:
        for j in range(box[0],box[2]):
            I_matrix[j][box[1]] = ([255,0,0])
            I_matrix[j][box[3]-1] = ([255,0,0])
        for k in range(box[1],box[3]):
            I_matrix[box[0]][k] = ([255,0,0])
            I_matrix[box[2]-1][k] = ([255,0,0])

    # Draw and save the image
    I_image = Image.fromarray(I_matrix, 'RGB')
    I_image.show()
    I_image.save('Redlight-RL-' + str(i) + '.jpg')
