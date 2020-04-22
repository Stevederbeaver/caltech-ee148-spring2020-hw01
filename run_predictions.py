import os
import numpy as np
import json
from PIL import Image

def dot_product(A, B):
    '''
    This function takes two matrices A, B of the same size and returns their
    dot product (sum of the entries in their Hadamard product)
    '''
    return sum(sum(sum(np.multiply(A, B))))

def normalize(A):
    '''
    This function takes an arbitrary matrix A as the input and normalize it
    '''
    A1 = A / 255.0
    coeff = np.sqrt(dot_product(A1, A1))
    B = A1 / coeff
    return B

def check_intersecting(A, given_list):
    '''
    This function takes a "box" A and a list of boxes as the input. It check
    whether A intersects with any box in the given list
    '''
    for B in given_list:
        if A[0] >= B[2] or B[0] >= A[2]:
            continue
        elif A[3] <= B[1] or B[3] <= A[1]:
            continue

        return True

    return False

def detect_red_light(I, filter, threshold):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below.

    '''
    BEGIN YOUR CODE

    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        bounding_boxes.append([tl_row,tl_col,br_row,br_col])

    '''
    x, y, z = I.shape
    dim_x, dim_y, dim_z = filter.shape
    for i in range(x - dim_x + 1):
        for j in range(y - dim_y + 1):
            box1 = [i, j, i + dim_x, j + dim_y]
            I_transform = normalize(I[i:i + dim_x, j:j + dim_y, :])
            # Only append the box under certain conditions
            if dot_product(I_transform, filter) > threshold:
                if check_intersecting(box1, bounding_boxes) == False:
                    bounding_boxes.append(box1)
    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../data/hw01_preds'
os.makedirs(preds_path) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]
# Initialization
preds = {}
threshold = 0.93
Image_sample= np.asarray(Image.open(os.path.join(data_path,file_names[0])))
red_light_sample = Image_sample[155:171, 316:323, :]
red_light_filter = normalize(red_light_sample)

for i in range(len(file_names)):
    print(i)

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    # Append the list of boxes
    preds[file_names[i]] = detect_red_light(I, red_light_filter, threshold)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
