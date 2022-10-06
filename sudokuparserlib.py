""" Contains auxilliary functions for the SudokuParser Project
"""

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


import tensorflow as tf
from tensorflow import keras


def draw_sudoku(im, A, pos, width):
  """ Draw a sudoku on a PIL image object

  Args:
    im (Image): PIL image object to draw on
    A (np.array): Numpy Array of shape (9,9) containing the entries of the Sudoku
    pos (tuple): Tuple of two integers (top, left) determining the top-left corner of the Sudoku in im
    width (int): Width of the Sudoku

  Returns:
    None
  """

  # denote by left, top the coordinates of the top-left corner (=pos)
  left, top = pos
  # delta will denote the width (and height) of one of the squares of the sudoku
  delta = width // 9
  # shrink the width of the sudoku to one that is divisible by 9 and call it w
  w = delta * 9

  draw = ImageDraw.Draw(im)

  # draw a white background rectangle for the sudoku
  draw.rectangle([left, top, left+w, top+w], "white")

  # draw 10 vertical lines
  for i in range(10):
    # usually the width of line i is 1
    if i % 3:
      line_width = 1
    # unless i is divisible by 3 then draw it thicker
    else:
      line_width = 2
    x = left + i*delta
    draw.line([x,top,x,top+w], "black", width=line_width)
  

  # draw 10 horizontal lines
  for j in range(10):
    # usually the width of line j is 1
    if j % 3:
      line_width = 1
    # unless j is divisible by 3 then draw it thicker
    else:
      line_width = 2
    y = top + j * delta
    draw.line([left,y, left+w, y], "black", width=line_width)
  
  # load the louis font
  font = ImageFont.truetype("./Fonts/louis_george_cafe/louis.ttf", size=int(0.9*delta))

  # insert the numbers in each square
  for j in range(9):
    for i in range(9):
      x = left + (delta//3) + i * delta
      y = top + (delta//10) + j * delta
      if A[j,i]:
        draw.text((x,y), str(A[j,i]), font = font, fill = "black")


def find_projective_data(dom, tar):
  """ Computes projective data for the PERSPECTIVE (projective) transfrom of an Image object.

  Args:
    dom (List[List[int]]): List of four points to be transformed
    tar (List[List[int]]): List of four points that dom will be projectively transformed to 

  Returns:
    List[float]: List of projective transformation data ready to be fed into Image.transform
                 method when mode set to Image.PERSPECTIVE
  """

  A0 = np.concatenate((np.array(dom), np.ones((4,1))),axis=1)
  A1 = A0[:3].transpose()
  p3 = A0[3]
  coeffA = np.linalg.solve(A1,p3)
  A2 = np.matmul(A1, np.diag(coeffA))
  
  B0 = np.concatenate((np.array(tar), np.ones((4,1))),axis=1)
  B1 = B0[:3].transpose()
  q3 = B0[3]
  coeffB = np.linalg.solve(B1,q3)
  B2 = np.matmul(B1, np.diag(coeffB))

  M = np.matmul(B2,np.linalg.inv(A2))
  M = M/M[2,2]
  return [M[0,0],M[0,1],M[0,2],M[1,0],M[1,1],M[1,2],M[2,0],M[2,1]]


def generate_random_mock_sudoku(size=(200,200)):
  """ Generates a random projectively distorted sudoku image

  Args:
    size = (200,200): Tuple (width, height) determining the size of the generated image

  Returns:
    tuple[Image, np.array, np.array]: A tuple (im, coords, A) where im is an Image object
      of the given size on which a random Sudoku was drawn, coords is a Numpy array of shape
      (8,) holding the coordinates of the projectively distorted sudoku, and A is a Numpy array 
      of shape (9,9) holding the entries of the sudoku.
  """

  # Initialize array with zeros
  A = np.zeros((9,9),int)
  # Choose random random probability to put a number
  p = np.random.rand()
  
  for i in range(9):
    for j in range(9):
      # With probability p put a random number between 1 and 9 in A[i,j]
      if np.random.rand() < p:
        A[i,j] = np.random.randint(1,10)
  
  im = Image.new("RGBA", size)

  # the width should be 80% of the image size
  width = (4 * np.min(size)) // 5

  # to avoid rounding errors later on we make sure that the width is actually a multiple of 9
  width = (width // 9) * 9

  # set the margin to 10% of the image size
  margin = np.min(size)//10

  # we start in the upper left corner (pos)
  pos = np.array([margin, margin])

  # and draw a sudoku with entries A at position pos with width width
  draw_sudoku(im, A, pos, width)

  # define the coordinates of the four corner points of the sudoku as p0, ..., p4
  p0 = pos
  p1 = pos + np.array([width,0])
  p2 = pos + np.array([0,width])
  p3 = pos + np.array([width,width])

  # wiggle those corners a little bit by adding a random number in [-margin, margin]
  q0 = p0 + np.random.randint(-margin,margin,size=2)
  q1 = p1 + np.random.randint(-margin,margin,size=2)
  q2 = p2 + np.random.randint(-margin,margin,size=2)
  q3 = p3 + np.random.randint(-margin,margin,size=2)

  # apply a projective transformation to the sudoku, that sends pi to qi

  data = find_projective_data([q0,q1,q2,q3],[p0,p1,p2,p3])
  im = im.transform(size, Image.PERSPECTIVE, data = data, resample = Image.BICUBIC)

  # return the image, the corners of the transformed image and the sudoku numbers as an array
  return im, np.concatenate([q0,q1,q2,q3], dtype = float), A


def generate_data(N = 1000, size=(200,200)):
  """ Generates training data to learn sudoku corners

  Args:
    N = 1000: Number of sudokus to generate as training data
    size = (200, 200): Size of sudokus to generate

  Returns:
    Tuple[np.array, np.array]: Returns a tuple (X, y) where X is a Numpy array
      of shape (N, *size, 1) holding the generated grayscale images of sudokus
      and y is a Numpy array of shape (N,8) holding the corresponding 
      corner coordinates.

  """

  X, y = [], []
  for _ in range(N):
    im, corners, _ = generate_random_mock_sudoku(size)
    X.append(img_to_array(im))
    y.append(corners)
  
  return np.array(X), np.array(y)



def img_to_array(im):
  """ Returns the given image as grayscale array with pixel brightness scaled to [0,1]

  Args:
    im (Image): Original PIL Image object to transform to an array

  Returns:
    np.array: Corresponding Numpy array of shape im.size with entries in [0,1].
  """
  return np.expand_dims(np.array(im.convert("L"), dtype = float)/255, axis = -1)


class preprocess_sudoku(keras.layers.Layer):
  """ Custom Keras Preprocessing Layer wrapping the functionality the function 'img_to_array'

  Methods:
  --------

    call(input): Transforms input Image object to grayscale Numpy array like 'img_to_array'

  """

  def call(self, input):
    """ Returns the given image as grayscale array with pixel brightness scaled to [0,1]

    Args:
      input (Image): Original PIL Image object to transform to an array

    Returns:
      np.array: Corresponding Numpy array of shape im.size with entries in [0,1].
    """

    return np.expand_dims(np.array(input.convert("L"), dtype = float)/255, axis = -1)


def array_to_img(A):
  """ Reverses the effect of 'img_to_array' returning the Image object corresponding to A

  Args:
    A (np.array): Numpy array with entries in [0, 1] describing the pixel brightness

  Returns:
    Image: Grayscale PIL Image object corresponding to the Numpy array A
  """

  return Image.fromarray(np.uint8(255 * A.squeeze(axis = -1)), mode = "L").convert("RGB")


def draw_polygon(im, y, **kwargs):
  """ Draws a trapezoid with corner coordinates y onto the Image im.

  Args:
    im (Image): PIL Image object to draw on
    y (List[int]): List of length 8 holding the corner coordinates of the trapezoid
    **kwargs: Additional options that are past to the ImageDraw.draw.line method

  Returns:
    None
  """

  draw = ImageDraw.Draw(im)

  # read the four corner points
  p = [(int(y[i]), int(y[i+1])) for i in range(0, 8, 2)]

  # draw the bounding polygon
  # the order 0, 1, 3, 2 is because the neural net returns the coords in this order
  draw.line([ p[0], p[1]], **kwargs)
  draw.line([ p[1], p[3]], **kwargs)
  draw.line([ p[3], p[2]], **kwargs)
  draw.line([ p[2], p[0]], **kwargs)



def transform_back(A, z, margin = 2, size = (200, 200)):
  """ Projectively transforms the grayscale image given as a Numpy array A such that 
  the points given in z are sent to the corners of the new image

  Args:
    A (np.array): Grayscale image given as a Numpy array A with pixel brightness in [0,1]
    z (List[int]): List of length 8 holding the coordinates of the points that will be
      mapped to the corners of the new image
    margin = 2: Margin size of the resulting image
    size = (200, 200): Size of the resulting image

  Returns:
    Image: Returns an Image object obtained from mapping the points given by z to the corners
  """

  width = ((size[0]-2*margin)//9) * 9
  q0 = np.array([margin,margin])
  q1 = q0 + np.array([width, 0])
  q2 = q0 + np.array([0, width])
  q3 = q0 + np.array([width, width])

  p = [ np.array([z[i], z[i+1]]) for i in range(0,8,2)]

  data = find_projective_data([q0,q1,q2,q3], p)

  im = Image.fromarray(np.uint8(255 * A.squeeze(axis = -1)), mode = "L")
  im = im.transform(size, Image.PERSPECTIVE, data = data, resample = Image.BICUBIC)
  return im


def crop_sudoku(im, margin=2):
  """ Given an image im (Image) of a sudoku filling the entire image we return a list of smaller cropped 
    images of the sudokus fields/squares.

    Args:
      im (Image): PIL Image object of a sudoku filling the entire image
      margin = 2: Margin to enlarge the standard squares of the entries by. This
        is useful to compensate for some error in the corner detection.

    Returns:
      List[Image]: List of Image objects of the entries of the sudoku
  """

  crops = []
  size = im.size
  delta = (size[0] - 2*margin) // 9
  width = delta * 9
  for j in range(9):
    for i in range(9):
      crops.append(im.crop(
          [i * delta , j * delta, (i + 1) * delta + 2*margin, (j + 1) * delta + 2*margin]
      ))
  
  return crops