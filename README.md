# SudokuParser
The purpose of this Python project is to parse Sudokus from images using TensorFlow. 

More precisely, we generate random images of Sudokus which are then projectively distorted.
Keeping track of the corners of the Sudokus we obtain our first dataset.
This is then fed into a deep neural network to locate the corners of the Sudoku in the image.
The information of where the corners of a Sudoku are in a given image then allows us to apply another projective transformation to straighten out the image such that the resulting Sudoku has a square shape.

In a second step, we crop the entries of such random straightened out Sudokus. 
This yields a second dataset containing images of entries and their classification as integers. 
We then feed this dataset into a second neural network to learn the OCR-task of recognizing the entries as integers.

By combining both steps we succeed in reading the entries from a projectively distorted image of a Sudoku.
