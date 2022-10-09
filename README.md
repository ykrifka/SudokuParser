# SudokuParser
The purpose of this Python project is to parse Sudokus from images using TensorFlow. 

More precisely, we generate random images of Sudokus which are then projectively distorted.
Keeping track of the corners of the Sudokus we obtain our first dataset.
This is then fed into a deep neural network to locate the corners of the sudoku in the image.
The information of where the corners of a Sudoku are in a given image then allows us to apply another projective transformation to straighten out the image such that the Sudoku has a square shape.

In a second step, we crop the entries of such random straightened out Sudokus. 
This yields a second dataset containing images of entries (numbers) and their classification. 
We then feed this dataset into a second neural network to learn the OCR-task of recognizing the entries of a Sudoku.

By combining both steps we succeed in reading the entries from a projectively distorted image of a Sudoku.
