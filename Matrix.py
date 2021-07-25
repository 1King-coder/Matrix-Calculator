from random import randint
from typing import Union

class Matrix:
    """
    Matrix Object supporting
    most of basic matrix operations.
    """
    def __init__(self, arg: Union[list, tuple]) -> None:
        """
        Can create a Matrix Object with
        a sent matrix array or create one
        with random numbers with the size sent
        in a tuple (rows, columns).
        """
        if type(arg) == list:
            self.matrix: list = arg

        if type(arg) == tuple:
            self.matrix: list = Matrix.gen_matrix(arg[0], arg[1]).matrix

        self.rows_num = len(self.matrix)
        self.cols_num = len(self.matrix[0])


    @staticmethod
    def map_matrix(func, rows: int, cols: int) -> 'Matrix':
        """
        Receive a function and return
        a new result Matrix.
        """

        new_matrix = []
        
        for i in range(rows):
            new_matrix.append([])

            for j in range(cols):
                new_matrix[i].append(func(i, j))

        return Matrix(new_matrix)
    
    @staticmethod
    def gen_matrix(rows: int, columns: int) -> list:
        """
        Generate a matrix with
        the sent size and random numbers
        between 0 and 100.
        """
        return Matrix.map_matrix(
            lambda i, j: randint(0, 100), 
            rows, 
            columns
        )
        
    def __str__(self) -> str:
        """
        Pretty print a Matrix.
        """
        print()
        matrix_str = ''
        
        for i in range(self.rows_num):
            matrix_str += '|'

            for j in range(self.cols_num):
                if j == self.cols_num-1:
                    matrix_str += f'{self.matrix[i][j]:3}'
                    continue

                matrix_str += f'{self.matrix[i][j]:3} '
            matrix_str += '|\n'

        return matrix_str
    
    def __add__(self, other_matrix: 'Matrix') -> 'Matrix':

        """
        Allow to add Matrices with operator.
        """

        if len(other_matrix.matrix) != self.rows_num or len(other_matrix.matrix[0]) != self.cols_num:
            raise IndexError('Matrices must be of the same size.')

        
        return Matrix.map_matrix(
            lambda i, j: 
                self.matrix[i][j] + other_matrix.matrix[i][j],
            self.rows_num,
            self.cols_num
        )
    
    def __sub__(self, other_matrix: 'Matrix') -> 'Matrix':

        """
        Allow to subtract Matrices with operator.
        """

        if len(other_matrix.matrix) != self.rows_num or len(other_matrix.matrix[0]) != self.cols_num:
            raise IndexError('Matrices must be of the same size.')

        new_matrix = []

        return Matrix.map_matrix(
            lambda i, j: self.matrix[i][j] - other_matrix.matrix[i][j],
            self.rows_num,
            self.cols_num
        )
    
    def __mul__(self, param: Union['Matrix', int, float]) -> 'Matrix':
        
        """
        Allow to multiply Matrices or
        a Matrix by a real number with operator.
        """

        matrix_1 = Matrix(self.matrix)
        
        if type(param) == Matrix:

            if self.cols_num != param.rows_num:
                raise IndexError("The number of columns of a matrix must be equal to the other's number of rows.")

            return Matrix.mul_matrices(matrix_1, param)
        
        return Matrix.mul_matrix_by_number(matrix_1, param)
        
    
    @staticmethod
    def mul_matrices(matrix_1: 'Matrix', matrix_2: 'Matrix') -> 'Matrix':
        """
        Multiply a Matrix by other.
        """

        def multply_matrices (i, j, num = 0):
            for k in range(matrix_1.cols_num):
                num += matrix_1.matrix[i][k] * matrix_2.matrix[k][j]
            return num

        return Matrix.map_matrix(
            multply_matrices,
            matrix_1.rows_num,
            matrix_2.cols_num
        )
        

    @staticmethod
    def mul_matrix_by_number(matrix: 'Matrix', num: Union[int, float]) -> 'Matrix':
        """
        Multiply a Matrix by a real number.
        """
        
        return Matrix.map_matrix(
            lambda i, j: matrix.matrix[i][j] * num,
            matrix.rows_num,
            matrix.cols_num
        )

    @staticmethod
    def transpose(matrix: 'Matrix') -> 'Matrix':
        """
        Transpose a Matrix.
        """
        return Matrix.map_matrix(
            lambda i,j: matrix.matrix[j][i],
            matrix.cols_num,
            matrix.rows_num
        )


class Determinant:
    def __init__(self):
        pass

if __name__ == "__main__":
    A = Matrix([
        [1, 4, 5],
        [7, 2, 6],
        [8, 9, 3]
    ])

    

    print(A)
    

        
