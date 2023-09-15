import math
import sympy as sym

def  determinant3x3 (matrix: list) -> float or 'sym.Add':
    
    principal = matrix[0][0]*matrix[1][1]*matrix[2][2] + \
                matrix[0][1]*matrix[1][2]*matrix[2][0] + \
                matrix[0][2]*matrix[1][0]*matrix[2][1]
    
    secondary = matrix[0][2]*matrix[1][1]*matrix[2][0] + \
                matrix[1][2]*matrix[2][1]*matrix[0][0] + \
                matrix[2][2]*matrix[0][1]*matrix[1][0]
    
    return principal - secondary



i, j, k = sym.symbols('i j k')

class Vector (list):


    def __init__ (self, x: float = 0, y: float = 0, z: float = None, *args):
        self.x = x
        self.append(x)
        
        self.y = y
        self.append(y)

        self.z = 0
        if not isinstance(z, None.__class__):
            self.append(z)

        if args:
            for index in range(len(args)):
                self.append(args[index])
                self.__dict__[f'z{index}'] = args[index]

    @property
    def dimension (self):
        return len(self.entrys)
    
    @property
    def entrys (self):
        entrys = [entry for entry in self]
        return entrys

    def __repr__(self) -> str:
        string = ' '.join([f"{key}={entry}" for key, entry in self.__dict__.items()])
        return f'Vector({string})'
    
    def __str__(self) -> str:
        string = ' '.join([f"{entry}," for entry in self.entrys])
        return f'({string[:len(string) - 1]})'
    
    @property
    def module (self) -> float:
        """
        Vector's module.
        """

        if isinstance(self.z, None.__class__):
            return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))
       
        return math.sqrt(sum([math.pow(entry, 2) for entry in self.entrys]))
    
    
    @property
    def direction (self) -> 'Vector':
        """
        Unitary version of the vector.
        """
        if isinstance(self.z, None.__class__):
            return Vector(self.x/self.module, self.y/self.module)

        return Vector(*[entry/self.module for entry in self.entrys])
    
    @property
    def with_unitary_vectors (self) -> 'sym.Add':
        """
        Representation of the vectors with i, j and k.
        """
        z = self.z

        if isinstance(self.z, None.__class__):
            z = 0

        res = 0
        other_axes = [self.entrys[i]*sym.symbols(f'z{i-3}') for i in range(3, self.dimension)]
        for axe in other_axes:
            res += axe

        return self.x*i + self.y*j + z*k + res
    
    @property
    def column_matrix (self) -> list:
        """
        Column matrix vector representation.
        """
        return [
            [entry] for entry in self.entrys
        ]
    
    @property
    def row_matrix (self) -> list:
        """
        Row matrix vector representation.
        """
        return [
            [*self.entrys]
        ]

    def scalar (self, other_vector: 'Vector') -> float:
        """
        Scalar product between the vectors.
        """

        if other_vector.dimension != self.dimension:
            raise ArithmeticError("Can not make scalar product between vectors with different dimensions")

        if isinstance(self.z, None.__class__):
            return self.x*other_vector.x + self.y*other_vector.y
        
        return sum([self.entrys[i]*other_vector.entrys[i] for i in range(self.dimension)])
    
    def vectorial (self, other_vector: 'Vector') -> 'Vector':
        """
        Vectorial product between the vectors.
        """
        z1 = self.z
        z2 = other_vector.z
        if isinstance(self.z, None.__class__) and other_vector.dimension == self.dimension:
            z1 = 0
            z2 = 0
        
        if self.dimension > 3:
            raise ArithmeticError("Can not make vetorial product in dimension greater than 3")

        res = determinant3x3([
            [i, j, k],
            [self.x, self.y, z1],
            [other_vector.x, other_vector.y, z2],
        ])

        x, y, z = res.coeff(i), res.coeff(j), res.coeff(k)
        return Vector(x, y, z)

    def angule_between (self, other_vector: 'Vector') -> float:
        """
        Cos(0) = (u.v)/(||u||.||v||)
        """
        teta = math.acos((self * other_vector) / (self.module * other_vector.module))
        return round(math.degrees(teta), 2)

    def __mul__ (self, other_vector: 'Vector') -> float or 'Vector':
        if isinstance(other_vector, Vector):
            return self.scalar(other_vector)
        
        return Vector(*[round(entry*other_vector, 6) for entry in self.entrys])
    
    def __lt__(self, other_vector: 'Vector') -> bool:
        """
        Angle between the vectors.
        """
        return self.angule_between(other_vector)

    def __gt__(self, other_vector: 'Vector') -> bool:
        """
        Complementary angle between the vectors
        """
        return 180 - self.angule_between(other_vector)
    
    def __add__ (self, other_vector: 'Vector') -> 'Vector':
        if self.dimension != other_vector.dimension:
            raise ArithmeticError('Can not sum different dimensions vectors!')
        
        return Vector(*[round(self.entrys[i] + other_vector.entrys[i], 6) for i in range(len(self.entrys))])
    
    def __sub__ (self, other_vector: 'Vector') -> 'Vector':
        if self.dimension != other_vector.dimension:
            raise ArithmeticError('Can not sum different dimensions vectors!')
        
        return Vector(*[round(self.entrys[i] - other_vector.entrys[i], 6) for i in range(len(self.entrys))])
    
    
    def __round__ (self, decimals: int):
        return Vector(*[round(entry, decimals) for entry in self.entrys])
    
    def reverse (self) -> None:
        from copy import deepcopy
        reversed_entrys = deepcopy(self.entrys)
        reversed_entrys.reverse()
        self.__init__(*reversed_entrys)

    def invert (self) -> None:
        self.__init__ (*(self*-1).entrys)
    
    
    
    
if __name__ == '__main__':
    
    v = Vector(*[1, 2])
    print(v.with_unitary_vectors)