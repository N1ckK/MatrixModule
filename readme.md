# Documentation for the matrix module

The matrix module implements calculating with matrices and vectors in python.
Matrices are stored in two dimensional lists.

## Contents
- **Classes**
    - `matrix.Matrix`
    - `matrix.Vector(Matrix)`
- **Private Functions**
    - `matrix._copy`
    - `matrix._flatten`
    - `matrix._cut`
    - `matrix._all`
    - `matrix._extend`
- **Public Functions**
    - `matrix.normalized`
    - `matrix.isLinearIndependent`
    - `matrix.zero_matrix`
    - `matrix.identity_matrix`
    - `matrix.add`
    - `matrix.multiply`
    - `matrix.power`
    - `matrix.scale`
    - `matrix.trace`
    - `matrix.determinant`
    - `matrix.minor`
    - `matrix.transpose`
    - `matrix.triangular`
    - `matrix.gaussian_elimination`
    - `matrix.cramers_rule`
    - `matrix.inverse`
    - `matrix.rank`
    - `matrix.eigenvalues`
    - `matrix.diagonalize`


------------------------------------------------------------------------------

## <center>Classes</center>

#### matrix.Matrix

~~~~
def __init__(self, entries: list)
~~~~
Initialization method for creating a *(m × n)* Matrix. \_\_init\_\_ expects a two
dimensional list. The given list should contain *m* lists each containing *n*
elements. Elements allowed are: *int*, *float*.

Example:
~~~~
__init__([  [a_{11}, a_{12}, a_{13}],
            [a_{21}, a_{22}, a_{23}],
            [a_{31}, a_{32}, a_{33}]
         ])
~~~~

will yield

$$\left(
    \begin{array}{ccc}
        a_{11} & a_{12} & a_{13} \\
        a_{21} & a_{22} & a_{23} \\
        a_{31} & a_{32} & a_{33}
    \end{array}
\right)$$

------------------------------------------------------------------------------

~~~~
def row(self, j: int) -> list
~~~~
When called on a matrix, the j-th row of the matrix will be returned as a list.
If *j* is out of bounds and *IndexError* will be raised.

------------------------------------------------------------------------------

~~~~
def col(self, i: int)
~~~~
When called on a matrix, the *i*-th column of the matrix will be returned as a
list.
If *i* is out of bounds and *IndexError* will be raised.


------------------------------------------------------------------------------

~~~~
def set_row(self, r: list, j: int)
~~~~
When called on a matrix, the *j*-th row will be set to the elements in given list *r*. The given list has to match the dimension of the Matrix. If this is not the case an *AssertionError* will be raised.

------------------------------------------------------------------------------

~~~~
def set_col(self, c: list, i: int)
~~~~
When called on a matrix, the *i*-th column will be set to the elements in given list *c*. he given list has to match the dimension of the Matrix. If this is not the case an *AssertionError* will be raised.


------------------------------------------------------------------------------

~~~~
def empty(self, n: int, m: int)
~~~~
When called on a matrix, the matrix will be set to the Zero Matrix *(m × n)*. The matrix will be set to a new object upon call.

Example:
~~~~
empty(4, 3)
~~~~

will yield

$$\left(
    \begin{array}{ccc}
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0 \\
        0 & 0 & 0
    \end{array}
\right)$$

------------------------------------------------------------------------------

~~~~
@property
def entries(self)
~~~~
When called on a matrix, the two dimensional list (as described above) of all elements of the matrix will be returned.

------------------------------------------------------------------------------

~~~~
@property
def m(self)
~~~~
When called on a matrix, the number of rows will be returned.

------------------------------------------------------------------------------

~~~~
@property
def n(self)
~~~~
When called on a matrix, the number of columns will be returned.

------------------------------------------------------------------------------

~~~~
def __getitem__(self, key)
~~~~
When called on a matrix, the elements specified in *key* will be returned.
Valid Keys:
Let $i, j \in \mathbb{N}_0$ and $A = (a_{ij})_ {n,m}$
- $\text{matrix}[i,j]$ will yield $a_{j+1,i+1}$ of $A$.
- $\text{matrix}[-1, i]$ will yield $[a_{i+1, 1}, ..., a_{i+1, n}]$ of $A$.
- $\text{matrix}[i]$ will yield $a_{1, i+1}, ..., a_{m, i+1}$ of $A$.

Example:
~~~~
matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix[0, 0] == 1
matrix[2, 1] == 8
matrix[-1, 1] == [4, 5, 6]
matrix[2] == [3, 6, 9]
~~~~

------------------------------------------------------------------------------

~~~~
def __setitem__(self, key, value)
~~~~
When called on a matrix, the elements specified in *key* will be set to value.
Valid keys:
Let $i, j \in \mathbb{N}_0$ and $A = (a_{ij})_ {n,m}$
- $\text{matrix}[i,j]$ will set $a_{j+1,i+1}$ to the value of *value*.
- $\text{matrix}[-1, i]$ will set $[a_{i+1, 1}, ..., a_{i+1, n}]$ to *value*.
- $\text{matrix}[i]$ will set $a_{1, i+1}, ..., a_{m, i+1}$ to *value*.


------------------------------------------------------------------------------

~~~~
def __eq__(self, other)
~~~~
Returns true if the matrix of self equals the matrix of other. Equality is
there by defined as $\forall i,j: a_{ij} = b_{ij}$ with
$A = (a_{ij})_ {n,m}$ and $B = (a_ij)_ {n,m}$

------------------------------------------------------------------------------

~~~~
def __add__(self, other)
~~~~


------------------------------------------------------------------------------

~~~~
def __sub__(self, other)
~~~~

------------------------------------------------------------------------------

~~~~
def __mul__(self, other)
~~~~

------------------------------------------------------------------------------

~~~~
def __rmul__(self, other)
~~~~

------------------------------------------------------------------------------

~~~~
def __abs__(self)
~~~~
Returns the determinant of given determinant. An assertion error is raised if
the matric is not quadratic.

------------------------------------------------------------------------------

~~~~
def __bool__(self)
~~~~
Returns true of the matrix is the Zero Matrix else false.

------------------------------------------------------------------------------

~~~~
def __iter__(self):
~~~~

------------------------------------------------------------------------------

~~~~
def __next__(self):
~~~~

------------------------------------------------------------------------------

~~~~
def __str__(self)
~~~~
Prints a string representation of the matrix to the console. Numbers are
rounded to two decimal places.

------------------------------------------------------------------------------

#### matrix.Vector(Matrix)

~~~~
def __init__(self, *args)
~~~~
Initialization method for creating a *(m × 1)* Vector. \_\_init\_\_ expects a  
list. The given list should contain *m* elements allowed are: *int*, *float*.

Example:
~~~~
__init__([a_{1},a_{2},a_{3}])
~~~~

will yield

$$
\begin{bmatrix}
    a_{1} \\
    a_{2} \\
    a_{3}
\end{bmatrix}
$$

------------------------------------------------------------------------------

~~~~
def empty(self, n)
~~~~

------------------------------------------------------------------------------

~~~~
def __len__(self)
~~~~
Returns the dimension of the vector.

Example:
$$
\begin{bmatrix}
    a_{1} \\
    a_{2} \\
    ... \\
    a_{n}
\end{bmatrix}
$$
will yield $n$.

------------------------------------------------------------------------------

~~~~
def __abs__(self)
~~~~
Returns returns the length of the Vector.

------------------------------------------------------------------------------

~~~~
def __add__(self, other)
~~~~

------------------------------------------------------------------------------


~~~~
def __sub__(self, other)
~~~~

------------------------------------------------------------------------------


~~~~
def __rmul__(self, other)
~~~~

------------------------------------------------------------------------------

~~~~
def __getitem__(self, key)
~~~~

------------------------------------------------------------------------------

~~~~
def __setitem__(self, key, value)
~~~~

------------------------------------------------------------------------------

## <center>Private Functions</center>

#### matrix.\_copy

~~~~
def _copy(matrix)
~~~~
Returns a copy of a given matrix or vector.

------------------------------------------------------------------------------

#### matrix.\_flatten

~~~~
def _flatten(lst: list) -> list
~~~~
Flattens the given list *li*.

------------------------------------------------------------------------------

#### matrix.\_cut

~~~~
def _cut(list: list, index: int) -> list
~~~~
Removes index from list and returns the list without the index.

------------------------------------------------------------------------------

#### matrix.\_all

~~~~
def _all(matrix, right=None)
~~~~

Is an iterator which yields all pairs $(i, j)$ which have Elements in $\text{matrix}[i, j]$ associated to them. If *right* is specified the iterator will returns all pairs $(i,j)$ that occur in the product of *matrix* and *right*.

------------------------------------------------------------------------------

#### matrix.\_extend

~~~~
def _extend(lst: list, length: int, fill=0)
~~~~
Extends the given list to desired length by filling the list with
specified element fill. If length is less or equal to length of the
list, the list its self will be returned.

------------------------------------------------------------------------------


## <center>Public Functions</center>

#### matrix.normalized

~~~~
def normalized(vector)
~~~~
Normalizes the given vector.

------------------------------------------------------------------------------

#### matrix.isLinearIndependent

~~~~
def isLinearIndependent(*vectors) -> bool
~~~~
Takes in vectors and determines wheather they are linear linear independent.
If vectors are of different dimension, all vectors will be converted to same
dimension by appending zeros.

------------------------------------------------------------------------------

#### matrix.zero_matrix

~~~~
def zero_matrix(m: int, n: int)
~~~~
Returns the m x n Zero Matrix

------------------------------------------------------------------------------

#### matrix.identity_matrix

~~~~
def identity_matrix(n: int)
~~~~
Returns the n x n Identity Matrix

------------------------------------------------------------------------------

#### matrix.add

~~~~
def add(matrix, matrix2)
~~~~
Adds matrix and matrix2 and returns the sum.
If the dimensions of *matrix* and *matrix2* are invalid, an *AssertionError* will be raised. The sum will be returned as a new matrix, the given matrices stay unaltered.

------------------------------------------------------------------------------

#### matrix.multiply

~~~~
def multiply(matrix, matrix2)
~~~~
Multiplies matrix (m × p) and matrix2 (p × n) and returns the product.
Matrix dimensions are checked beforehand; an assertion error will be raised
if dimensions are invalid. matrix2 can also be a (p × 1) vector. The product will be returned as a new matrix, the given matrices stay unaltered.

------------------------------------------------------------------------------

#### matrix.power

~~~~
def power(matrix, p: int)
~~~~
Raises matrix the the p-th power and returns it. If the matrix is not quadratic, an *AssertionError* will be raised.

------------------------------------------------------------------------------

#### matrix.scale

~~~~
def scale(matrix, s: float)
~~~~
Scales matrix by s. The result will be returned as a new matrix, the given matrices stay unaltered.

------------------------------------------------------------------------------

#### matrix.trace

~~~~
def trace(matrix) -> float
~~~~
Returns the trace of the given matrix. The matrix has to be quadratic, otherwise an *AssertionError* will be raised.

------------------------------------------------------------------------------

#### matrix.determinant

~~~~
def determinant(matrix) -> float
~~~~
Returns the determinant of the given matrix.
The matrix has to be quadratic otherwise an *AssertionError* will be raised.

------------------------------------------------------------------------------

#### matrix.minor

~~~~
def minor(matrix, i: int, j: int)
~~~~
Returns the minor matrix of matrix with column i and row j left out.

------------------------------------------------------------------------------

#### matrix.transpose

~~~~
def transpose(matrix)
~~~~
Returns transposed matrix.

------------------------------------------------------------------------------

#### matrix.triangular

~~~~
def triangular(matrix)
~~~~
Brings the matrix into triangular form and returns it

------------------------------------------------------------------------------

#### matrix.gaussian_elimination

~~~~
def gaussian_elimination(A, b)
~~~~
When applied, the function will return $Ax=b$ in triangular form.

------------------------------------------------------------------------------

#### matrix.cramers_rule

~~~~
def cramers_rule(A, b)
~~~~
When applied, the function will return the Vector that satisfies $Ax=b$. The
Vector $x$ has to be unambiguous. This is the case if the determinant of $A$
is not $0$, otherwise an *AsssertionError* will be raised.

------------------------------------------------------------------------------

#### matrix.inverse

~~~~
def inverse(matrix)
~~~~

Returns the inverse of *matrix*.

------------------------------------------------------------------------------

#### matrix.rank

~~~~
def rank(matrix) -> int
~~~~

Returns the rank of the given matrix

------------------------------------------------------------------------------

#### matrix.eigenvalues (in work)

~~~~
def eigenvalues(matrix)
~~~~

Returns all eigenvalues of the matrix.

------------------------------------------------------------------------------

#### matrix.diagonalize (in work)

~~~~
def diagonalize(matrix)
~~~~
