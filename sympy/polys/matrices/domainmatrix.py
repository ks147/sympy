"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
from sympy.core.sympify import _sympify

from ..constructor import construct_domain

from .exceptions import NonSquareMatrixError, ShapeError

from .ddm import DDM


class DomainMatrix:

    def __init__(self, rows, shape, domain):
        self.rep = DDM(rows, shape, domain)
        self.shape = shape
        self.domain = domain

    @classmethod
    def from_ddm(cls, ddm):
        return cls(ddm, ddm.shape, ddm.domain)

    @classmethod
    def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
        """Converts a list to Domain Matrix

        Examples
        ========
        Define a Domain Matrix on the field of Integers

        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_list_sympy(1 , 2, [[1, 0]])
        >>> A
        DomainMatrix([[1, 0]], (1, 2), ZZ)

        We now define a Domain Matrix on the field of Rational Numbers

        >>> from sympy import Rational
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_list_sympy(2, 2, [[Rational(1,2), Rational(3,4)],[2, Rational(1,5)]])
        >>> A
        DomainMatrix([[1/2, 3/4], [2, 1/5]], (2, 2), QQ)

        """
        assert len(rows) == nrows
        assert all(len(row) == ncols for row in rows)

        items_sympy = [_sympify(item) for row in rows for item in row]

        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        domain_rows = [[items_domain[ncols*r + c] for c in range(ncols)] for r in range(nrows)]

        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, **kwargs):
        """Converts a Matrix to Domain Matrix of
        suitable Domain

        Examples
        ========
        Define a Domain Matrix on the field of Rationals

        >>> from sympy import Matrix, Rational
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_Matrix(Matrix([
        ...     [Rational(1,2), Rational(3,4)],
        ...     [2, Rational(1,5)],
        ... ]))
        >>> A
        DomainMatrix([[1/2, 3/4], [2, 1/5]], (2, 2), QQ)

        Define a Domain matrix on the field of Reals.
        >>> from sympy import Matrix
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_Matrix(Matrix([
        ...     [1.0, 3.4],
        ...     [2.4, 1]
        ... ]))
        >>> A
        DomainMatrix([[1.0, 3.4], [2.4, 1.0]], (2, 2), RR)

        """
        return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):

        K, items_K = construct_domain(items_sympy, **kwargs)
        return K, items_K

    def convert_to(self, K):
        """ Converts an object of a given Domain to another specified
        domain K

        Parameters
        ==========

        K: Matrix is to be converted to Domain K

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ, RR
        >>> A = DomainMatrix([[QQ(1,2), 3]], (1, 2), QQ)
        >>> A.convert_to(RR)
        DomainMatrix([[0.5, 3.0]], (1, 2), RR)

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ, RR
        >>> A = DomainMatrix([[RR(1.5), RR(1)], [RR(1.2), RR(0.3)]], (2, 2), RR)
        >>> A.convert_to(QQ)
        DomainMatrix([[3/2, 1], [6/5, 3/10]], (2, 2), QQ)

        """
        Kold = self.domain
        if K == Kold:
            return self.from_ddm(self.rep.copy())
        new_rows = [[K.convert_from(e, Kold) for e in row] for row in self.rep]
        return DomainMatrix(new_rows, self.shape, K)

    def to_field(self):
        K = self.domain.get_field()
        return self.convert_to(K)

    def unify(self, other):
        """Unifies the domain of 2 Matrices, so that
        they both belong to a common unified Domain

        Parameters
        ==========

        other: other Domain Matrix to be unified

        Returns
        =======

        Two Matrices that contain the elements of the input
        matrices but they now belong to a unified Domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ, QQ, RR
        >>> A = DomainMatrix([[1, 2, 3]], (1, 3), ZZ)
        >>> B = DomainMatrix([[RR(1.2), RR(2.0), RR(3.4)]], (1, 3), RR)
        >>> A.unify(B)
        (DomainMatrix([[1.0, 2.0, 3.0]], (1, 3), RR), DomainMatrix([[1.2, 2.0, 3.4]], (1, 3), RR))

        """
        K1 = self.domain
        K2 = other.domain
        if K1 == K2:
            return self, other
        K = K1.unify(K2)
        if K1 != K:
            self = self.convert_to(K)
        if K2 != K:
            other = other.convert_to(K)
        return self, other

    def to_Matrix(self):
        """ Converts an object of Domain Matrix class to
        sympy's Matrix Class

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[1, 2], [2, 3]], (2, 2), ZZ)
        >>> A.to_Matrix()
        Matrix([
        [1, 2],
        [2, 3]])

        """
        from sympy.matrices.dense import MutableDenseMatrix
        rows_sympy = [[self.domain.to_sympy(e) for e in row] for row in self.rep]
        return MutableDenseMatrix(rows_sympy)

    def __repr__(self):
        rows_str = ['[%s]' % (', '.join(map(str, row))) for row in self.rep]
        rowstr = '[%s]' % ', '.join(rows_str)
        return 'DomainMatrix(%s, %r, %r)' % (rowstr, self.shape, self.domain)

    def __add__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, DomainMatrix):
            return A.matmul(B)
        elif B in A.domain:
            return A.from_ddm(A.rep * B)
        else:
            return NotImplemented

    def __rmul__(A, B):
        if B in A.domain:
            return A.from_ddm(A.rep * B)
        else:
            return NotImplemented

    def __pow__(A, n):
        """A ** n"""
        if not isinstance(n, int):
            return NotImplemented
        return A.pow(n)

    def add(A, B):
        if A.shape != B.shape:
            raise ShapeError("shape")
        if A.domain != B.domain:
            raise ValueError("domain")
        return A.from_ddm(A.rep.add(B.rep))

    def sub(A, B):
        if A.shape != B.shape:
            raise ShapeError("shape")
        if A.domain != B.domain:
            raise ValueError("domain")
        return A.from_ddm(A.rep.sub(B.rep))

    def neg(A):
        return A.from_ddm(A.rep.neg())

    def mul(A, b):
        return A.from_ddm(A.rep.mul(b))

    def matmul(A, B):
        return A.from_ddm(A.rep.matmul(B.rep))

    def pow(A, n):
        """Return Matrix A raised to the power n
        A ** n

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> A = DomainMatrix([[1, 3], [2, 1]], (2, 2), ZZ)
        >>> pow(A, 3)
        DomainMatrix([[19, 27], [18, 19]], (2, 2), ZZ)

        """
        if n < 0:
            raise NotImplementedError('Negative powers')
        elif n == 0:
            m, n = A.shape
            rows = [[A.domain.zero] * m for _ in range(m)]
            for i in range(m):
                rows[i][i] = A.domain.one
            return type(A)(rows, A.shape, A.domain)
        elif n == 1:
            return A
        elif n % 2 == 1:
            return A * A**(n - 1)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn

    def rref(self):
        """Return reduced row-echelon form of a Domain matrix and indices of pivot vars.
        Ensure that the Domain of the input Matrix is a field.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [1, QQ(1,2), 3],
        ...     [2, QQ(3,4), 1],
        ...     [QQ(-1,2), 2,0]], (3, 3), QQ)
        >>> rref_matrix, rref_pivots = A.rref()
        >>> rref_matrix
        DomainMatrix([[1.0, 0.0, 0.0], [0, 1.0, 0.0], [0, 0.0, 1.0]], (3, 3), QQ)
        >>> rref_pivots
        (0, 1, 2)

        """
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        rref_ddm, pivots = self.rep.rref()
        return self.from_ddm(rref_ddm), tuple(pivots)

    def nullspace(self):
        """Returns the Nullspace of a Domain Matrix.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> A = DomainMatrix([
        ... [1, -1],
        ... [2, -2]], (2, 2), ZZ)
        >>> A.nullspace()
        DomainMatrix([[1.0, 1]], (1, 2), ZZ)

        """
        return self.from_ddm(self.rep.nullspace())

    def inv(self):
        """Returns the inverse of a Domain Matrix if the Domain is a field.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ... [2, -1, 0],
        ... [-1, 2, -1],
        ... [0, 0, 2]], (3, 3), QQ)
        >>> A.inv()
        DomainMatrix([[0.66666666666666663, 0.33333333333333331, 0.16666666666666666], [0.33333333333333331, 0.66666666666666663, 0.33333333333333331], [0.0, 0.0, 0.5]], (3, 3), QQ)

        """
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        m, n = self.shape
        if m != n:
            raise NonSquareMatrixError
        inv = self.rep.inv()
        return self.from_ddm(inv)

    def det(self):
        """Returns the determinant of a Domain Matrix.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import ZZ
        >>> A = DomainMatrix([
        ... [2, -1, 0],
        ... [-1, 2, -1],
        ... [0, 0, 2]], (3, 3), ZZ)
        >>> A.det()
        mpz(6)

        """
        m, n = self.shape
        if m != n:
            raise NonSquareMatrixError
        return self.rep.det()

    def lu(self):
        """Returns (L, U, swaps) where L is a lower triangular matrix with unit diagonal, U is an upper triangular matrix,
        and swaps is a list of row swap index pairs.
        If A is the original matrix, then A = (L*U).permute(swaps),
        and the row permutation matrix P such that P*A = L*U can be computed by P=eye(A.row).permute(swaps).

        The domain of the matrix A must be a field.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ... [2, -1, 0],
        ... [-1, 2, -1],
        ... [0, 0, 2]], (3, 3), QQ)
        >>> L, U, swaps = A.lu()
        >>> L
        DomainMatrix([[1, 0, 0], [-0.5, 1, 0], [0.0, 0.0, 1]], (3, 3), QQ)
        >>> U
        DomainMatrix([[2, -1, 0], [0, 1.5, -1.0], [0, 0, 2.0]], (3, 3), QQ)
        >>> swaps
        []
        >>> L*U == A
        True

        """
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        L, U, swaps = self.rep.lu()
        return self.from_ddm(L), self.from_ddm(U), swaps

    def lu_solve(self, rhs):
        """Solve the linear system Ax = rhs for x using LU decomposition
        of the Domain Matrix A.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [2, -1, 0],
        ...     [-1, 2, -1],
        ...     [0, 0, 2]], (3, 3), QQ)
        >>> b = DomainMatrix([[1],[4],[0]], (3, 1), QQ)
        >>> x = A.lu_solve(b)
        >>> x
        DomainMatrix([[2.0], [3.0], [0.0]], (3, 1), QQ)

        """
        if self.shape[0] != rhs.shape[0]:
            raise ShapeError("Shape")
        if not self.domain.is_Field:
            raise ValueError('Not a field')
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_ddm(sol)

    def charpoly(self):
        """Computes characteristic polynomial det(x*I - M) where I is the identity matrix and
         M is the input Domain Matrix
        A PurePoly is returned, so using different variables for x does not affect the comparison or the polynomials:

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([
        ...     [2, -1, 0],
        ...     [-1, 2, -1],
        ...     [0, 0, 2]], (3, 3), QQ)
        >>> A.charpoly()
        [mpq(1,1), mpq(-6,1), mpq(11,1), mpq(-6,1)]

        """
        m, n = self.shape
        if m != n:
            raise NonSquareMatrixError("not square")
        return self.rep.charpoly()

    @classmethod
    def eye(cls, n, domain):
        """Return Identity matrix of size n x n, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> DomainMatrix.eye(3, QQ)
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)

        """
        return cls.from_ddm(DDM.eye(n, domain))

    def __eq__(A, B):
        """A == B"""
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        return A.rep == B.rep
