.. _polys-domainmatrix:

===============================================
Introducing the domainmatrix of the poly module
===============================================

This page introduces the idea behind domainmatrix which is used in SymPy's
:mod:`sympy.polys` module. This is a relatively advanced topic so for a better understanding 
it is recommended to read about :py:class:`~.Domain` and :py:class:`~.ddm`along with
:mod:`sympy.matrices` module.

What is domainmatrix?
=====================

To say in a one-liner, it is associating Matrix with :py:class:`~.Domain`.

Briefly we can say, a domainmatrix represents a matrix with elements that are in a particular
Domain. Each domainmatrix internally wraps a DDM which is used for thelower-level operations. 
The idea is that the domainmatrix class provides the convenience routines for converting 
between Expr and the poly domains as well as unifying matrices with different domains.

In general, we represent a matrix without concerning about the :py:class:`~.Domain` as:
    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> Matrix1
    

