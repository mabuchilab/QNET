#!/usr/bin/env python
# encoding=utf-8
r"""
The Gough-James Circuit Algebra
===============================

In their works on networks of open quantum systems [2]_, [3]_ Gough and James have introduced an algebraic method to derive the Quantum Markov model for a full network of cascaded quantum systems from the reduced Markov models of its constituents.
A general system with an equal number :math:`n` of input and output channels is described by the parameter triplet :math:`\left(\mathbf{S}, \mathbf{L}, H\right)`, where :math:`H` is the effective internal *Hamilton operator* for the system, :math:`\mathbf{L} = (L_1, L_2, \dots, L_n)^T` the *coupling vector* and :math:`\mathbf{S} = (S_{jk})_{j,k=1}^n` is the *scattering matrix* (whose elements are themselves operators).
An element :math:`L_k` of the coupling vector is given by a system operator that describes the system's coupling to the :math:`k`-th input channel. Similarly, the elements :math:`S_{jk}` of the scattering matrix are in general given by system operators describing the scattering between different field channels :math:`j` and :math:`k`.
The only conditions on the parameters are that the hamilton operator is self-adjoint and the scattering matrix is unitary:

.. math::

    H^* = H \text{ and } \mathbf{S}^\dagger \mathbf{S} = \mathbf{S} \mathbf{S}^\dagger = \mathbbm{1}_n.
    

We adhere to the conventions used by Gough and James, i.e. we write the imaginary unit is given by :math:`i := \sqrt{-1}`, the adjoint of an operator :math:`A` is given by :math:`A^*`, the element-wise adjoint of an operator matrix :math:`\mathbf{M}` is given by :math:`\mathbf{M}^\sharp`. Its transpose is given by :math:`\mathbf{M}^T` and the combination of these two operations, i.e. the adjoint operator matrix is given by :math:`\mathbf{M}^\dagger = (\mathbf{M}^T)^\sharp = (\mathbf{M}^\sharp)^T`.

Fundamental Circuit Operations
------------------------------

(figures)

In [3]_, Gough and James have introduced two operations that allow the construction of quantum optical 'feedforward' networks:

    1) The *concatenation* product describes the situation where two arbitrary systems are formally attached to each other without optical scattering between the two systems' in- and output channels

    .. math::
	
        \left(\mathbf{S}_1, \mathbf{L}_1, H_1\right) \boxplus \left(\mathbf{S}_2, \mathbf{L}_2, H_2\right) = \left(\begin{pmatrix} \mathbf{S}_1 & 0 \\ 0 & \mathbf{S}_2 \end{pmatrix}, \begin{pmatrix}\mathbf{L}_1 \\ \mathbf{L}_1 \end{pmatrix}, H_1 + H_2 \right)

    Note however, that even without optical scattering, the two subsystems may interact directly via shared quantum degrees of freedom.

    2) The *series* product is to be used for two systems :math:`Q_j = \left(\mathbf{S}_j, \mathbf{L}_j, H_j \right)`, :math:`j=1,2` of equal channel number :math:`n` where all output channels of :math:`Q_1` are fed into the corresponding input channels of :math:`Q_2`

    .. math::

        \left(\mathbf{S}_2, \mathbf{L}_2, H_2 \right) \lhd \left( \mathbf{S}_1, \mathbf{L}_1, H_1 \right) = \left(\mathbf{S}_2 \mathbf{S}_1,\mathbf{L}_2 + \mathbf{S}_2\mathbf{L}_1 , H_1 + H_2 + \Im\left\{\mathbf{L}_2^\dagger \mathbf{S}_2 \mathbf{L}_1\right\}\right)

From their definition it can be seen that the results of applying both the series product and the concatenation product not only yield valid circuit component triplets that obey the constraints, but they are also associative operations.\footnote{For the concatenation product this is immediately clear, for the series product in can be quickly verified by computing :math:`(Q_1 \lhd Q_2) \lhd Q_3` and :math:`Q_1 \lhd (Q_2 \lhd Q_3)`. 
To make the network operations complete in the sense that it can also be applied for situations with optical feedback, an additional rule is required: The *feedback* operation  describes the case where the :math:`k`-th output channel of a system with :math:`n\ge 2` is fed back into the :math:`l`-th input channel. The result is a component with :math:`n-1` channels:

.. math::
	
    \left[\;\left(\mathbf{S}, \mathbf{L}, H \right)\;\right]_{k \to l} = \left(\tilde{\mathbf{S}}, \tilde{\mathbf{L}}, \tilde{H}\right),

where the effective parameters are given by [2]_

.. math::
	
    \tilde{\mathbf{S}} & = \mathbf{S}_{\cancel{[k,l]}} +  \begin{pmatrix} S_{1l} \\ S_{2l} \\ \vdots \\ S_{k-1\, l} \\ S_{k+1\, l} \\ \vdots \\ S_{n l}\end{pmatrix}(1 - S_{kl})^{-1}  \begin{pmatrix} S_{k 1} & S_{k2} & \cdots & S_{kl-1} & S_{kl+1} & \cdots & S_{k n}\end{pmatrix}, \\
    \tilde{\mathbf{L}} & = \mathbf{L}_{\cancel{[k]}} + \begin{pmatrix} S_{1l} \\ S_{2l} \\ \vdots \\ S_{k-1\, l} \\ S_{k+1\, l} \\ \vdots \\ S_{n l}\end{pmatrix} (1 - S_{kl})^{-1} L_k, \\
    \tilde{H} & = H + \Im\left\{\ \left[\sum_{j=1}^n L_j^* S_{jl}\right] (1 - S_{kl})^{-1} L_k \right\}.

Here we have written :math:`\mathbf{S}_{\cancel{[k,l]}}` as a shorthand notation for the matrix :math:`\mathbf{S}` with the :math:`k`-th row and :math:`l`-th column removed and similarly :math:`\mathbf{L}_{\cancel{[k]}}` is the vector :math:`\mathbf{L}` with its :math:`k`-th entry removed.
Moreover, it can be shown that in the case of multiple feedback loops, the result is independent of the order in which the feedback operation is applied. Note however that some care has to be taken with the indices of the feedback channels when permuting the feedback operation.

The possibility of treating the quantum circuits algebraically offers some valuable insights:
A given full-system triplet :math:`(\mathbf{S}, \mathbf{L}, H )` may very well allow for different ways of decomposing it algebraically into networks of physically realistic subsystems. The algebraic treatment thus establishes a notion of dynamic equivalence between potentially very different physical setups.
Given a certain number of fundamental building blocks such as beamsplitters, phases and cavities, from which we construct complex networks, we can investigate what kinds of composite systems can be realized. If we also take into account the adiabatic limit theorems for QSDEs (cite Bouten2008a,Bouten2008) the set of physically realizable systems is further expanded.
Hence, the algebraic methods not only facilitate the analysis of quantum circuits, but ultimately they may very well lead to an understanding of how to construct a general system :math:`(\mathbf{S}, \mathbf{L}, H)` from some set of elementary systems.
There already exist some investigations along these lines for the particular subclass of *linear* systems (cite Nurdin2009a,Nurdin2009b) which can be thought of as a networked collection of quantum harmonic oscillators.

Representation as Python objects
--------------------------------

This file features an implementation of the Gough-James circuit algebra rules as introduced in [2]_ and [3]_.
Python objects that are of the :py:class:`Circuit` type have some of their operators overloaded to realize symbolic circuit algebra operations:

    >>> A = CircuitSymbol('A', 2)
    >>> B = CircuitSymbol('B', 2)
    >>> A << B
        SeriesProduct(A, B)
    >>> A + B
        Concatenation(A, B)
    >>> FB(A, 0, 1)
        Feedback(A, 0, 1)

Basic algebraic properties
--------------------------

By observing that we can define for a general system :math:`Q = (\mathbf{S}, \mathbf{L}, H)` its *series inverse* system :math:`Q^{\lhd -1} := (\mathbf{S}^\dagger, - \mathbf{S}^\dagger \mathbf{L}, - H)`

.. math::
    
    (\mathbf{S}, \mathbf{L}, H) \lhd (\mathbf{S}^\dagger, - \mathbf{S}^\dagger \mathbf{L}, - H) =   (\mathbf{S}^\dagger, - \mathbf{S}^\dagger \mathbf{L}, - H) \lhd (\mathbf{S}, \mathbf{L}, H) = (\mathbbm{1}_n, 0, 0) =: \id{n},

we see that the series product induces a group structure on the set of :math:`n`-channel circuit components for any :math:`n \ge 1`.
It can easily be verified that the series inverse of the basic operations is calculated as follows

.. math::
    
    \left(Q_1 \lhd Q_2\right)^{\lhd -1} & = Q_2^{\lhd -1} \lhd Q_1^{\lhd -1} \\
    \left(Q_1 \boxplus Q_2\right)^{\lhd -1} & = Q_1^{\lhd -1} \boxplus Q_2^{\lhd -1} \\
    \left([Q]_{k\to l}\right)^{\lhd -1} & = \left[Q^{\lhd -1}\right]_{l\to k}.

In the following, we denote the number of channels of any given system :math:`Q = (\mathbf{S}, \mathbf{L}, H)` by :math:`\cdim{Q} := n`.
The most obvious expression simplification is the associative expansion of concatenations and series:

.. math::
    
    (A_1 \lhd A_2) \lhd (B_1 \lhd B_2) & = A_1 \lhd A_2 \lhd B_1 \lhd B_2 \\
    (C_1 \boxplus C_2) \boxplus (D_1 \boxplus D_2) & = C_1 \boxplus C_2 \boxplus D_1 \boxplus D_2

A further interesting property that follows intuitively from the graphical representation (cf.~Fig.~\ref{fig:decomposition_law}) is the following tensor decomposition law

.. math::

    (A \boxplus B) \lhd (C \boxplus D) = (A \lhd C) \boxplus (B \lhd D),

which is valid for :math:`\cdim{A} = \cdim{C}` and :math:`\cdim{B} = \cdim{D}`.
(figures decomposition_law, Equivalent circuits, where a red box marks a series product and a blue box marks a concatenation. The second version \ref{fig:dist2} has the advantage of making more explicit that the overall circuit consists of two channels without direct optical scattering.)

As mentioned in the caption to Figure, it will most often be preferable to use the RHS expression of Equation () as this enables us to understand the flow of optical channels more easily from the algebraic expression.
In [3]_ Gough and James denote a system that can be expressed as a concatenation as *reducible*. A system that cannot be further decomposed into concatenated subsystems is accordingly called *irreducible*.
As follows intuitively from a graphical representation any given complex system :math:`Q = (\mathbf{S}, \mathbf{L}, H)` admits a decomposition into :math:`1 \le N \le \cdim{Q}` irreducible subsystems :math:`Q = Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N`, where their channel dimensions satisfy :math:`\cdim{Q_j}\ge 1, \, j=1,2, \dots N` and :math:`\sum_{j=1}^N \cdim{Q_j} = \cdim{Q}`. While their individual parameter triplets themselves are not uniquely determined\footnote{Actually the scattering matrices :math:`\{\mathbf{S}_j\}` and the coupling vectors :math:`\{\mathbf{L}_j\}` *are* uniquely determined, but the Hamiltonian parameters :math:`\{H_j\}` must only obey the constraint :math:`\sum_{j=1}^N H_j = H`.}, the sequence of their channel dimensions :math:`(\cdim{Q_1}, \cdim{Q_2},\dots \cdim{Q_N}) =: \bls{Q}` clearly is. We denote this tuple as the block structure of :math:`Q`.
We are now able to generalize the decomposition law in the following way:
Given two systems of :math:`n` channels with the same block structure :math:`\bls{A} = \bls{B} = (n_1, ... n_N)`, there exist decompositions of :math:`A` and :math:`B` such that

.. math::
    
    A \lhd B = (A_1 \lhd B_1) \boxplus \dots \boxplus (A_N \lhd B_N)

with :math:`\cdim{A_j} = \cdim{B_j} = n_j,\, j = 1, \dots N`.
However, even in the case that the two block structures are not equal, there may still exist non-trivial compatible block decompositions that at least allow a partial application of the decomposition law.
Consider the example presented in Figure (block_structures).
(figure, block_structures, Even in the case of a series between systems with unequal block structures, there often exists a non-trivial common block decomposition that simplifies the overall expression.)

Permutation objects
-------------------

The algebraic representation of complex circuits often requires systems that only permute channels without actual scattering. The group of permutation matrices is simply a subgroup of the unitary (operator) matrices. For any permutation matrix :math:`\mathbf{P}`, the system described by :math:`(\mathbf{P},\mathbf{0},0)` represents a pure permutation of the optical fields (ref fig permutation).
(fig, permutation, A graphical representation of :math:`\mathbf{P}_\sigma` where :math:`\sigma \equiv (4,1,5,2,3)` in image tuple notation.)

A permutation :math:`\sigma` of :math:`n` elements (:math:`\sigma \in \Sigma_n`) is often represented in the following form :math:`\begin{pmatrix} 1 & 2 & \dots & n \\ \sigma(1) & \sigma(2) & \dots & \sigma(n)\end{pmatrix}`, but obviously it is also sufficient to specify the tuple of images :math:`(\sigma(1), \sigma(2), \dots, \sigma(n))`.
We now define the permutation matrix via its matrix elements

.. math::
    
    (\mathbf{P}_\sigma)_{kl} = \delta_{k \sigma(l)} = \delta_{\sigma^{-1}(k) l}.

Such a matrix then maps the :math:`j`-th unit vector onto the :math:`\sigma(j)`-th unit vector or equivalently the :math:`j`-th incoming optical channel is mapped to the :math:`\sigma(j)`-th outgoing channel.
In contrast to a definition often found in mathematical literature this definition ensures that the representation matrix for a composition of permutations :math:`\sigma_2 \circ \sigma_1` results from a product of the individual representation matrices in the same order :math:`\mathbf{P}_{\sigma_2 \circ \sigma_1} = \mathbf{P}_{\sigma_2} \mathbf{P}_{ \sigma_1}`. This can be shown directly on the order of the matrix elements

.. math::
    
    (\mathbf{P}_{\sigma_2 \circ \sigma_1})_{kl} = \delta_{k (\sigma_2 \circ \sigma_1)(l)} = \sum_j \delta_{k j} \delta_{ j (\sigma_2 \circ \sigma_1)(l)} = \sum_j \delta_{k \sigma_2(j)} \delta_{ \sigma_2(j) (\sigma_2 \circ \sigma_1)(l)} \\
    = \sum_j \delta_{k \sigma_2(j)} \delta_{ \sigma_2(j) \sigma_2(\sigma_1(l))} = \sum_j \delta_{k \sigma_2(j)} \delta_{j  \sigma_1(l)} = \sum_j (\mathbf{P}_{\sigma_2})_{kj} (\mathbf{P}_{\sigma_1})_{jl},

where the third equality corresponds simply to a reordering of the summands and the fifth equality follows from the bijectivity of :math:`\sigma_2`.
In the following we will often write :math:`P_{\sigma}` as a shorthand for :math:`(\mathbf{P}_{\sigma}, \mathbf{0},0)`. Thus, our definition ensures that we may simplify any series of permutation systems in the most intuitive way: :math:`P_{\sigma_2} \lhd P_{\sigma_1} = P_{\sigma_2 \circ \sigma_1}`. Obviously the set of permutation systems of :math:`n` channels and the series product are a subgroup of the full system series group of :math:`n` channels. Specifically, it includes the identity :math:`\id{n} = P_{\sigma_{{\rm id}_n}}`.

From the orthogonality of the representation matrices it directly follows that :math:`\mathbf{P}_{\sigma}^T = \mathbf{P}_{\sigma^{-1}}`
For future use we also define a concatenation between permutations

.. math::
    
    \sigma_1 \boxplus \sigma_2 := \begin{pmatrix} 1 & 2 & \dots & n & n + 1 & n+2 & \dots &n + m \\ \sigma_1(1) & \sigma_1(2) & \dots & \sigma_1(n) & n + \sigma_2(1) & n + \sigma_2(2) & \dots & n + \sigma_2(m) \end{pmatrix},

which satisfies :math:`P_{\sigma_1} \boxplus P_{\sigma_2} = P_{\sigma_1 \boxplus \sigma_2}` by definition.
Another helpful definition is to introduce a special set of permutations that map specific ports into each other but leave the relative order of all other ports intact:

.. math::

    \omega_{l \gets k}^{(n)} := \begin{cases} 
                        % \sigma_{{\rm id}_n}  & \mbox{ for } k = l \\  
                        \left( \begin{array}{ccccccccccc} 
                                        1 & \dots & k-1 & k & k+1 & \dots & l-1 & l   & l+1 & \dots & n \\
                                        1 & \dots & k-1 & l & k   & \dots & l-2 & l-1 & l+1 & \dots & n 
                                \end{array}\right) & \mbox{ for } k < l \\
                        \left(\begin{array}{ccccccccccc} 
                                        1 & \dots &  l-1 & l   & l+1 & \dots & k-1 & k & k+1 & \dots & n \\
                                        1 & \dots &  l-1 & l+1 & l+2 & \dots & k   & l & k+1 & \dots & n 
                               \end{array}\right) & \mbox{ for } k > l
                    \end{cases}

We define the corresponding system objects as :math:`W_{l \gets k}^{(n)} := P_{\omega_{l \gets k}^{(n)}}`.

Permutations and Concatenations
-------------------------------

Given a series :math:`P_{\sigma} \lhd (Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N)` where the :math:`Q_j` are irreducible systems, we analyze in which cases it is possible to (partially) "move the permutation through" the concatenated expression. Obviously we could just as well investigate the opposite scenario :math:`(Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N) \lhd P_{\sigma}`, but this second scenario is closely related\footnote{Series-Inverting a series product expression also results in an inverted order of the operand inverses :math:`(Q_1 \lhd Q_2)^{\lhd -1} = Q_2^{\lhd-1} \lhd Q_1^{\lhd-1}`. Since the inverse of a permutation (concatenation) is again a permutation (concatenation), the cases are in a way "dual" to each other.}.

**Block-permuting permutations**

The simples case is realized when the permutation simply permutes whole blocks intactly (fig ref block_permuting_permutations)
(fig, block_permuting_permutations, A block permuting series).

Given a block structure :math:`\mathbf{n} := (n_1, n_2, \dots n_N)` a permutation :math:`\sigma \in \Sigma_n` is said to *block permute* :math:`\mathbf{n}` iff there exists a permutation :math:`\tilde{\sigma} \in \Sigma_N` such that 

.. math::

    P_{\sigma} \lhd (Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N) & = \left(P_{\sigma} \lhd (Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N) \lhd P_{\sigma^{-1}}\right) \lhd P_{\sigma} \\
    & = (Q_{\tilde{\sigma}(1)} \boxplus Q_{\tilde{\sigma}(2)} \boxplus \dots \boxplus Q_{\tilde{\sigma}(N)}) \lhd P_{\sigma}

Hence, the permutation :math:`\sigma`, given in image tuple notation, block permutes :math:`\mathbf{n}` iff for all :math:`1 \le j \le N`  and for all :math:`0 \le k < n_j` we have :math:`\sigma(o_j + k) = \sigma(o_j) + k`, where we have introduced the block offsets :math:`o_j := 1 + \sum_{j' < j} n_j`.
When these conditions are satisfied, :math:`\tilde{\sigma}` may be obtained by demanding that :math:`\tilde{\sigma}(a) > \tilde{\sigma}(b) \Leftrightarrow \sigma(o_a) > \sigma(o_b)`. This equivalence reduces the computation of :math:`\tilde{\sigma}` to sorting a list in a specific way.

**Block-factorizing permutations**

The next-to-simplest case is realized when a permutation :math:`\sigma` can be decomposed :math:`\sigma = \sigma_{\rm b} \circ \sigma_{\rm i}` into a permutation :math:`\sigma_{\rm b}` that block permutes the block structure :math:`\mathbf{n}` and an internal permutation :math:`\sigma_{\rm i}` that only permutes within each block, i.e.~:math:`\sigma_{\rm i} = \sigma_1 \boxplus \sigma_2 \boxplus \dots \boxplus \sigma_N`. In this case we can perform the following simplifications

.. math::
    
    P_{\sigma} \lhd (Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N) = P_{\sigma_b} \lhd \left[ (P_{\sigma_1} \lhd Q_1) \boxplus (P_{\sigma_2} \lhd Q_2) \boxplus \dots \boxplus (P_{\sigma_N} \lhd Q_N)\right].

We see that we have reduced the problem to the above discussed case. The result is now

.. math::
    
    P_{\sigma} \lhd (Q_1 \boxplus \dots \boxplus Q_N) = \left[ (P_{\sigma_{\tilde{\sigma_{\rm b}}(1)}} \lhd Q_{\tilde{\sigma_{\rm b}}(1)}) \boxplus \dots \boxplus (P_{\sigma_{\tilde{\sigma_{\rm b}}(N)}} \lhd Q_{\tilde{\sigma_{\rm b}}(N)})\right] \lhd P_{\sigma_{\rm b}}.

In this case we say that :math:`\sigma` *block factorizes* according to the block structure :math:`\mathbf{n}`. 
Figure  illustrates an example of this case.

(fig, block_factorizing_permutations, A block factorizable series.)

A permutation :math:`\sigma` block factorizes according to the block structure :math:`\mathbf{n}` iff for all :math:`1 \le j \le N` we have :math:`\max_{0 \le k < n_j}\sigma(o_j + k)  - \min_{0 \le k' < n_j}\sigma(o_j + k') = n_j - 1`, with the block offsets defined as above. In other words, the image of a single block is coherent in the sense that no other numbers from outside the block are mapped into the integer range spanned by the minimal and maximal points in the block's image. The equivalence follows from our previous result and the bijectivity of :math:`\sigma`.

**The general case**

In general there exists no unique way how to split apart the action of a permutation on a block structure. However, it is possible to define a some rules that allow us to "move as much of the permutation" as possible to the RHS of the series. This involves the factorization :math:`\sigma = \sigma_{\rm x} \circ \sigma_{\rm b} \circ \sigma_{\rm i}` defining a specific way of constructing both :math:`\sigma_{\rm b}` and :math:`\sigma_{\rm i}` from :math:`\sigma`. The remainder :math:`\sigma_{\rm x}` can then be calculated through

.. math::
    
    \sigma_{\rm x} := \sigma \circ \sigma_{\rm i}^{-1} \circ \sigma_{\rm b}^{-1}.

Hence, by construction, :math:`\sigma_{\rm b} \circ \sigma_{\rm i}` factorizes according to :math:`\mathbf{n}` so only :math:`\sigma_{\rm x}` remains on the exterior LHS of the expression.

So what then are the rules according to which we construct the block permuting :math:`\sigma_{\rm b}` and the decomposable :math:`\sigma_{\rm i}`?
We wish to define :math:`\sigma_{\rm i}` such that the remainder :math:`\sigma \circ \sigma_{\rm i}^{-1} = \sigma_{\rm x} \circ \sigma_{\rm b}` does not cross any two signals that are emitted from the same block. Since by construction :math:`\sigma_{\rm b}` only permutes full blocks anyway this means that :math:`\sigma_{\rm x}` also does not cross any two signals emitted from the same block.
This completely determines :math:`\sigma_{\rm i}` and we can therefore calculate :math:`\sigma \circ \sigma_{\rm i}^{-1} = \sigma_{\rm x} \circ \sigma_{\rm b}` as well. To construct :math:`\sigma_{\rm b}` it is sufficient to define an total order relation on the blocks that only depends on the block structure :math:`\mathbf{n}` and on :math:`\sigma \circ \sigma_{\rm i}^{-1}`. We define the order on the blocks such that they are ordered according to their minimal image point under :math:`\sigma`. Since :math:`\sigma \circ \sigma_{\rm i}^{-1}` does not let any block-internal lines cross, we can thus order the blocks according to the order of the images of the first signal :math:`\sigma \circ \sigma_{\rm i}^{-1}(o_j)`. In (ref fig general_factorization) we have illustrated this with an example.

(fig, general_factorization, A general series with a non-factorizable permutation. In the intermediate step \ref{fig:bfg1a} we have explicitly separated :math:`\sigma = \sigma_{\rm x} \circ \sigma_{\rm b} \circ \sigma_{\rm i}`).

Finally, it is a whole different question, why we would want move part of a permutation through the concatenated expression in this first place as the expressions usually appear to become more complicated rather than simpler. This is, because we are currently focussing only on single series products between two systems. In a realistic case we have many systems in series and among these there might be quite a few permutations. Here, it would seem advantageous to reduce the total number of permutations within the series by consolidating them where possible: :math:`P_{\sigma_2} \lhd P_{\sigma_1} = P_{\sigma_2 \circ \sigma_1}`. To do this, however, we need to try to move the permutations through the full series and collect them on one side (in our case the RHS) where they can be combined to a single permutation.
Since it is not always possible to move a permutation through a concatenation (as we have seen above), it makes sense to at some point in the simplification process reverse the direction in which we move the permutations and instead collect them on the LHS. Together these two strategies achieve a near perfect permutation simplification.

Feedback of a concatenation
---------------------------

A feedback operation on a concatenation can always be simplified in one of two ways: If the outgoing and incoming feedback ports belong to the same irreducible subblock of the concatenation, then the feedback can be directly applied only to that single block. For an illustrative example see (Figure ref fc_irr).

(fig, fc_irr, Reduction to feedback of subblock).


If, on the other, the outgoing feedback port is on a different subblock than the incoming, the resulting circuit actually does not contain any real feedback and we can find a way to reexpress it algebraically by means of a series product (cf. Figures fc_re1 and fc_re2).

(fig, fc_re1, Reduction of feedback to series, first example)

(fig,fc_re2, Reduction of feedback to series, second example)


To discuss the case in full generality consider the feedback expression :math:`[A \boxplus B]_{k \to l}` with :math:`\cdim{A} = n_A` and  :math:`\cdim{B} = n_B` and where :math:`A` and :math:`B` are not necessarily irreducible.
There are four different cases to consider.

    * :math:`k,l \le n_A`: In this case the simplified expression should be :math:`[A]_{k \to l} \boxplus B`
    
    * :math:`k,l > n_A`: Similarly as before but now the feedback is restricted to the second operand :math:`A \boxplus [B]_{(k-n_A) \to (l-n_A)}`, cf. Fig. (ref fig fc_irr).
    
    * :math:`k \le n_A < l`: This corresponds to a situation that is actually a series and can be re-expressed as :math:`(\id{n_A - 1} \boxplus B) \lhd W_{(l-1) \gets k}^{(n)} \lhd (A + \id{n_B - 1})`, cf. Fig. (ref fig fc_re1).
    
    * :math:`l \le n_A < k`: Again, this corresponds a series but with a reversed order compared to above :math:`(A + \id{n_B - 1}) \lhd W_{l \gets (k-1)}^{(n)} \lhd (\id{n_A - 1} \boxplus B)`, cf. Fig. (ref fig fc_re2).

Feedback of a series
--------------------

There are two important cases to consider for the kind of expression at either end of the series:
A series starting or ending with a permutation system or a series starting or ending with a concatenation.

(fig, fs_c, Reduction of series feedback with a concatenation at the RHS)

(fig, fs_p, Reduction of series feedback with a permutation at the RHS)


    1) :math:`[A \lhd (C \boxplus D)]_{k \to l}`: We define :math:`n_C = \cdim{C}` and :math:`n_A = \cdim{A}`. Without too much loss of generality, let's assume that :math:`l \le n_C` (the other case is quite similar). We can then pull :math:`D` out of the feedback loop:
    :math:`[A \lhd (C \boxplus D)]_{k \to l} \longrightarrow [A \lhd (C \boxplus \id{n_D})]_{k \to l} \lhd (\id{n_C - 1} \boxplus D)`.
    Obviously, this operation only makes sense if :math:`D \neq \id{n_D}`. The case :math:`l>n_C` is quite similar, except that we pull :math:`C` out of the feedback. See Figure (ref fig fs_c) for an example.
    
    2) We now consider :math:`[(C \boxplus D) \lhd E]_{k \to l}` and we assume :math:`k \le n_C` analogous to above. Provided that :math:`D \neq \id{n_D}`, we can pull it out of the feedback and get :math:`(\id{n_C -1} \boxplus D) \lhd [(C \boxplus \id{n_D}) \lhd E]_{k \to l}`.
    
    3) :math:`[A \lhd P_\sigma]_{k \to l}`: The case of a permutation within a feedback loop is a lot more intuitive to understand graphically (e.g., cf. Figure ref fig fs_p). Here, however we give a thorough derivation of how a permutation can be reduced to one involving one less channel and moved outside of the feedback.
    First, consider the equality :math:`[A \lhd W_{j \gets l}^{(n)}]_{k \to l} = [A]_{k \to j}` which follows from the fact that :math:`W_{j \gets l}^{(n)}` preserves the order of all incoming signals except the :math:`l`-th.
    Now, rewrite 
    
    .. math::
    
            [A \lhd P_\sigma]_{k \to l} & = [A \lhd P_\sigma \lhd W_{l \gets n}^{(n)} \lhd W_{n \gets l}^{(n)}]_{k \to l} \\
                                        & = [A \lhd P_\sigma \lhd W_{l \gets n}^{(n)} ]_{k \to n} \\
                                        & = [A \lhd W_{\sigma(l) \gets n}^{(n)} \lhd (W_{n \gets \sigma(l)}^{(n)} \lhd P_\sigma \lhd W_{l \gets n}) ]_{k \to n}
    
    Turning our attention to the bracketed expression within the feedback, we clearly see that it must be a permutation system :math:`P_{\sigma'} = W_{n \gets \sigma(l)}^{(n)} \lhd P_\sigma \lhd W_{l \gets n}^{(n)}` that maps :math:`n \to l \to \sigma(l) \to n`. We can therefore write :math:`\sigma' = \tilde{\sigma} \boxplus \sigma_{{\rm id}_1}` or equivalently :math:`P_{\sigma'} = P_{\tilde{\sigma}} \boxplus \id{1}` But this means, that the series within the feedback ends with a concatenation and from our above rules we know how to handle this:
    
    .. math::
    
            [A \lhd P_\sigma]_{k \to l} & = [A \lhd W_{\sigma(l) \gets n}^{(n)} \lhd (P_{\tilde{\sigma}} \boxplus \id{1})]_{k \to n} \\
                                        & = [A \lhd W_{\sigma(l) \gets n}^{(n)}]_{k \to n} \lhd P_{\tilde{\sigma}} \\
                                        & = [A]_{k \to \sigma(l)} \lhd P_{\tilde{\sigma}},
    
    where we know that the reduced permutation is the well-defined restriction to :math:`n-1` elements of :math:`\sigma' = \left(\omega_{n \gets \sigma{l}}^{(n)} \circ \sigma \circ \omega_{l \gets n}^{(n)}\right)`.

    4) The last case is analogous to the previous one and we will only state the results without a derivation:
    
    .. math::
    
             [P_\sigma \lhd A]_{k \to l} & = P_{\tilde{\sigma}} \lhd  [A]_{\sigma^{-1}(k) \to l},
    
    where the reduced permutation is given by the (again well-defined) restriction of :math:`\omega_{n \gets k}^{(n)} \circ \sigma \circ \omega_{\sigma^{-1}(k) \gets n}^{(n)}` to :math:`n-1` elements.


.. [1] Gough, James & Nurdin (2010). Squeezing components in linear quantum feedback networks. Physical Review A, 81(2). doi:10.1103/PhysRevA.81.023804
.. [2] Gough & James (2008). Quantum Feedback Networks: Hamiltonian Formulation. Communications in Mathematical Physics, 287(3), 1109-1132. doi:10.1007/s00220-008-0698-8
.. [3] Gough & James (2009). The Series Product and Its Application to Quantum Feedforward and Feedback Networks. IEEE Transactions on Automatic Control, 54(11), 2530-2544. doi:10.1109/TAC.2009.2031205

"""
from __future__ import division
import os, time
from operator_algebra import *


class CannotConvertToSLH(AlgebraException):
    """
    Is raised when a circuit algebra object cannot be converted to a concrete SLH object.
    """


class CannotConvertToABCD(AlgebraException):
    """
    Is raised when a circuit algebra object cannot be converted to a concrete ABCD object.
    """


class CannotVisualize(AlgebraException):
    """
    Is raised when a circuit algebra object cannot be visually represented.
    """

class WrongCDimError(AlgebraError):
    """
    Is raised when two object are tried to joined together in series but have different channel dimensions.
    """


class CircuitVisualizer(object):
    """
    Visualization wrapper class that implements IPython's _repr_png_ method to
    generate a graphical representation (in PNG format) of its circuit object.
    Use as::

        CircuitVisualizer(circuit)

    :param circuit: The circuit expression to visualize
    :type circuit: Circuit
    """

    _circuit = None

    def __init__(self, circuit):
        #noinspection PyRedeclaration
        self._circuit = circuit

    def _repr_png_(self):
        import qnet.misc.circuit_visualization as circuit_visualization
        from tempfile import gettempdir

        tmp_dir = gettempdir()
        fname = tmp_dir + "/tmp_{}.png" .format(hash(str(self._circuit)))

        if circuit_visualization.draw_circuit(self._circuit, fname):

            for k in range(5):
                if os.path.exists(fname):
                    break
                else:
                    time.sleep(.5)

            try:
                with open(fname, "rb") as png_file:
                    fdata = png_file.read()
                os.remove(fname)
                return fdata
            except:
                print ("Could not open visualization file for {!s}".format(self._circuit))
                raise CannotVisualize()

        else:
            raise CannotVisualize()

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self._circuit)

class IncompatibleBlockStructures(AlgebraError):
    """
    Is raised when a circuit decomposition into a block-structure is requested
    that is icompatible with the actual block structure of the circuit expression.
    """



class Circuit(object):
    """
    Abstract base class for the circuit algebra elements.
    """

    __metaclass__ = ABCMeta

    @property
    def cdim(self):
        """
        The channel dimension of the circuit expression,
        i.e. the number of external bosonic noises/inputs that the circuit couples to.
        """
        return self._cdim

    @abstractproperty
    def _cdim(self):
        raise NotImplementedError(self.__class__.__name__)

    @property
    def block_structure(self):
        """
        If the circuit is *reducible* (i.e., it can be represented as a :py:class:Concatenation: of individual circuit expressions),
        this gives a tuple of cdim values of the subblocks.
        E.g. if A and B are irreducible and have ``A.cdim = 2``, ``B.cdim = 3``

            >>> A = CircuitSymbol('A', 2)
            >>> B = CircuitSymbol('B', 3)

        Then the block structure of their Concatenation is:

            >>> (A + B).block_structure
                (2,3),

        while

            >>> A.block_structure
                (2,)
            >>> B.block_structure
                (3,)

        """
        return self._block_structure

    @property
    def _block_structure(self):
        return self.cdim,

    def index_in_block(self, channel_index):
        """
        Yield the index a channel has within the subblock it belongs to.
        I.e., only for reducible circuits, this gives a result different from the argument itself.

        :param channel_index: The index of the external channel
        :type channel_index: int
        :return: The index of the external channel within the subblock it belongs to.
        :rtype: int
        :raise: ValueError

        """
        if channel_index < 0 or channel_index >= self.cdim:
            raise ValueError()

        struct = self.block_structure

        if len(struct) == 1:
            return channel_index, 0
        i = 1
        while sum(struct[:i]) <= channel_index and i < self.cdim:
            i +=1
        block_index = i - 1
        index_in_block = channel_index - sum(struct[:block_index])

        return index_in_block, block_index


    def get_blocks(self, block_structure = None):
        """
        For a reducible circuit, get a sequence of subblocks that when concatenated again yield the original circuit.
        The block structure given has to be compatible with the circuits actual block structure,
        i.e. it can only be more coarse-grained.

        :param block_structure: The block structure according to which the subblocks are generated (default = ``None``, corresponds to the circuit's own block structure)
        :type block_structure: tuple
        :return: A tuple of subblocks that the circuit consists of.
        :rtype: tuple
        :raises: IncompatibleBlockStructures
        """
        if block_structure is None:
            #noinspection PyRedeclaration
            block_structure = self.block_structure
        try:
            return self._get_blocks(block_structure)
        except IncompatibleBlockStructures as e:
            raise e

    def _get_blocks(self, block_structure):
        if block_structure == self.block_structure:
            return (self, )
        raise IncompatibleBlockStructures("Requested incompatible block structure %s" % (block_structure,))

    def series_inverse(self):
        """
        Return the inverse object (under the series product) for a circuit.
        In general for any X

            >>> X << X.series_inverse() == X.series_inverse() << X == cid(X.cdim)
                True
        """
        return self._series_inverse()

    def _series_inverse(self):
        return SeriesInverse.create(self)

    def feedback(self, out_index = None, in_index = None):
        """
        Return a circuit with self-feedback from the output port (zero-based) ``out_index`` to the input port ``in_index``.

        :param out_index: The output port from which the feedback connection leaves (zero-based, default = ``None`` corresponds to the *last* port).
        :type out_index: int or NoneType
        :param in_index: The input port into which the feedback connection goes (zero-based, default = ``None`` corresponds to the *last* port).
        :type in_index: int or NoneType
        """
        if out_index is None:
            #noinspection PyRedeclaration
            out_index = self.cdim -1
        if in_index is None:
            #noinspection PyRedeclaration
            in_index = self.cdim -1

        return self._feedback(out_index, in_index)

    def _feedback(self, out_index, in_index):
        return Feedback.create(self, out_index, in_index)

    def show(self):
        """
        Show the circuit expression in an IPython notebook.
        """
        return CircuitVisualizer(self)

    def creduce(self):
        """
        If the circuit is reducible, try to reduce each subcomponent once.
        Depending on whether the components at the next hierarchy-level are themselves reducible,
        successive ``circuit.creduce()`` operations yields an increasingly fine-grained decomposition of a circuit into its most primitive elements.
        """
        return self._creduce()

    @abstractmethod
    def _creduce(self):
        return self

    def toSLH(self):
        """
        Return the SLH representation of a circuit. This can fail if there are un-substituted pure circuit all_symbols (:py:class:`CircuitSymbol`) left
        in the expression or if the circuit includes *non-passive* ABCD models (cf. [1]_)


        """
        return self._toSLH()

    @abstractmethod
    def _toSLH(self):
        raise NotImplementedError(self.__class__.__name__)

    def toABCD(self, linearize = False):
        """
        Return the ABCD representation of a circuit expression. If `linearize=True` all operator expressions giving rise to non-linear equations of motion are dropped.
        This can fail if there are un-substituted pure circuit all_symbols (:py:class:`CircuitSymbol`) left in the expression or if `linearize = False` and the circuit includes non-linear SLH models.
        (cf. [1]_)


        :param linearize: Whether or not to explicitly neglect non-linear contributions (default = False)
        :type linearize: bool
        :return: ABCD model for the circuit
        :rtype: ABCD
        """
        return self._toABCD(linearize)

    @abstractmethod
    def _toABCD(self, linearize):
        raise NotImplementedError(self.__class__.__name__)

    def coherent_input(self, *input_amps):
        """
        Feed coherent input amplitudes into the circuit.
        E.g. For a circuit with channel dimension of two,
        `C.coherent_input(0,1)` leads to an input amplitude of zero into the first and one into the second port.

        :param input_amps: The coherent input amplitude for each port
        :type input_amps: any of :py:attr:`qnet.algebra.operator_algebra.Operator.scalar_types`
        :return: The circuit including the coherent inputs.
        :rtype: Circuit
        :raise: WrongCDimError
        """
        return self._coherent_input(*input_amps)

    def _coherent_input(self, *input_amps):
        if len(input_amps) != self.cdim:
            raise WrongCDimError()
        return self << SLH(identity_matrix(self.cdim), Matrix((input_amps,)).T, 0)

    @property
    def space(self):
        """
        All Hilbert space degree of freedoms associated with a given circuit component.
        """
        return self._space

    @abstractproperty
    def _space(self):
        raise NotImplementedError(self.__class__)


    def __lshift__(self, other):
        if isinstance(other, Circuit):
            return SeriesProduct.create(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Circuit):
            return Concatenation.create(self, other)
        return NotImplemented



class SLH(Circuit, Operation):
    """
    SLH class to encapsulate an open system model that is parametrized as described in [2]_ , [3]_ ::

        SLH(S, L, H)

    :param S: The scattering matrix (with in general Operator-valued elements)
    :type S: Matrix
    :param L: The coupling vector (with in general Operator-valued elements)
    :type L: Matrix
    :param H: The internal Hamilton operator
    :type H: Operator
    :raise: ValueError
    """

    #noinspection PyRedeclaration,PyUnresolvedReferences
    def __init__(self, S, L, H):
        if not isinstance(S, Matrix):
            S = Matrix(S)
        if not isinstance(L, Matrix):
            L = Matrix(L)
        if S.shape[0] != L.shape[0]:
            raise ValueError('S and L misaligned: S = {!r}, L = {!r}'.format(S, L))

        #noinspection PyArgumentList
        super(SLH, self).__init__(S, L, H)

    @property
    def _cdim(self):
        return self.S.shape[0]

    def _creduce(self):
        return self

    @property
    def S(self):
        """
        The scattering matrix (with in general Operator-valued elements) of shape ``(cdim, cdim)``

        :rtype: Matrix
        """
        return self.operands[0]

    @property
    def L(self):
        """
        The coupling vector (with in general Operator-valued elements) of shape ``(cdim, 1)``

        :rtype: Matrix
        """
        return self.operands[1]

    #noinspection PyRedeclaration
    @property
    def H(self):
        """
        The internal Hamilton operator.

        :rtype: Operator
        """
        return self.operands[2]


    @property
    def _space(self):
        return self.S.space * self.L.space * space(self.H)


    def series_with_slh(self, other):
        """
        Evaluate the series product with another :py:class:``SLH`` object.

        :param other: An upstream SLH circuit.
        :type other: SLH
        :return: The combines system.
        :rtype: SLH
        """
        new_S = self.S * other.S
        new_L = self.S * other.L + self.L

        delta =  ImAdjoint(self.L.adjoint() * self.S * other.L)

        if isinstance(delta, Matrix):
            new_H = self.H + other.H + delta[0,0]
        else:
            assert delta == 0
            new_H = self.H + other.H

        return SLH(new_S, new_L, new_H)

    def concatenate_slh(self, other):
        """
        Evaluate the concatenation product with another SLH object.

        :param other: Another SLH object
        :type other: SLH
        :return: The combined system
        :rtype: SLH
        """
        selfS = self.S
        otherS = other.S
        new_S = block_matrix(selfS, zeros((selfS.shape[0],otherS.shape[1]), dtype = int), zeros((otherS.shape[0],selfS.shape[1]), dtype = int), otherS)
        new_L = vstack((self.L, other.L))
        new_H = self.H + other.H

        return SLH(new_S, new_L, new_H)

    def __str__(self):
        return "({!s}, {!s}, {!s})".format(self.S, self.L, self.H)

    def _tex(self):
        return r"\left( {}, {}, {} \right)".format(tex(self.S), tex(self.L), tex(self.H))

    def _toSLH(self):
        return self

    def expand(self):
        """
        Expand out all operator expressions within S, L and H and return a new SLH object with these expanded expressions.

        :rtype: SLH
        """
        return SLH(self.S.expand(), self.L.expand(), self.H.expand())

    def HL_to_qutip(self, full_space = None):
        """
        Generate and return QuTiP representation matrices for the Hamiltonian and the collapse operators.

        :param full_space: The Hilbert space in which to represent the operators.
        :type full_space: HilbertSpace or None
        :return tuple: (H, [L1, L2, ...]) as numerical qutip.Qobj representations.
        """
        if full_space:
            if not full_space >= self.space:
                raise AlgebraError("full_space = {} needs to at least include self.space = {}".format(str(full_space), str(self.space)))
        else:
            full_space = self.space
        H = self.H.to_qutip(full_space)
        Ls = [L.to_qutip(full_space) for L in self.L.matrix.flatten() if isinstance(L, Operator)]

        return H, Ls



#    _block_structure_ = None
#
#    @property
#    def _block_structure(self):
#        if self.cdim and not self._block_structure_:
#            self._block_structure_ = self.S.block_structure
#        return self._block_structure_
#
#    def _get_blocks(self, block_structure):
#        Sblocks = self.S._get_blocks(block_structure)
#        Lblocks = self.L._get_blocks(block_structure)
#        Hblocks = (self.H,) + ((0,)*(len(block_structure)-1))
#        return tuple(SLH(S,L,H) for (S,L,H) in zip(Sblocks, Lblocks, Hblocks))


    def _series_inverse(self):
        return SLH(self.S.adjoint(), - self.S.adjoint()*self.L, -self.H)

    def _feedback(self, out_index, in_index):
        if not isinstance(self.S, Matrix) or not isinstance(self.L, Matrix):
            return Feedback(self, out_index, in_index)

        n = self.cdim - 1

        if out_index != n:
            return (map_signals_circuit({out_index:n}, self.cdim).toSLH() << self).feedback(in_index = in_index)
        elif in_index != n:
            return (self << map_signals_circuit({n:in_index}, self.cdim).toSLH()).feedback()


        S, L, H = self.operands

        one_minus_Snn = sympyOne - S[n,n]

        if isinstance(one_minus_Snn, Operator):
            if isinstance(one_minus_Snn, ScalarTimesOperator) and one_minus_Snn.operands[1] == IdentityOperator():
                one_minus_Snn = one_minus_Snn.coeff
            else:
                raise AlgebraError('Inversion not implemented for general operators')

        one_minus_Snn_inv = sympyOne/one_minus_Snn

        new_S = S[:n,:n] + S[0:n , n:] * one_minus_Snn_inv * S[n:, 0 : n]

        new_L = L[:n] + S[0:n, n] * one_minus_Snn_inv * L[n]
        delta_H  = Im((L.adjoint() * S[:,n:]) * one_minus_Snn_inv * L[n,0])

        if isinstance(delta_H, Matrix):
            delta_H = delta_H[0,0]
        new_H = H + delta_H

        return SLH(new_S, new_L, new_H)


    #noinspection PyRedeclaration
    def symbolic_liouvillian(self, rho = None):
        """
        Compute the symbolic Liouvillian acting on a state rho.
        If no rho is given, an OperatorSymbol is created in its place.
        This correspnds to the RHS of the master equation
        in which an average is taken over the external noise degrees of freedom.

        :param rho: A symbolic density matrix operator
        :type rho: Operator
        :return: The RHS of the master equation.
        :rtype: Operator
        """
        L, H = self.L, self.H
        if rho is None:
            rho = OperatorSymbol('rho', self.space)
        return -I*(H*rho - rho*H) + sum( Lk * rho * adjoint(Lk)
                             -  (adjoint(Lk)*Lk * rho + rho * adjoint(Lk)*Lk) / 2
                                                for Lk in L.matrix.flatten())


    #noinspection PyRedeclaration
    def symbolic_heisenberg_eom(self, X = None, noises = None):
        """
        Compute the symbolic Heisenberg equations of motion of a system operator X.
        If no X is given, an OperatorSymbol is created in its place.
        If no noises are given, this correspnds to the ensemble-averaged Heisenberg equation of motion.

        :param X: A system operator
        :type X: Operator
        :param noises: A vector of noise inputs
        :type noises: Operator
        :return: The RHS of the Heisenberg equations of motion of X.
        :rtype: Operator
        """
        L, H = self.L, self.H

        if X is None:
            X = OperatorSymbol('X', L.space | H.space)

        ret =  I*(H*X - X*H) + sum(adjoint(Lk)* X * Lk \
                    -  (adjoint(Lk)*Lk * X + X * adjoint(Lk)*Lk) / 2 \
                                                            for Lk in L.matrix.flatten())
        if noises is not None:
            if not isinstance(noises, Matrix):
                noises = Matrix(noises)
            LambdaT = (noises.conjugate() * noises.transpose()).transpose()
            assert noises.shape == L.shape
            S = self.S
            ret += (adjoint(noises) * S.adjoint() * (X * L - L * X)).evalf()[0,0] \
                    + ((L.adjoint() *X - X * L.adjoint()) * S * noises).evalf()[0,0]
            if len(S.space & X.space):
                comm = (S.adjoint() * X * S - X)
                ret +=  (comm * LambdaT).evalf().trace()
        return ret

    def __iter__(self):
        return iter((self.S, self.L, self.H))

    def __len__(self):
        return 3

    def _toABCD(self, linearize):
        #TODO implement SLH._toABCD()
        pass

#    def _mathematica(self):
#        return "SLH[%s, %s, %s]" % (mathematica(self.S), mathematica(self.L), mathematica(self.H))


#TODO ADD ABCD class and toABCD() methods
@check_signature
class ABCD(Circuit, Operation):
    r"""
    ABCD model class in amplitude representation.

        ``ABCD(A, B, C, D, w, space)``

    I.e. for a doubled up vector a = (a_1, ..., a_n, a_1^*, ... a_n^*)^T = double_up((a_1, ..., a_n)^T)
    and doubled up noises dA = (dA_1, ..., dA_m, dA_1^*, ..., dA_m^*)^T = double_up((dA_1, ..., dA_n)^T)
    The equation of motion for a is

    .. math::
        da = A a dt + B (dA + double_up(w) dt)

    The output field dA' is given by

    .. math::
        dA' = C a dt + D (dA + double_up(w) dt)


    :param A: Coupling matrix: internal to internal, scalar valued elements, ``shape = (2*n,2*n)``
    :type A: Matrix
    :param B: Coupling matrix external input to internal, scalar valued elements, ``shape = (2*n,2*m)``
    :type B: Matrix
    :param C: Coupling matrix internal to external output, scalar valued elements, ``shape = (2*m,2*n)``
    :type C: Matrix
    :param D: Coupling matrix external input to output, scalar valued elements, ``shape = (2*m,2*m)``
    :type D: Matrix
    :param w: Coherent input amplitude vector, **NOT DOUBLED UP**, scalar valued elements, ``shape = (m,1)``
    :type w: Matrix
    :param space: Hilbert space with exactly n local factor spaces corresponding to the n internal degrees of freedom.
    :type space: HilbertSpace
    """
    signature = Matrix, Matrix, Matrix, Matrix, Matrix, HilbertSpace

    #noinspection PyArgumentList
    def __init__(self, A, B, C, D, w, space):
        super(ABCD, self).__init__(A, B, C, D, w, space)

    @classmethod
    def create(cls, A, B, C, D, w, space):
        """
        See ABCD documentation
        """
        n2, m2 = B.shape
        if not (n2%2):
            raise ValueError()
        if not (m2%2):
            raise ValueError()
        n, m = n2/2, m2/2

        if not A.shape == (n2,n2):
            raise ValueError()

        if not C.shape == (m2, n2):
            raise ValueError()

        if not D.shape == (m2, m2):
            raise ValueError()

        if not w.shape == (m, 1):
            raise ValueError()

        if not len(space.local_factors()) == n:
            raise AlgebraError(str(space) + " != " + str(n))

        return super(ABCD,cls).create(A, B, C, D, w, space)

    @property
    def A(self):
        """Coupling matrix: internal to internal, scalar valued elements, ``shape = (2*n,2*n)``"""
        return self.operands[0]

    @property
    def B(self):
        """Coupling matrix external input to internal, scalar valued elements, ``shape = (2*n,2*m)``"""
        return self.operands[1]

    @property
    def C(self):
        """Coupling matrix internal to external output, scalar valued elements, ``shape = (2*m,2*n)``"""
        return self.operands[2]

    @property
    def D(self):
        """Coupling matrix external input to output, scalar valued elements, ``shape = (2*m,2*m)``"""
        return self.operands[3]

    @property
    def w(self):
        """Coherent input amplitude vector, **NOT DOUBLED UP**, scalar valued elements, ``shape = (m,1)``"""
        return self.operands[4]


    @property
    def _space(self):
        """
        :rtype: HilbertSpace
        """
        return self.operands[5]

    @property
    def n(self):
        """
        The number of oscillators.

        :rtype: int
        """
        return len(self.space.local_factors())

    @property
    def m(self):
        """
        The number of external fields.

        :rtype: int
        """
        return self.D.shape[0]/2

    @property
    def _cdim(self):
        return self.m

    def _get_blocks(self, block_structure):
        return self,

    @property
    def _block_structure(self):
        return self.cdim,


    def _toABCD(self, linearize):
        return self


    def _toSLH(self):
        # TODO IMPLEMENT ABCD._toSLH()
        doubled_up_as = vstack((Matrix([[Destroy(spc) for spc in self.space.local_factors()]]).T,
                                Matrix([[Create(spc) for spc in self.space.local_factors()]]).T))


@check_signature
class CircuitSymbol(Circuit, Operation):
    """
    Circuit Symbol object, parametrized by an identifier and channel dimension.

        ``CircuitSymbol(identifier, cdim)``

    :type identifier: str
    :type cdim: int >= 0
    """
    signature = str, int

    def __str__(self):
        return self.identifier

    def _tex(self):
        return identifier_to_tex(self.identifier)

    @property
    def identifier(self):
        """
        The symbol identifier

        :type: str
        """
        return self.operands[0]

    @property
    def _cdim(self):
        return self.operands[1]


    def _toABCD(self, linearize):
        raise CannotConvertToABCD()

    def _toSLH(self):
        raise CannotConvertToSLH()

    def _creduce(self):
        return self

    _space = FullSpace

@singleton
class CIdentity(Circuit, Expression):
    """
    Single channel circuit identity system, the neutral element of single channel series products.
    """

    _cdim = 1

    def __str__(self):
        return "cid(1)"

    def _tex(self):
        return r"\mathbf{1}_1"

    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return NotImplemented

        if self.cdim == other.cdim:
            if self is other:
                return True
            try:
                return self.toSLH() == other.toSLH()
            except CannotConvertToSLH:
                return False
        return False

    def _toSLH(self):
        return SLH(Matrix([[1]]), Matrix([[0]]), 0)

    def _creduce(self):
        return self

    def _series_inverse(self):
        return self

    def _toABCD(self, linearize):
        return ABCD(zeros((0,0)), zeros((0,2)), zeros((2,0)), identity_matrix(2), zeros((1,1)), TrivialSpace)

    def _all_symbols(self):
        return {self}

    @property
    def _space(self):
        return TrivialSpace


@singleton
class CircuitZero(Circuit, Expression):
    """
    The zero circuit system, the neutral element of Concatenation. No ports, no internal dynamics.
    """
    _cdim = 0

    def __str__(self):
        return "cid(0)"

    def _tex(self):
        return r"\mathbf{1}_0"

    def __eq__(self, other):
        if self is other:
            return True
        if self.cdim == other.cdim:
            try:
                return self.toSLH() == other.toSLH()
            except CannotConvertToSLH:
                return False
        return False

    def _toSLH(self):
        return SLH(Matrix([[]]), Matrix([[]]), 0)

    def _toABCD(self, linearize):
        return ABCD(zeros((0,0)), zeros((0,0)), zeros((0,0)), zeros((0,0)), zeros((0,0)), TrivialSpace)

    def _creduce(self):
        return self

    def _all_symbols(self):
        return {}

    @property
    def _space(self):
        return TrivialSpace


cid_1 = CIdentity

def circuit_identity(n):
    """
    Return the circuit identity for n channels.

    :param n: The channel dimension
    :type n: int
    :return: n-channel identity circuit
    :rtype: Circuit
    """
    if n <= 0:
        return CircuitZero
    if n == 1:
        return cid_1
    return Concatenation(*((cid_1,)*n))

cid = circuit_identity



#noinspection PyRedeclaration,PyTypeChecker
def get_common_block_structure(lhs_bs, rhs_bs):
    """
    For two block structures ``aa = (a1, a2, ..., an)``, ``bb = (b1, b2, ..., bm)``
    generate the maximal common block structure so that every block from aa and bb
    is contained in exactly one block of the resulting structure.
    This is useful for determining how to apply the distributive law when feeding
    two concatenated Circuit objects into each other.

    Examples:
        ``(1, 1, 1), (2, 1) -> (2, 1)``
        ``(1, 1, 2, 1), (2, 1, 2) -> (2, 3)``

    :param lhs_bs: first block structure
    :type lhs_bs: tuple
    :param rhs_bs: second block structure
    :type rhs_bs: tuple

    """

    # for convenience the arguments may also be Circuit objects
    if isinstance(lhs_bs, Circuit):
        lhs_bs = lhs_bs.block_structure
    if isinstance(rhs_bs, Circuit):
        rhs_bs = rhs_bs.block_structure

    if sum(lhs_bs) != sum(rhs_bs):
        raise AlgebraError('Blockstructures have different total channel numbers.')

    if len(lhs_bs) == len(rhs_bs) == 0:
        return ()

    i = j = 1
    lsum = 0
    while True:
        lsum = sum(lhs_bs[:i])
        rsum = sum(rhs_bs[:j])
        if lsum < rsum:
            i +=1
        elif rsum < lsum:
            j += 1
        else:
            break

    return (lsum, ) + get_common_block_structure(lhs_bs[i:], rhs_bs[j:])




def check_cdims_mtd(dcls, clsmtd, cls, *ops):
    """
    Check that all operands (`ops`) have equal channel dimension.
    """
    if not len({o.cdim for o in ops}) == 1:
        raise ValueError("Not all operands have the same cdim:" + str(ops))
    return clsmtd(cls, *ops)

check_cdims = preprocess_create_with(check_cdims_mtd)




@assoc
@filter_neutral
@check_cdims
@match_replace_binary
@filter_neutral
@check_signature_assoc
class SeriesProduct(Circuit, Operation):
    """
    The series product circuit operation. It can be applied to any sequence of circuit objects that have equal channel dimension.

        ``SeriesProduct(*operands)``

    :param operands: Circuits in feedforward configuration.

    """
    signature = Circuit,
    _binary_rules = [
    ]


    @singleton
    class neutral_element(object):
        """
        Generic neutral element checker of the ``SeriesProduct``, it works for any channel dimension.
        """
        def __eq__(self, other):
#            print "neutral?", other
            return self is other or other == cid(other.cdim)
        def __ne__(self, other):
            return not (self == other)

    @property
    def _cdim(self):
        return self.operands[0].cdim

    def _toSLH(self):
        return reduce(lambda a, b: a.toSLH().series_with_slh(b.toSLH()), self.operands)

    def _creduce(self):
        return SeriesProduct.create(*[op.creduce() for op in self.operands])

    def _series_inverse(self):
        return SeriesProduct.create(*[o.series_inverse() for o in reversed(self.operands)])

    def _tex(self):
        ret  = " \lhd ".join("{{{}}}".format(o.tex()) if not isinstance(o, Concatenation)
                                else r"\left({}\right)".format(o.tex()) for o in self.operands)
#        print ret
        return ret

    def __str__(self):
        return " << ".join(str(o) if not isinstance(o, Concatenation)
                                else r"({!s})".format(o) for o in self.operands)

    def _toABCD(self, linearize):
        # TODO implement SeriesProduct._toABCD()
        pass

    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)


@assoc
@filter_neutral
@match_replace_binary
@filter_neutral
@check_signature_assoc
class Concatenation(Circuit, Operation):
    """
    The concatenation product circuit operation. It can be applied to any sequence of circuit objects.

        ``Concatenation(*operands)``

    :param Circuit operands: Circuits in parallel configuration.
    """

    signature = Circuit,

    neutral_element = CircuitZero

    _binary_rules = []

    def _tex(self):
        ops_strs = []
        id_count = 0
        for o in self.operands:
            if o == CIdentity:
                id_count += 1
            else:
                if id_count > 0:
                    ops_strs += [r"\mathbf{{1}}_{{{}}}".format(id_count)]
                    id_count = 0
                ops_strs += [tex(o) if not isinstance(o, SeriesProduct) else "({})".format(o.tex())]
        if id_count > 0:
            ops_strs += [r"\mathbf{{1}}_{{{}}}".format(id_count)]
        return r" \boxplus ".join(ops_strs)


    def __str__(self):
        ops_strs = []
        id_count = 0
        for o in self.operands:
            if o == CIdentity:
                id_count += 1
            else:
                if id_count > 0:
                    ops_strs += ["cid({})".format(id_count)]
                    id_count = 0
                ops_strs += [str(o) if not isinstance(o, SeriesProduct) else "({!s})".format(o)]
        if id_count > 0:
            ops_strs += ["cid({})".format(id_count)]
        return " + ".join(ops_strs)


    @property
    def _cdim(self):
        return sum((circuit.cdim for circuit in self.operands))

    def _toSLH(self):
        return reduce(lambda a, b: a.toSLH().concatenate_slh(b.toSLH()), self.operands)


    def _creduce(self):
        return Concatenation.create(*[op.creduce() for op in self.operands])

    @property
    def _block_structure(self):
        return sum((circuit.block_structure for circuit in self.operands), ())


    def _get_blocks(self, block_structure):

        blocks = []
        block_iter = iter(sum((op.get_blocks() for op in self.operands), ()))
        cbo = []
        current_length = 0
        for bl in block_structure:
            while current_length < bl:
                next_op = block_iter.next()
                cbo.append(next_op)
                current_length += next_op.cdim
            if current_length != bl:
                raise IncompatibleBlockStructures('requested blocks according to incompatible block_structure')
            blocks.append(Concatenation.create(*cbo))
            cbo = []
            current_length = 0
        return tuple(blocks)


    def _series_inverse(self):
        return Concatenation.create(*[o.series_inverse() for o in self.operands])


    def _feedback(self, out_index, in_index):

        n = self.cdim

        if out_index == n -1 and in_index == n -1:
            return Concatenation.create(*(self.operands[:-1] + (self.operands[-1].feedback(),)))


        in_index_in_block, in_block = self.index_in_block(in_index)
        out_index_in_block, out_block = self.index_in_block(out_index)


        blocks = self.get_blocks()

        if in_block == out_block:

            return Concatenation.create(*blocks[:out_block]) \
                + blocks[out_block].feedback(out_index = out_index_in_block, in_index = in_index_in_block) \
                + Concatenation.create(*blocks[out_block + 1:])
        ### no 'real' feedback loop, just an effective series
        #partition all blocks into just two


        if in_block < out_block:
            b1 = Concatenation.create(*blocks[:out_block])
            b2 = Concatenation.create(*blocks[out_block:])

            return (b1 + circuit_identity(b2.cdim - 1))  \
                    << map_signals_circuit({out_index - 1 :in_index}, n - 1) \
                        << (circuit_identity(b1.cdim - 1) + b2)
        else:
            b1 = Concatenation.create(*blocks[:in_block])
            b2 = Concatenation.create(*blocks[in_block:])

            return (circuit_identity(b1.cdim - 1) + b2) \
                    << map_signals_circuit({out_index : in_index - 1}, n - 1) \
                        << (b1 + circuit_identity(b2.cdim - 1))

    def _toABCD(self, linearize):
        # TODO implement Concatenation._toABCD()
        pass

    @property
    def _space(self):
        return prod((o.space for o in self.operands), TrivialSpace)

class CannotFactorize(Exception):
    pass



@check_signature
class CPermutation(Circuit, Operation):
    r"""
    The channel permuting circuit. This circuit expression is only a rearrangement of input and output fields.
    A channel permutation is given as a tuple of image points. Permutations are usually represented as

    A permutation :math:`\sigma \in \Sigma_n` of :math:`n` elements  is often represented in the following form

    .. math::
         \begin{pmatrix}
                1           &       2   & \dots &   n       \\
                \sigma(1)   & \sigma(2) & \dots & \sigma(n)
        \end{pmatrix},

    but obviously it is fully sufficient to specify the tuple of images :math:`(\sigma(1), \sigma(2), \dots, \sigma(n))`.
    We thus parametrize our permutation circuits only in terms of the image tuple.
    Moreover, we will be working with *zero-based indices*!

    A channel permutation circuit for a given permutation (represented as a python tuple of image indices)
    scatters the :math:`j`-th input field to the :math:`\sigma(j)`-th output field.

    It is instantiated as

        ``CPermutation(permutation)``

    :param permutation: Channel permutation image tuple.
    :type permutation: tuple
    """
    signature = tuple,

    #noinspection PyDocstring
    @classmethod
    def create(cls, permutation):
        """
        See CPermutation docs.
        """
        if not check_permutation(permutation):
            raise BadPermutationError(str(permutation))
        if list(permutation) == range(len(permutation)):
            return cid(len(permutation))
        return super(CPermutation, cls).create(permutation)

    _block_perms = None

    @property
    def block_perms(self):
        """
        If the circuit is reducible into permutations within subranges of the full range of channels,
        this yields a tuple with the internal permutations for each such block.

        :type: tuple
        """
        if not self._block_perms:
            self._block_perms = permutation_to_block_permutations(self.permutation)
        return self._block_perms


    @property
    def permutation(self):
        """
        The permutation image tuple.

        :type: tuple
        """
        return self.operands[0]


    def _toSLH(self):
        return SLH(permutation_matrix(self.permutation), zeros((self.cdim,1)), 0)

    def _toABCD(self, linearize):
        return self.toSLH().toABCD()

    def __str__(self):
        return "P_sigma{!r}".format(self.permutation)

    @property
    def _cdim(self):
        return len(self.permutation)

    def _creduce(self):
        return self

    def _tex(self):
        return "\mathbf{{P}}_\sigma \\begin{{pmatrix}} {} \\\ {} \\end{{pmatrix}}".format(" & ".join(map(str, range(self.cdim))), " & ".join(map(str, self.permutation)))

    def series_with_permutation(self, other):
        """
        Compute the series product with another channel permutation circuit.

        :type other: CPermutation
        :return: The composite permutation circuit (could also be the identity circuit for n channels)
        :rtype: Circuit
        """
        combined_permutation = tuple([self.permutation[p] for p in other.permutation])
        return CPermutation.create(combined_permutation)


    def _series_inverse(self):
        return CPermutation(invert_permutation(self.permutation))

    @property
    def _block_structure(self):
        return tuple(map(len, self.block_perms))

    def _get_blocks(self, block_structure):

        block_perms = []

        if block_structure == self.block_structure:
            return tuple(map(CPermutation.create, self.block_perms))

        if len(block_structure) > len(self.block_perms):
            raise Exception
        if sum(block_structure) != self.cdim:
            raise Exception
        current_perm = []
        block_perm_iter = iter(self.block_perms)
        for l in block_structure:
            while len(current_perm) < l:
                offset = len(current_perm)
                current_perm += [p + offset for p in block_perm_iter.next()]

            if len(current_perm) != l:
                # print block_structure, self.block_perms, block_perms
                raise Exception

            block_perms.append(tuple(current_perm))
            current_perm = []
        return tuple(map(CPermutation.create, block_perms))



    def _factorize_for_rhs(self, rhs):
        """
        Factorize a channel permutation circuit according the block structure of the upstream circuit.
        This allows to move as much of the permutation as possible *around* a reducible circuit upstream.
        It basically decomposes

            ``permutation << rhs --> permutation' << rhs' << residual'``

        where rhs' is just a block permutated version of rhs and residual'
        is the maximal part of the permutation that one may move around rhs.

        :param rhs: An upstream circuit object
        :type rhs: Circuit
        :return: new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit
        :rtype: tuple
        :raise: BadPermutationError
        """
        block_structure = rhs.block_structure

        block_perm, perms_within_blocks = block_perm_and_perms_within_blocks(self.permutation, block_structure)
        fblockp = full_block_perm(block_perm, block_structure)


        if not sorted(fblockp) == range(self.cdim):
            raise BadPermutationError()


        new_rhs_circuit = CPermutation.create(fblockp)
        within_blocks = [CPermutation.create(within_block) for within_block in perms_within_blocks]
        within_perm_circuit = sum(within_blocks, cid(0))
        rhs_blocks = rhs.get_blocks(block_structure)

        permuted_rhs_circuit = Concatenation.create(*[SeriesProduct.create(within_blocks[p], rhs_blocks[p]) \
                                                                for p in invert_permutation(block_perm)])

        new_lhs_circuit = self << within_perm_circuit.series_inverse() << new_rhs_circuit.series_inverse()


        return new_lhs_circuit, permuted_rhs_circuit, new_rhs_circuit




    def _feedback(self, out_index, in_index):
        n = self.cdim
        new_perm_circuit = map_signals_circuit( {out_index: (n-1)}, n) << self << map_signals_circuit({(n-1):in_index}, n)
        if new_perm_circuit == circuit_identity(n):
            return circuit_identity(n-1)
        new_perm = list(new_perm_circuit.permutation)
        n_inv = new_perm.index(n-1)
        new_perm[n_inv] = new_perm[n-1]

        return CPermutation.create(tuple(new_perm[:-1]))


    def _factor_rhs(self, in_index):
        """
        With::

            n           := self.cdim
            in_im       := self.permutation[in_index]
            m_{k->l}    := map_signals_circuit({k:l}, n)

        solve the equation (I) containing ``self``::

            self << m_{(n-1) -> in_index} == m_{(n-1) -> in_im} << (red_self + cid(1))          (I)

        for the (n-1) channel CPermutation ``red_self``.
        Return in_im, red_self.

        This is useful when ``self`` is the RHS in a SeriesProduct Object that is within a Feedback loop
        as it allows to extract the feedback channel from the permutation and moving the
        remaining part of the permutation (``red_self``) outside of the feedback loop.

        :param int in_index: The index for which to factor.
        """
        n = self.cdim
        if not (0 <= in_index < n):
            raise Exception
        in_im = self.permutation[in_index]
        # (I) is equivalent to
        #       m_{in_im -> (n-1)} <<  self << m_{(n-1) -> in_index} == (red_self + cid(1))     (I')
        red_self_plus_cid1 = map_signals_circuit({in_im:(n-1)}, n) << self << map_signals_circuit({(n-1): in_index}, n)
        if isinstance(red_self_plus_cid1, CPermutation):

            #make sure we can factor
            #noinspection PyUnresolvedReferences
            assert red_self_plus_cid1.permutation[(n-1)] == (n-1)

            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])

            return in_im, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_index ->  in_im}

            return in_im, circuit_identity(n-1)

    def _factor_lhs(self, out_index):
        """
        With::

            n           := self.cdim
            out_inv     := invert_permutation(self.permutation)[out_index]
            m_{k->l}    := map_signals_circuit({k:l}, n)

        solve the equation (I) containing ``self``::

            m_{out_index -> (n-1)} << self == (red_self + cid(1)) << m_{out_inv -> (n-1)}           (I)

        for the (n-1) channel CPermutation ``red_self``.
        Return out_inv, red_self.

        This is useful when 'self' is the LHS in a SeriesProduct Object that is within a Feedback loop
        as it allows to extract the feedback channel from the permutation and moving the
        remaining part of the permutation (``red_self``) outside of the feedback loop.

        :param out_index: The index for which to factor
        """
        n = self.cdim
        if not (0 <= out_index < n):
            print self, out_index
            raise Exception
        out_inv = self.permutation.index(out_index)

        # (I) is equivalent to
        #       m_{out_index -> (n-1)} <<  self << m_{(n-1) -> out_inv} == (red_self + cid(1))     (I')

        red_self_plus_cid1 = map_signals_circuit({out_index:(n-1)}, n) << self << map_signals_circuit({(n-1): out_inv}, n)

        if isinstance(red_self_plus_cid1, CPermutation):

            #make sure we can factor
            assert red_self_plus_cid1.permutation[(n-1)] == (n-1)

            #form reduced permutation object
            red_self = CPermutation.create(red_self_plus_cid1.permutation[:-1])

            return out_inv, red_self
        else:
            # 'red_self_plus_cid1' must be the identity for n channels.
            # Actually, this case can only occur
            # when self == m_{in_index ->  in_im}

            return out_inv, circuit_identity(n-1)

    @property
    def _space(self):
        return TrivialSpace



def P_sigma(*permutation):
    """
    Create a channel permutation circuit for the given index image values.
    :param permutation: image points
    :type permutation: int
    :return: CPermutation.create(permutation)
    :rtype: Circuit
    """
    return CPermutation.create(permutation)


def extract_signal(k, n):
    """
    Create a permutation that maps the k-th (zero-based) element to the last element,
    while preserving the relative order of all other elements.
    :param k: The index to extract
    :type k: int
    :param n: The total number of elements
    :type n: int
    :return: Permutation image tuple
    :rtype: tuple
    """
    return tuple(range(k) + [n-1] + range(k, n-1))


def extract_signal_circuit(k, cdim):
    """
    Create a channel permutation circuit that maps the k-th (zero-based) input to the last output,
    while preserving the relative order of all other channels.
    :param k: Extracted channel index
    :type k: int
    :param cdim: The channel dimension
    :type cdim: int
    :return: Permutation circuit
    :rtype: Circuit
"""
    return CPermutation.create(extract_signal(k, cdim))


def map_signals(mapping, n):
    """
    For a given {input:output} mapping in form of a dictionary,
    generate the permutation that achieves the specified mapping
    while leaving the relative order of all non-specified elements intact.
    :param mapping: Input-output mapping of indices (zero-based) {in1:out1, in2:out2,...}
    :type mapping: dict
    :param n: total number of elements
    :type n: int
    :return: Signal mapping permutation image tuple
    :rtype: tuple
    :raise: ValueError
    """
    free_values = range(n)


    for v in mapping.values():
        if v >= n:
            raise ValueError('the mapping cannot take on values larger than cdim - 1')
        free_values.remove(v)
    for k in mapping:
        if k >= n:
            raise ValueError('the mapping cannot map keys larger than cdim - 1')
    # sorted(set(range(n)).difference(set(mapping.values())))
    permutation = []
    # print free_values, mapping, n
    for k in range(n):
        if k in mapping:
            permutation.append(mapping[k])
        else:
            permutation.append(free_values.pop(0))
    # print permutation
    return tuple(permutation)

def map_signals_circuit(mapping, n):
    """
    For a given {input:output} mapping in form of a dictionary,
    generate the channel permutating circuit that achieves the specified mapping
    while leaving the relative order of all non-specified channels intact.
    :param mapping: Input-output mapping of indices (zero-based) {in1:out1, in2:out2,...}
    :type mapping: dict
    :param n: total number of elements
    :type n: int
    :return: Signal mapping permutation image tuple
    :rtype: Circuit
    """
    return CPermutation.create(map_signals(mapping, n))



def pad_with_identity(circuit, k, n):
    """
    Pad a circuit by 'inserting' an n-channel identity circuit at index k.
    I.e., a circuit of channel dimension N is extended to one of channel dimension N+n, where the channels
    k, k+1, ...k+n-1, just pass through the system unaffected.
    E.g. let A, B be two single channel systems

        >>> A = CircuitSymbol('A', 1)
        >>> B = CircuitSymbol('B', 1)
        >>> pad_with_identity(A+B, 1, 2)
            (A + cid(2) + B)

    This method can also be applied to irreducible systems, but in that case the result can not be decomposed as nicely.

    :type circuit: Circuit
    :param k: The index at which to insert the circuit
    :type k: int
    :param n: The number of channels to pass through
    :type n: int
    :return: An extended circuit that passes through the channels k, k+1, ..., k+n-1
    :rtype: Circuit
    """
    circuit_n = circuit.cdim
    combined_circuit = circuit + circuit_identity(n)
    permutation = range(k) + range(circuit_n, circuit_n + n) + range(k, circuit_n)
    return CPermutation.create(invert_permutation(permutation)) << combined_circuit << CPermutation.create(permutation)



@match_replace
@check_signature
class Feedback(Circuit, Operation):
    """
    The circuit feedback operation applied to a circuit of channel dimension > 1
    and an from an output port index to an input port index.

        ``Feedback(circuit, out_index, in_index)``

    :param circuit: The circuit that undergoes self-feedback
    :type circuit: Circuit
    :param out_index: The output port index.
    :type out_index: int
    :param in_index: The input port index.
    :type in_index: int
    """
    delegate_to_method = (Concatenation, SLH, CPermutation)
    signature = Circuit, int, int

    _rules = []

    @property
    def operand(self):
        """
        The circuit that undergoes feedback
        :rtype: Circuit
        """
        return self._operands[0]

    @property
    def out_in_pair(self):
        """
        Zero-based feedback port indices (out_index, in_index)
        :rtype: tuple
        """
        return self._operands[1:]

    @property
    def _cdim(self):
        return self.operand.cdim - 1

    #noinspection PyUnresolvedReferences
    @classmethod
    def create(cls, circuit, out_index, in_index):
        """
        See :py:class:Feedback: documentation.
        """
        if not isinstance(circuit, Circuit):
            raise ValueError()

        n = circuit.cdim
        if not n:
            raise ValueError()

        if n == 1:
            raise ValueError()

        if isinstance(circuit, cls.delegate_to_method):
            return circuit._feedback(out_index, in_index)

        return super(Feedback, cls).create(circuit, out_index, in_index)


    def _toSLH(self):
        return self.operand.toSLH().feedback(*self.out_in_pair)

    def _toABCD(self):
        # TODO implement Feedback._toABCD()
        raise NotImplementedError(self.__class__)

    def _creduce(self):
        return self.operand.creduce().feedback(*self.out_in_pair)


#    def substitute(self, var_map):
#        op = substitute(self.operand, var_map)
#        return op.feedback(*self.out_in_pair)

    def __str__(self):
        if self.out_in_pair == (self.operand.cdim - 1, self.operand.cdim - 1):
            return "FB(%s)" % self.operand
        o, i = self.out_in_pair
        return "FB(%s, %d, %d)" % (self.operand, o, i)

    def _tex(self):
        o, i = self.out_in_pair
        if self.out_in_pair == (self.cdim -1, self.cdim-1):
            return "\left\lfloor%s\\right\\rfloor" % tex(self.operand)
        return "\left\lfloor%s\\right\\rfloor_{%d\\to%d}" % (tex(self.operand), o, i)


    def _series_inverse(self):
        return Feedback.create(self.operand.series_inverse(), *reversed(self.out_in_pair))

    @property
    def _space(self):
        return self.operand.space


#noinspection PyRedeclaration
def FB(circuit, out_index = None, in_index = None):
    """
    Wrapper for :py:class:Feedback: but with additional default values.

        ``FB(circuit, out_index = None, in_index = None)``

    :param circuit: The circuit that undergoes self-feedback
    :type circuit: Circuit
    :param out_index: The output port index, default = None --> last port
    :type out_index: int
    :param in_index: The input port index, default = None --> last port
    :type in_index: int
    :return: The circuit with applied feedback operation.
    :rtype: Circuit
    """
    if out_index is None:
        out_index = circuit.cdim -1
    if in_index is None:
        in_index = circuit.cdim -1
    return Feedback.create(circuit, out_index, in_index)

@check_signature
class SeriesInverse(Circuit, Operation):
    """
    Symbolic series product inversion operation.

        ``SeriesInverse(circuit)``

    One generally has

        >>> SeriesInverse(circuit) << circuit == cid(circuit.cdim)
            True

    and

        >>> circuit << SeriesInverse(circuit) == cid(circuit.cdim)
            True

    :param Circuit circuit: The circuit system to invert.
    """
    signature = Circuit,

    delegate_to_method = (SeriesProduct, Concatenation, Feedback, SLH, CPermutation, CIdentity.__class__)

    @property
    def operand(self):
        """
        The un-inverted circuit

        :rtype: Circuit
        """
        return self.operands[0]


    @classmethod
    def create(cls, circuit):
        """
        See documentation for :py:class:`SeriesProduct`
        """
        if isinstance(circuit, SeriesInverse):
            return circuit.operand

        elif isinstance(circuit, cls.delegate_to_method):
            #noinspection PyUnresolvedReferences
            return circuit._series_inverse()

        return super(SeriesInverse, cls).create(circuit)

    @property
    def _cdim(self):
        return self.operand.cdim


    def _toSLH(self):
        return self.operand.toSLH().series_inverse()

    def _toABCD(self):
        raise AlgebraError("SeriesInverse not well-defined in ABCD model context")

    def _creduce(self):
        return self.operand.creduce().series_inverse()

    def _substitute(self, var_map):
        return substitute(self, var_map).series_inverse()

    @property
    def _space(self):
        return self.operand.space

    def __str__(self):
        return "[{!s}]^(-1)".format(self.operand)

    def _tex(self):
        return r"\left[ {} \right]^{{\lhd -1}}".format(tex(self.operand))


def _tensor_decompose_series(lhs, rhs):
    """
    Simplification method for lhs << rhs
    Decompose a series product of two reducible circuits with compatible block structures into
    a concatenation of individual series products between subblocks.
    This method raises CannotSimplify when rhs is a CPermutation in order not to conflict with other _rules.
    :type lhs: Circuit
    :type rhs: Circuit
    :return: The combined reducible circuit
    :rtype: Circuit
    :raise: CannotSimplify
    """
    if isinstance(rhs, CPermutation):
        raise CannotSimplify()
    res_struct = get_common_block_structure(lhs.block_structure, rhs.block_structure)
    if len(res_struct) > 1:
        blocks, oblocks = lhs.get_blocks(res_struct), rhs.get_blocks(res_struct)
        parallel_series = [SeriesProduct.create(lb, rb)  for (lb, rb) in izip(blocks, oblocks)]
        return Concatenation.create(*parallel_series)
    raise CannotSimplify()


def _factor_permutation_for_blocks(cperm, rhs):
    """
    Simplification method for cperm << rhs.
    Decompose a series product of a channel permutation and a reducible circuit with appropriate block structure
    by decomposing the permutation into a permutation within each block of rhs and a block permutation and a residual part.
    This allows for achieving something close to a normal form for circuit expression.
    :type cperm: CPermutation
    :type rhs: Circuit
    :rtype: Circuit
    :raise: CannotSimplify
    """
    rbs = rhs.block_structure
    if rhs == cid(rhs.cdim):
        return cperm
    if len(rbs) > 1:
        residual_lhs, transformed_rhs, carried_through_lhs = cperm._factorize_for_rhs(rhs)
        if residual_lhs == cperm:
            raise CannotSimplify()
        return SeriesProduct.create(residual_lhs, transformed_rhs, carried_through_lhs)
    raise CannotSimplify()


def _pull_out_perm_lhs(lhs, rest, out_index, in_index):
    """
    Pull out a permutation from the Feedback of a SeriesProduct with itself.

    :param lhs: The permutation circuit
    :type lhs: CPermutation
    :param rest: The other SeriesProduct operands
    :type rest: OperandsTuple
    :param out_index: The feedback output port index
    :type out_index: int
    :param in_index: The feedback input port index
    :type in_index: int
    :return: The simplified circuit
    :rtype: Circuit
    """
    out_inv , lhs_red = lhs._factor_lhs(out_index)
    return lhs_red << Feedback.create(SeriesProduct.create(*rest), out_inv, in_index)

def _pull_out_unaffected_blocks_lhs(lhs, rest, out_index, in_index):
    """
    In a self-Feedback of a series product, where the left-most operand is reducible,
    pull all non-trivial blocks outside of the feedback.

   :param lhs: The reducible circuit
   :type lhs: Circuit
   :param rest: The other SeriesProduct operands
   :type rest: OperandsTuple
   :param out_index: The feedback output port index
   :type out_index: int
   :param in_index: The feedback input port index
   :type in_index: int
   :return: The simplified circuit
   :rtype: Circuit
   """

    _, block_index = lhs.index_in_block(out_index)

    bs = lhs.block_structure

    nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
    before, block, after = lhs.get_blocks((nbefore, nblock, nafter))

    if before != cid(nbefore) or after != cid(nafter):
        outer_lhs = before + cid(nblock - 1) + after
        inner_lhs = cid(nbefore) + block + cid(nafter)
        return outer_lhs << Feedback.create(SeriesProduct.create(inner_lhs, *rest), out_index, in_index)
    elif block == cid(nblock):
        outer_lhs = before + cid(nblock - 1) + after
        return outer_lhs << Feedback.create(SeriesProduct.create(*rest), out_index, in_index)
    raise CannotSimplify()


#noinspection PyDocstring
def _pull_out_perm_rhs(rest, rhs, out_index, in_index):
    """
    Similar to :py:func:_pull_out_perm_lhs: but on the RHS of a series product self-feedback.
    """
    in_im, rhs_red = rhs._factor_rhs(in_index)
    return Feedback.create(SeriesProduct.create(*rest), out_index, in_im) << rhs_red

def _pull_out_unaffected_blocks_rhs(rest, rhs, out_index, in_index):
    """
    Similar to :py:func:_pull_out_unaffected_blocks_lhs: but on the RHS of a series product self-feedback.
    """
    _, block_index = rhs.index_in_block(in_index)
    bs = rhs.block_structure
    nbefore, nblock, nafter = sum(bs[:block_index]), bs[block_index], sum(bs[block_index + 1:])
    before, block, after = rhs.get_blocks((nbefore, nblock, nafter))
    if before != cid(nbefore) or after != cid(nafter):
        outer_rhs = before + cid(nblock - 1) + after
        inner_rhs = cid(nbefore) + block + cid(nafter)
        return Feedback.create(SeriesProduct.create(*(rest + (inner_rhs,))), out_index, in_index) << outer_rhs
    elif block == cid(nblock):
        outer_rhs = before + cid(nblock - 1) + after
        return Feedback.create(SeriesProduct.create(*rest), out_index, in_index) << outer_rhs
    raise CannotSimplify()


#noinspection PyDocstring
def _series_feedback(series, out_index, in_index):
    """
    Invert a series self-feedback twice to get rid of unnecessary permutations.
    """
    series_s = series.series_inverse().series_inverse()
    if series_s == series:
        raise CannotSimplify()
    return series_s.feedback(out_index, in_index)

A_CPermutation = wc("A", head = CPermutation)
B_CPermutation = wc("B", head = CPermutation)
C_CPermutation = wc("C", head = CPermutation)
D_CPermutation = wc("D", head = CPermutation)

A_Concatenation= wc("A", head = Concatenation)
B_Concatenation = wc("B", head = Concatenation)

A_SeriesProduct = wc("A", head = SeriesProduct)

A_Circuit = wc("A", head = Circuit)
B_Circuit = wc("B", head = Circuit)
C_Circuit = wc("C", head = Circuit)

A__Circuit = wc("A__", head = Circuit)
B__Circuit = wc("B__", head = Circuit)
C__Circuit = wc("C__", head = Circuit)

A_SLH = wc("A", head = SLH)
B_SLH = wc("B", head = SLH)

A_ABCD = wc("A", head = ABCD)
B_ABCD = wc("B", head = ABCD)

j_int = wc("j", head = int)
k_int = wc("k", head = int)

SeriesProduct._binary_rules += [
    ((A_CPermutation, B_CPermutation), lambda A, B: A.series_with_permutation(B)),
    ((A_SLH, B_SLH), lambda A, B: A.series_with_slh(B)),
    ((A_ABCD, B_ABCD), lambda A, B: A.series_with_abcd(B)),
    ((A_Circuit, B_Circuit), lambda A, B: _tensor_decompose_series(A,B)),
    ((A_CPermutation, B_Circuit), lambda A, B: _factor_permutation_for_blocks(A,B))
]

Concatenation._binary_rules += [
    ((A_SLH, B_SLH), lambda A, B: A.concatenate_slh(B)),
    ((A_ABCD, B_ABCD), lambda A, B: A.concatenate_abcd(B)),
    ((A_CPermutation, B_CPermutation), lambda A, B: CPermutation.create(concatenate_permutations(A.operands[0], B.operands[0]))),
    ((A_CPermutation, CIdentity), lambda A: CPermutation.create(concatenate_permutations(A.operands[0], (0,)))),
    ((CIdentity, B_CPermutation ), lambda B: CPermutation.create(concatenate_permutations((0,), B.operands[0]))),
    ((SeriesProduct(A__Circuit, B_CPermutation), SeriesProduct(C__Circuit, D_CPermutation)), lambda A, B, C, D: (SeriesProduct.create(*A) + SeriesProduct.create(*C)) << (B + D)),
    ((SeriesProduct(A__Circuit, B_CPermutation), C_Circuit), lambda A, B, C: (SeriesProduct.create(*A) + C) << (B + cid(C.cdim))),
    ((A_Circuit, SeriesProduct(B__Circuit, C_CPermutation)), lambda A, B, C: (A + SeriesProduct.create(*B)) << (cid(A.cdim) + C)),
]

Feedback._rules += [
    ((A_SeriesProduct, j_int, k_int), lambda A, j, k: _series_feedback(A, j, k)),
    ((SeriesProduct(A_CPermutation, B__Circuit),j_int, k_int ), lambda A, B, j, k: _pull_out_perm_lhs(A,B,j,k)),
    ((SeriesProduct(A_Concatenation, B__Circuit),j_int, k_int ), lambda A, B, j, k: _pull_out_unaffected_blocks_lhs(A,B,j,k)),
    ((SeriesProduct(A__Circuit, B_CPermutation),j_int, k_int ), lambda A, B, j, k: _pull_out_perm_rhs(A,B,j,k)),
    ((SeriesProduct(A__Circuit, B_Concatenation),j_int, k_int ), lambda A, B, j, k: _pull_out_unaffected_blocks_rhs(A,B,j,k)),
]

