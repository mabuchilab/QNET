.. _circuit_rules:

Properties and Simplification of Circuit Algebraic Expressions
==============================================================



By observing that we can define for a general system :math:`Q = (\mathbf{S}, \mathbf{L}, H)` its *series inverse* system :math:`Q^{\lhd -1} := (\mathbf{S}^\dagger, - \mathbf{S}^\dagger \mathbf{L}, - H)`

.. math::

    (\mathbf{S}, \mathbf{L}, H) \lhd (\mathbf{S}^\dagger, - \mathbf{S}^\dagger \mathbf{L}, - H) =   (\mathbf{S}^\dagger, - \mathbf{S}^\dagger \mathbf{L}, - H) \lhd (\mathbf{S}, \mathbf{L}, H) = (\mathbb{I}_n, 0, 0) =: {\rm id}_{n},

we see that the series product induces a group structure on the set of :math:`n`-channel circuit components for any :math:`n \ge 1`.
It can easily be verified that the series inverse of the basic operations is calculated as follows

.. math::

    \left(Q_1 \lhd Q_2\right)^{\lhd -1} & = Q_2^{\lhd -1} \lhd Q_1^{\lhd -1} \\
    \left(Q_1 \boxplus Q_2\right)^{\lhd -1} & = Q_1^{\lhd -1} \boxplus Q_2^{\lhd -1} \\
    \left([Q]_{k\to l}\right)^{\lhd -1} & = \left[Q^{\lhd -1}\right]_{l\to k}.

In the following, we denote the number of channels of any given system :math:`Q = (\mathbf{S}, \mathbf{L}, H)` by :math:`{\rm cdim}\;{Q} := n`.
The most obvious expression simplification is the associative expansion of concatenations and series:

.. math::

    (A_1 \lhd A_2) \lhd (B_1 \lhd B_2) & = A_1 \lhd A_2 \lhd B_1 \lhd B_2 \\
    (C_1 \boxplus C_2) \boxplus (D_1 \boxplus D_2) & = C_1 \boxplus C_2 \boxplus D_1 \boxplus D_2

A further interesting property that follows intuitively from the graphical representation (cf.~Fig.~\ref{fig:decomposition_law}) is the following tensor decomposition law

.. math::

    (A \boxplus B) \lhd (C \boxplus D) = (A \lhd C) \boxplus (B \lhd D),

which is valid for :math:`{\rm cdim}\;{A} = {\rm cdim}\;{C}` and :math:`{\rm cdim}\;{B} = {\rm cdim}\;{D}`.

The following figures demonstrate the ambiguity of the circuit algebra:

.. figure:: _static/plots/dist1.png
    :width: 6cm
    
    :math:`(A \boxplus B) \lhd (C \boxplus D)`

.. figure:: _static/plots/dist2.png
    :width: 6cm
    
    :math:`(A \lhd C) \boxplus (B \lhd D)`


Here, a red box marks a series product and a blue box marks a concatenation. The second version expression has the advantage of making more explicit that the overall circuit consists of two channels without direct optical scattering.

It will most often be preferable to use the RHS expression of the tensor decomposition law above as this enables us to understand the flow of optical signals more easily from the algebraic expression.
In [GoughJames09]_ Gough and James denote a system that can be expressed as a concatenation as *reducible*. A system that cannot be further decomposed into concatenated subsystems is accordingly called *irreducible*.
As follows intuitively from a graphical representation any given complex system :math:`Q = (\mathbf{S}, \mathbf{L}, H)` admits a decomposition into :math:`1 \le N \le {\rm cdim}\;{Q}` irreducible subsystems :math:`Q = Q_1 \boxplus Q_2 \boxplus \dots \boxplus Q_N`, where their channel dimensions satisfy :math:`{\rm cdim}\;{Q_j}\ge 1, \, j=1,2, \dots N` and :math:`\sum_{j=1}^N {\rm cdim}\;{Q_j} = {\rm cdim}\;{Q}`. While their individual parameter triplets themselves are not uniquely determined\footnote{Actually the scattering matrices :math:`\{\mathbf{S}_j\}` and the coupling vectors :math:`\{\mathbf{L}_j\}` *are* uniquely determined, but the Hamiltonian parameters :math:`\{H_j\}` must only obey the constraint :math:`\sum_{j=1}^N H_j = H`.}, the sequence of their channel dimensions :math:`({\rm cdim}\;{Q_1}, {\rm cdim}\;{Q_2},\dots {\rm cdim}\;{Q_N}) =: {\rm bls}\;{Q}` clearly is. We denote this tuple as the block structure of :math:`Q`.
We are now able to generalize the decomposition law in the following way:
Given two systems of :math:`n` channels with the same block structure :math:`{\rm bls}\;{A} = {\rm bls}\;{B} = (n_1, ... n_N)`, there exist decompositions of :math:`A` and :math:`B` such that

.. math::

    A \lhd B = (A_1 \lhd B_1) \boxplus \dots \boxplus (A_N \lhd B_N)

with :math:`{\rm cdim}\;{A_j} = {\rm cdim}\;{B_j} = n_j,\, j = 1, \dots N`.
However, even in the case that the two block structures are not equal, there may still exist non-trivial compatible block decompositions that at least allow a partial application of the decomposition law.
Consider the example presented in Figure (block_structures).

.. figure:: _static/plots/blocks1.png
    :width: 6cm

    Series :math:`"(1,2,1) \lhd (2,1,1)"`

.. figure:: _static/plots/blocks2.png
    :width: 6cm

    Optimal decomposition into :math:`(3,1)`



Even in the case of a series between systems with unequal block structures, there often exists a non-trivial common block decomposition that simplifies the overall expression.

Permutation objects
-------------------

The algebraic representation of complex circuits often requires systems that only permute channels without actual scattering. The group of permutation matrices is simply a subgroup of the unitary (operator) matrices. For any permutation matrix :math:`\mathbf{P}`, the system described by :math:`(\mathbf{P},\mathbf{0},0)` represents a pure permutation of the optical fields (ref fig permutation).

.. figure:: _static/plots/permutation.png

    A graphical representation of :math:`\mathbf{P}_\sigma` where :math:`\sigma \equiv (4,1,5,2,3)` in image tuple notation.


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
In the following we will often write :math:`P_{\sigma}` as a shorthand for :math:`(\mathbf{P}_{\sigma}, \mathbf{0},0)`. Thus, our definition ensures that we may simplify any series of permutation systems in the most intuitive way: :math:`P_{\sigma_2} \lhd P_{\sigma_1} = P_{\sigma_2 \circ \sigma_1}`. Obviously the set of permutation systems of :math:`n` channels and the series product are a subgroup of the full system series group of :math:`n` channels. Specifically, it includes the identity :math:`{\rm id}{n} = P_{\sigma_{{\rm id}_n}}`.

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

The simples case is realized when the permutation simply permutes whole blocks intactly

.. figure:: _static/plots/block_permutation1.png
    :width: 6cm

    :math:`P_\sigma \lhd (A_1 \boxplus A_2)`

.. figure:: _static/plots/block_permutation2.png
    :width: 6cm

    :math:`(A_2 \boxplus A_1) \lhd P_\sigma` 

A block permuting series.

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
The following figure illustrates an example of this case.

.. figure:: _static/plots/block_factorization1.png
    :height: 2cm

    :math:`P_\sigma \lhd (A_1 \boxplus A_2)`

.. figure:: _static/plots/block_factorization1a.png
    :height: 2cm

    :math:`P_{\sigma_b} \lhd P_{\sigma_i} \lhd (A_1 \boxplus A_2)`

.. figure:: _static/plots/block_factorization2.png
    :height: 2cm

    :math:`((P_{\sigma_2} \lhd A_2) \boxplus A_1) \lhd P_{\sigma_{\rm b}}`

A block factorizable series.

A permutation :math:`\sigma` block factorizes according to the block structure :math:`\mathbf{n}` iff for all :math:`1 \le j \le N` we have :math:`\max_{0 \le k < n_j}\sigma(o_j + k)  - \min_{0 \le k' < n_j}\sigma(o_j + k') = n_j - 1`, with the block offsets defined as above. In other words, the image of a single block is coherent in the sense that no other numbers from outside the block are mapped into the integer range spanned by the minimal and maximal points in the block's image. The equivalence follows from our previous result and the bijectivity of :math:`\sigma`.

**The general case**

In general there exists no unique way how to split apart the action of a permutation on a block structure. However, it is possible to define a some rules that allow us to "move as much of the permutation" as possible to the RHS of the series. This involves the factorization :math:`\sigma = \sigma_{\rm x} \circ \sigma_{\rm b} \circ \sigma_{\rm i}` defining a specific way of constructing both :math:`\sigma_{\rm b}` and :math:`\sigma_{\rm i}` from :math:`\sigma`. The remainder :math:`\sigma_{\rm x}` can then be calculated through

.. math::

    \sigma_{\rm x} := \sigma \circ \sigma_{\rm i}^{-1} \circ \sigma_{\rm b}^{-1}.

Hence, by construction, :math:`\sigma_{\rm b} \circ \sigma_{\rm i}` factorizes according to :math:`\mathbf{n}` so only :math:`\sigma_{\rm x}` remains on the exterior LHS of the expression.

So what then are the rules according to which we construct the block permuting :math:`\sigma_{\rm b}` and the decomposable :math:`\sigma_{\rm i}`?
We wish to define :math:`\sigma_{\rm i}` such that the remainder :math:`\sigma \circ \sigma_{\rm i}^{-1} = \sigma_{\rm x} \circ \sigma_{\rm b}` does not cross any two signals that are emitted from the same block. Since by construction :math:`\sigma_{\rm b}` only permutes full blocks anyway this means that :math:`\sigma_{\rm x}` also does not cross any two signals emitted from the same block.
This completely determines :math:`\sigma_{\rm i}` and we can therefore calculate :math:`\sigma \circ \sigma_{\rm i}^{-1} = \sigma_{\rm x} \circ \sigma_{\rm b}` as well. To construct :math:`\sigma_{\rm b}` it is sufficient to define an total order relation on the blocks that only depends on the block structure :math:`\mathbf{n}` and on :math:`\sigma \circ \sigma_{\rm i}^{-1}`. We define the order on the blocks such that they are ordered according to their minimal image point under :math:`\sigma`. Since :math:`\sigma \circ \sigma_{\rm i}^{-1}` does not let any block-internal lines cross, we can thus order the blocks according to the order of the images of the first signal :math:`\sigma \circ \sigma_{\rm i}^{-1}(o_j)`. In (ref fig general_factorization) we have illustrated this with an example.

.. figure:: _static/plots/block_factorization_g1.png
    :height: 1.6cm

    :math:`P_\sigma \lhd (A_1 \boxplus A_2)`

.. figure:: _static/plots/block_factorization_g1a.png
    :height: 1.6cm

    :math:`P_{\sigma_{\rm x}} \lhd P_{\sigma_{\rm b}} \lhd P_{\sigma_{\rm i}} \lhd (A_1 \boxplus A_2)`

.. figure:: _static/plots/block_factorization_g2.png
    :height: 1.6cm

    :math:`(P_{\sigma_{\rm x}} \lhd (P_{\sigma_2} \lhd A_2) \boxplus A_1) \lhd P_{\sigma_{\rm b}}`

A general series with a non-factorizable permutation. In the intermediate step we have explicitly separated :math:`\sigma = \sigma_{\rm x} \circ \sigma_{\rm b} \circ \sigma_{\rm i}`.

Finally, it is a whole different question, why we would want move part of a permutation through the concatenated expression in this first place as the expressions usually appear to become more complicated rather than simpler. This is, because we are currently focussing only on single series products between two systems. In a realistic case we have many systems in series and among these there might be quite a few permutations. Here, it would seem advantageous to reduce the total number of permutations within the series by consolidating them where possible: :math:`P_{\sigma_2} \lhd P_{\sigma_1} = P_{\sigma_2 \circ \sigma_1}`. To do this, however, we need to try to move the permutations through the full series and collect them on one side (in our case the RHS) where they can be combined to a single permutation.
Since it is not always possible to move a permutation through a concatenation (as we have seen above), it makes sense to at some point in the simplification process reverse the direction in which we move the permutations and instead collect them on the LHS. Together these two strategies achieve a near perfect permutation simplification.

Feedback of a concatenation
---------------------------

A feedback operation on a concatenation can always be simplified in one of two ways: If the outgoing and incoming feedback ports belong to the same irreducible subblock of the concatenation, then the feedback can be directly applied only to that single block. For an illustrative example see the figures below:


.. figure:: _static/plots/feedback_concatenation_irre_a.png
    :height: 2cm

    :math:`[A_1 \boxplus A_2]_{2 \to 3}`

.. figure:: _static/plots/feedback_concatenation_irre_b.png
    :height: 2cm

    :math:`A_1 \boxplus [A_2]_{1 \to 2}`

Reduction to feedback of subblock.


If, on the other, the outgoing feedback port is on a different subblock than the incoming, the resulting circuit actually does not contain any real feedback and we can find a way to reexpress it algebraically by means of a series product.


.. figure:: _static/plots/feedback_concatenation_re1_a.png
    :width: 4cm

    :math:`[A_1 \boxplus A_2]_{1 \to 3}`

.. figure:: _static/plots/feedback_concatenation_re1_b.png
    :width: 10cm

    :math:`A_2 \lhd W_{2 \gets 1}^{(2)} \lhd (A_2 \boxplus {\rm id}_{1})`


Reduction of feedback to series, first example

.. figure:: _static/plots/feedback_concatenation_re2_a.png
    :width: 4cm

    :math:`[A_1 \boxplus A_2]_{2 \to 1}`

.. figure:: _static/plots/feedback_concatenation_re2_b.png
    :width: 8cm

    :math:`(A_1 \boxplus {\rm id}_{1}) \lhd A_2`

Reduction of feedback to series, second example


To discuss the case in full generality consider the feedback expression :math:`[A \boxplus B]_{k \to l}` with :math:`{\rm cdim}\;{A} = n_A` and  :math:`{\rm cdim}\;{B} = n_B` and where :math:`A` and :math:`B` are not necessarily irreducible.
There are four different cases to consider.

    * :math:`k,l \le n_A`: In this case the simplified expression should be :math:`[A]_{k \to l} \boxplus B`

    * :math:`k,l > n_A`: Similarly as before but now the feedback is restricted to the second operand :math:`A \boxplus [B]_{(k-n_A) \to (l-n_A)}`, cf. Fig. (ref fig fc_irr).

    * :math:`k \le n_A < l`: This corresponds to a situation that is actually a series and can be re-expressed as :math:`({\rm id}{n_A - 1} \boxplus B) \lhd W_{(l-1) \gets k}^{(n)} \lhd (A + {\rm id}{n_B - 1})`, cf. Fig. (ref fig fc_re1).

    * :math:`l \le n_A < k`: Again, this corresponds a series but with a reversed order compared to above :math:`(A + {\rm id}{n_B - 1}) \lhd W_{l \gets (k-1)}^{(n)} \lhd ({\rm id}{n_A - 1} \boxplus B)`, cf. Fig. (ref fig fc_re2).

Feedback of a series
--------------------

There are two important cases to consider for the kind of expression at either end of the series:
A series starting or ending with a permutation system or a series starting or ending with a concatenation.

.. figure:: _static/plots/feedback_series_ca.png
    :height: 1.8cm

    :math:`[A_3 \lhd (A_1 \boxplus A_2)]_{2 \to 1}`

.. figure:: _static/plots/feedback_series_cb.png
    :height: 1.8cm

    :math:`(A_3 \lhd (A_1 \boxplus {\rm id}_{2})) \lhd A_2`

Reduction of series feedback with a concatenation at the RHS

.. figure:: _static/plots/feedback_series_pa.png

    :math:`[A_3 \lhd P_\sigma]_{2 \to 1}`

.. figure:: _static/plots/feedback_series_pb.png

    :math:`[A_3]_{2 \to 3} \lhd P_{\tilde{\sigma}}`

Reduction of series feedback with a permutation at the RHS


    1) :math:`[A \lhd (C \boxplus D)]_{k \to l}`: We define :math:`n_C = {\rm cdim}\;{C}` and :math:`n_A = {\rm cdim}\;{A}`. Without too much loss of generality, let's assume that :math:`l \le n_C` (the other case is quite similar). We can then pull :math:`D` out of the feedback loop:
    :math:`[A \lhd (C \boxplus D)]_{k \to l} \longrightarrow [A \lhd (C \boxplus {\rm id}{n_D})]_{k \to l} \lhd ({\rm id}{n_C - 1} \boxplus D)`.
    Obviously, this operation only makes sense if :math:`D \neq {\rm id}{n_D}`. The case :math:`l>n_C` is quite similar, except that we pull :math:`C` out of the feedback. See Figure (ref fig fs_c) for an example.

    2) We now consider :math:`[(C \boxplus D) \lhd E]_{k \to l}` and we assume :math:`k \le n_C` analogous to above. Provided that :math:`D \neq {\rm id}{n_D}`, we can pull it out of the feedback and get :math:`({\rm id}{n_C -1} \boxplus D) \lhd [(C \boxplus {\rm id}{n_D}) \lhd E]_{k \to l}`.

    3) :math:`[A \lhd P_\sigma]_{k \to l}`: The case of a permutation within a feedback loop is a lot more intuitive to understand graphically (e.g., cf. Figure ref fig fs_p). Here, however we give a thorough derivation of how a permutation can be reduced to one involving one less channel and moved outside of the feedback.
    First, consider the equality :math:`[A \lhd W_{j \gets l}^{(n)}]_{k \to l} = [A]_{k \to j}` which follows from the fact that :math:`W_{j \gets l}^{(n)}` preserves the order of all incoming signals except the :math:`l`-th.
    Now, rewrite

    .. math::

            [A \lhd P_\sigma]_{k \to l} & = [A \lhd P_\sigma \lhd W_{l \gets n}^{(n)} \lhd W_{n \gets l}^{(n)}]_{k \to l} \\
                                        & = [A \lhd P_\sigma \lhd W_{l \gets n}^{(n)} ]_{k \to n} \\
                                        & = [A \lhd W_{\sigma(l) \gets n}^{(n)} \lhd (W_{n \gets \sigma(l)}^{(n)} \lhd P_\sigma \lhd W_{l \gets n}) ]_{k \to n}

    Turning our attention to the bracketed expression within the feedback, we clearly see that it must be a permutation system :math:`P_{\sigma'} = W_{n \gets \sigma(l)}^{(n)} \lhd P_\sigma \lhd W_{l \gets n}^{(n)}` that maps :math:`n \to l \to \sigma(l) \to n`. We can therefore write :math:`\sigma' = \tilde{\sigma} \boxplus \sigma_{{\rm id}_1}` or equivalently :math:`P_{\sigma'} = P_{\tilde{\sigma}} \boxplus {\rm id}{1}` But this means, that the series within the feedback ends with a concatenation and from our above rules we know how to handle this:

    .. math::

            [A \lhd P_\sigma]_{k \to l} & = [A \lhd W_{\sigma(l) \gets n}^{(n)} \lhd (P_{\tilde{\sigma}} \boxplus {\rm id}{1})]_{k \to n} \\
                                        & = [A \lhd W_{\sigma(l) \gets n}^{(n)}]_{k \to n} \lhd P_{\tilde{\sigma}} \\
                                        & = [A]_{k \to \sigma(l)} \lhd P_{\tilde{\sigma}},

    where we know that the reduced permutation is the well-defined restriction to :math:`n-1` elements of :math:`\sigma' = \left(\omega_{n \gets \sigma{l}}^{(n)} \circ \sigma \circ \omega_{l \gets n}^{(n)}\right)`.

    4) The last case is analogous to the previous one and we will only state the results without a derivation:

    .. math::

             [P_\sigma \lhd A]_{k \to l} & = P_{\tilde{\sigma}} \lhd  [A]_{\sigma^{-1}(k) \to l},

    where the reduced permutation is given by the (again well-defined) restriction of :math:`\omega_{n \gets k}^{(n)} \circ \sigma \circ \omega_{\sigma^{-1}(k) \gets n}^{(n)}` to :math:`n-1` elements.




