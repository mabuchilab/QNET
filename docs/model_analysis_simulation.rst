================================
Symbolic Analysis and Simulation
================================

.. _symbolic_analysis:

Symbolic Analysis of the Pseudo NAND gate and the Pseudo NAND SR-Latch
======================================================================


Pseudo NAND gate
****************

In[1]:

.. code:: python

    from qnet.algebra.circuit_algebra import *

In[2]:

.. code:: python

    from qnet.circuit_components import pseudo_nand_cc as nand
    
    # real parameters
    kappa = symbols('kappa', positive = True)
    Delta, chi, phi, theta = symbols('Delta, chi, phi, theta', real = True)
    
    # complex parameters
    A, B, beta = symbols('A, B, beta')
    
    N = nand.PseudoNAND('N', kappa=kappa, Delta=Delta, chi=chi, phi=phi, theta=theta, beta=beta)
    N

Out[2]:

.. math::

    {\rm N}

Circuit Analysis of Pseudo NAND gate
------------------------------------

In[3]:

.. code:: python

    N.creduce()

Out[3]:

.. math::

    \left(\mathbf{1}_{1} \boxplus (\left(\mathbf{1}_{1} \boxplus (\left({\rm N.P} \boxplus \mathbf{1}_{1}\right) \lhd {{\rm N.BS2}} \lhd \left({W(\beta)} \boxplus \mathbf{1}_{1}\right))\right) \lhd {\mathbf{P}_\sigma \begin{pmatrix} 0 & 1 & 2 \\ 0 & 2 & 1 \end{pmatrix}} \lhd \left({\rm N.K} \boxplus \mathbf{1}_{1}\right))\right) \lhd \left({\rm N.BS1} \boxplus \mathbf{1}_{2}\right) \lhd {\mathbf{P}_\sigma \begin{pmatrix} 0 & 1 & 2 & 3 \\ 0 & 1 & 3 & 2 \end{pmatrix}}

In[4]:

.. code:: python

    # yields a single block
    N.show()
    
    # decompose into sub components
    N.creduce().show() 

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_00.png

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_01.png

SLH model
---------

In[5]:

.. code:: python

    NSLH = N.coherent_input(A, B, 0, 0).toSLH()
    NSLH

Out[5]:

.. math::

    \left( \begin{pmatrix} \frac{1}{2} \sqrt{2} & - \frac{1}{2} \sqrt{2} & 0 & 0 \\ \frac{1}{2} \sqrt{2} & \frac{1}{2} \sqrt{2} & 0 & 0 \\ 0 & 0 & e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right) & - e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) \\ 0 & 0 & \operatorname{sin}\left(\theta\right) & \operatorname{cos}\left(\theta\right)\end{pmatrix}, \begin{pmatrix} \frac{1}{2} \sqrt{2} A - \frac{1}{2} \sqrt{2} B \\  \left(\frac{1}{2} \sqrt{2} A + \frac{1}{2} \sqrt{2} B\right) +  \sqrt{\kappa} {a_{{{\rm N.K}}}} \\  \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right) -  \sqrt{\kappa} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N.K}}}} \\  \beta \operatorname{sin}\left(\theta\right) +  \sqrt{\kappa} \operatorname{cos}\left(\theta\right) {a_{{{\rm N.K}}}}\end{pmatrix},  \frac{1}{2} \mathbf{\imath} \left( - \left(\frac{1}{2} \sqrt{2} A \sqrt{\kappa} + \frac{1}{2} \sqrt{2} B \sqrt{\kappa}\right) {a_{{{\rm N.K}}}^\dagger} +  \left(\frac{1}{2} \sqrt{2} \sqrt{\kappa} \overline{A} + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \overline{B}\right) {a_{{{\rm N.K}}}}\right) +  \chi {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} {a_{{{\rm N.K}}}} +  \Delta {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} \right)

Heisenberg equation of motion of the mode operator :math:`a`
------------------------------------------------------------

In[6]:

.. code:: python

    s = N.space
    a = Destroy(s)
    a

Out[6]:

.. math::

    {a_{{{\rm N.K}}}}

In[7]:

.. code:: python

    NSLH.symbolic_heisenberg_eom(a).expand().simplify_scalar()

Out[7]:

.. math::

     \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- A - B\right) -  \left(\mathbf{\imath} \Delta + \kappa\right) {a_{{{\rm N.K}}}} -  2 \mathbf{\imath} \chi {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} {a_{{{\rm N.K}}}}

Super operator algebra: The system's liouvillian and a re-derivation of the eom for :math:`a` via the super-operator adjoint of the liouvillian.
------------------------------------------------------------------------------------------------------------------------------------------------

In[8]:

.. code:: python

    LLN = NSLH.symbolic_liouvillian().expand().simplify_scalar()
    LLN

Out[8]:

.. math::

    \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(A + B\right) {\rm spost}\left[{a_{{{\rm N.K}}}^\dagger}\right] + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- \overline{A} - \overline{B}\right) {\rm spost}\left[{a_{{{\rm N.K}}}}\right] + \mathbf{\imath} \chi {\rm spost}\left[{a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} {a_{{{\rm N.K}}}}\right] + \left(\mathbf{\imath} \Delta - \kappa\right) {\rm spost}\left[{a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}}\right] + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- A - B\right) {\rm spre}\left[{a_{{{\rm N.K}}}^\dagger}\right] + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(\overline{A} + \overline{B}\right) {\rm spre}\left[{a_{{{\rm N.K}}}}\right] - \mathbf{\imath} \chi {\rm spre}\left[{a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} {a_{{{\rm N.K}}}}\right] - \left(\mathbf{\imath} \Delta + \kappa\right) {\rm spre}\left[{a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}}\right] + 2 \kappa {\rm spre}\left[{a_{{{\rm N.K}}}}\right] {\rm spost}\left[{a_{{{\rm N.K}}}^\dagger}\right]

In[9]:

.. code:: python

    (LLN.superadjoint() * a).expand().simplify_scalar()

Out[9]:

.. math::

     \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- A - B\right) -  \left(\mathbf{\imath} \Delta + \kappa\right) {a_{{{\rm N.K}}}} -  2 \mathbf{\imath} \chi {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} {a_{{{\rm N.K}}}}


A full Pseudo-NAND SR-Latch
***************************

In[10]:

.. code:: python

    N1 = nand.PseudoNAND('N_1', kappa=kappa, Delta=Delta, chi=chi, phi=phi, theta=theta, beta=beta)
    N2 = nand.PseudoNAND('N_2', kappa=kappa, Delta=Delta, chi=chi, phi=phi, theta=theta, beta=beta)
    
    # NAND gates in mutual feedback configuration
    NL = (N1 + N2).feedback(2, 4).feedback(5, 0).coherent_input(A, 0, 0, B, 0, 0)
    NL

Out[10]:

.. math::

    {\left\lfloor\left(\mathbf{1}_{3} \boxplus {\rm N_2}\right) \lhd \left(({\mathbf{P}_\sigma \begin{pmatrix} 0 & 1 & 2 & 3 \\ 0 & 1 & 3 & 2 \end{pmatrix}} \lhd {{\rm N_1}}) \boxplus \mathbf{1}_{3}\right)\right\rfloor_{5\to0}} \lhd \left({W(A)} \boxplus \mathbf{1}_{2} \boxplus {W(B)} \boxplus \mathbf{1}_{2}\right)

The circuit algebra simplification rules have already eliminated one of
the two feedback operations in favor or a series product.

In[11]:

.. code:: python

    NL.show()
    NL.creduce().show()
    NL.creduce().creduce().show()

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_02.png

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_03.png

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_04.png

SLH model
---------

In[12]:

.. code:: python

    NLSLH = NL.toSLH().expand().simplify_scalar()
    NLSLH

Out[12]:

.. math::

    \left( \begin{pmatrix} - \frac{1}{2} \sqrt{2} & 0 & 0 & 0 & \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right) & - \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) \\ \frac{1}{2} \sqrt{2} & 0 & 0 & 0 & \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right) & - \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) \\ 0 & \operatorname{sin}\left(\theta\right) & \operatorname{cos}\left(\theta\right) & 0 & 0 & 0 \\ 0 & \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right) & - \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) & - \frac{1}{2} \sqrt{2} & 0 & 0 \\ 0 & \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right) & - \frac{1}{2} \sqrt{2} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) & \frac{1}{2} \sqrt{2} & 0 & 0 \\ 0 & 0 & 0 & 0 & \operatorname{sin}\left(\theta\right) & \operatorname{cos}\left(\theta\right)\end{pmatrix}, \begin{pmatrix}  \frac{1}{2} \sqrt{2} \left(- A + \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) -  \frac{1}{2} \sqrt{2} \sqrt{\kappa} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N_2.K}}}} \\  \frac{1}{2} \sqrt{2} \left(A + \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) +  \sqrt{\kappa} {a_{{{\rm N_1.K}}}} -  \frac{1}{2} \sqrt{2} \sqrt{\kappa} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N_2.K}}}} \\  \beta \operatorname{sin}\left(\theta\right) +  \sqrt{\kappa} \operatorname{cos}\left(\theta\right) {a_{{{\rm N_1.K}}}} \\  \frac{1}{2} \sqrt{2} \left(- B + \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) -  \frac{1}{2} \sqrt{2} \sqrt{\kappa} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N_1.K}}}} \\  \frac{1}{2} \sqrt{2} \left(B + \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) -  \frac{1}{2} \sqrt{2} \sqrt{\kappa} e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N_1.K}}}} +  \sqrt{\kappa} {a_{{{\rm N_2.K}}}} \\  \beta \operatorname{sin}\left(\theta\right) +  \sqrt{\kappa} \operatorname{cos}\left(\theta\right) {a_{{{\rm N_2.K}}}}\end{pmatrix},  \frac{1}{4} \sqrt{2} \mathbf{\imath} \sqrt{\kappa} \left(- A - \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) {a_{{{\rm N_1.K}}}^\dagger} +  \frac{1}{4} \sqrt{2} \mathbf{\imath} \sqrt{\kappa} \left(- B - \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) {a_{{{\rm N_2.K}}}^\dagger} +  \frac{\sqrt{2} \mathbf{\imath} \sqrt{\kappa} \left(e^{\mathbf{\imath} \phi} \overline{A} + \operatorname{cos}\left(\theta\right) \overline{\beta}\right)}{4 e^{\mathbf{\imath} \phi}} {a_{{{\rm N_1.K}}}} +  \frac{\sqrt{2} \mathbf{\imath} \sqrt{\kappa} \left(e^{\mathbf{\imath} \phi} \overline{B} + \operatorname{cos}\left(\theta\right) \overline{\beta}\right)}{4 e^{\mathbf{\imath} \phi}} {a_{{{\rm N_2.K}}}} +  \chi {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_1.K}}}} +  \Delta {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} +  \frac{\sqrt{2} \mathbf{\imath} \kappa \left(e^{2 \mathbf{\imath} \phi} -1\right) \operatorname{sin}\left(\theta\right)}{4 e^{\mathbf{\imath} \phi}} {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_2.K}}}} +  \chi {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} {a_{{{\rm N_2.K}}}} +  \Delta {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} +  \frac{\sqrt{2} \mathbf{\imath} \kappa \left(e^{2 \mathbf{\imath} \phi} -1\right) \operatorname{sin}\left(\theta\right)}{4 e^{\mathbf{\imath} \phi}} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_2.K}}}^\dagger} \right)

Heisenberg equations of motion for the mode operators
-----------------------------------------------------

In[13]:

.. code:: python

    NL.space

Out[13]:

.. math::

    {{\rm N_1.K}} \otimes {{\rm N_2.K}}

In[14]:

.. code:: python

    s1, s2 = NL.space.operands
    a1 = Destroy(s1)
    a2 = Destroy(s2)

In[15]:

.. code:: python

    da1dt = NLSLH.symbolic_heisenberg_eom(a1).expand().simplify_scalar()
    da1dt

Out[15]:

.. math::

     \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- A - \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) -  \left(\mathbf{\imath} \Delta + \kappa\right) {a_{{{\rm N_1.K}}}} +  \frac{1}{2} \sqrt{2} \kappa e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N_2.K}}}} -  2 \mathbf{\imath} \chi {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_1.K}}}}

In[16]:

.. code:: python

    da2dt = NLSLH.symbolic_heisenberg_eom(a2).expand().simplify_scalar()
    da2dt

Out[16]:

.. math::

     \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- B - \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) +  \frac{1}{2} \sqrt{2} \kappa e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {a_{{{\rm N_1.K}}}} -  \left(\mathbf{\imath} \Delta + \kappa\right) {a_{{{\rm N_2.K}}}} -  2 \mathbf{\imath} \chi {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} {a_{{{\rm N_2.K}}}}

Show Exchange-Symmetry of the Pseudo NAND latch Liouvillian super operator
--------------------------------------------------------------------------

Simultaneously exchanging the degrees of freedom and the coherent input
amplitudes leaves the liouvillian unchanged.

In[17]:

.. code:: python

    C = symbols('C')
    LLNL = NLSLH.symbolic_liouvillian().expand().simplify_scalar()
    LLNL

Out[17]:

.. math::

    \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(A + \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) {\rm spost}\left[{a_{{{\rm N_1.K}}}^\dagger}\right] + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(B + \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) {\rm spost}\left[{a_{{{\rm N_2.K}}}^\dagger}\right] + \frac{\sqrt{2} \sqrt{\kappa} \left(- e^{\mathbf{\imath} \phi} \overline{A} - \operatorname{cos}\left(\theta\right) \overline{\beta}\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spost}\left[{a_{{{\rm N_1.K}}}}\right] + \frac{\sqrt{2} \sqrt{\kappa} \left(- e^{\mathbf{\imath} \phi} \overline{B} - \operatorname{cos}\left(\theta\right) \overline{\beta}\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spost}\left[{a_{{{\rm N_2.K}}}}\right] + \mathbf{\imath} \chi {\rm spost}\left[{a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_1.K}}}}\right] + \left(\mathbf{\imath} \Delta - \kappa\right) {\rm spost}\left[{a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}}\right] + \frac{\sqrt{2} \kappa \operatorname{sin}\left(\theta\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spost}\left[{a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_2.K}}}}\right] + \mathbf{\imath} \chi {\rm spost}\left[{a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} {a_{{{\rm N_2.K}}}}\right] + \left(\mathbf{\imath} \Delta - \kappa\right) {\rm spost}\left[{a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}}\right] + \frac{\sqrt{2} \kappa \operatorname{sin}\left(\theta\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spost}\left[{a_{{{\rm N_1.K}}}} {a_{{{\rm N_2.K}}}^\dagger}\right] + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- A - \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) {\rm spre}\left[{a_{{{\rm N_1.K}}}^\dagger}\right] + \frac{1}{2} \sqrt{2} \sqrt{\kappa} \left(- B - \beta e^{\mathbf{\imath} \phi} \operatorname{cos}\left(\theta\right)\right) {\rm spre}\left[{a_{{{\rm N_2.K}}}^\dagger}\right] + \frac{\sqrt{2} \sqrt{\kappa} \left(e^{\mathbf{\imath} \phi} \overline{A} + \operatorname{cos}\left(\theta\right) \overline{\beta}\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spre}\left[{a_{{{\rm N_1.K}}}}\right] + \frac{\sqrt{2} \sqrt{\kappa} \left(e^{\mathbf{\imath} \phi} \overline{B} + \operatorname{cos}\left(\theta\right) \overline{\beta}\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spre}\left[{a_{{{\rm N_2.K}}}}\right] - \mathbf{\imath} \chi {\rm spre}\left[{a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_1.K}}}}\right] - \left(\mathbf{\imath} \Delta + \kappa\right) {\rm spre}\left[{a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}}\right] + \frac{1}{2} \sqrt{2} \kappa e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {\rm spre}\left[{a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_2.K}}}}\right] - \mathbf{\imath} \chi {\rm spre}\left[{a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} {a_{{{\rm N_2.K}}}}\right] - \left(\mathbf{\imath} \Delta + \kappa\right) {\rm spre}\left[{a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}}\right] + \frac{1}{2} \sqrt{2} \kappa e^{\mathbf{\imath} \phi} \operatorname{sin}\left(\theta\right) {\rm spre}\left[{a_{{{\rm N_1.K}}}} {a_{{{\rm N_2.K}}}^\dagger}\right] + 2 \kappa {\rm spre}\left[{a_{{{\rm N_1.K}}}}\right] {\rm spost}\left[{a_{{{\rm N_1.K}}}^\dagger}\right] + \frac{\sqrt{2} \kappa \left(- e^{2 \mathbf{\imath} \phi} -1\right) \operatorname{sin}\left(\theta\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spre}\left[{a_{{{\rm N_1.K}}}}\right] {\rm spost}\left[{a_{{{\rm N_2.K}}}^\dagger}\right] + \frac{\sqrt{2} \kappa \left(- e^{2 \mathbf{\imath} \phi} -1\right) \operatorname{sin}\left(\theta\right)}{2 e^{\mathbf{\imath} \phi}} {\rm spre}\left[{a_{{{\rm N_2.K}}}}\right] {\rm spost}\left[{a_{{{\rm N_1.K}}}^\dagger}\right] + 2 \kappa {\rm spre}\left[{a_{{{\rm N_2.K}}}}\right] {\rm spost}\left[{a_{{{\rm N_2.K}}}^\dagger}\right]

In[18]:

.. code:: python

    C = symbols('C')
    (LLNL.substitute({A:C}).substitute({B:A}).substitute({C:B}) - LLNL.substitute({s1:s2,s2:s1}).expand().simplify_scalar()).expand().simplify_scalar()


Out[18]:

.. math::

    \hat{0}

.. _numerical_analysis:


Numerical Analysis via QuTiP_
=============================

.. _QuTiP: http://code.google.com/p/qutip/


Input-Output Logic of the Pseudo-NAND Gate
******************************************

In[19]:

.. code:: python

    NSLH.space

Out[19]:

.. math::

    {{\rm N.K}}

In[20]:

.. code:: python

    NSLH.space.dimension = 75

Numerical parameters taken from

**Mabuchi, H. (2011). Nonlinear interferometry approach to photonic
sequential logic. Appl. Phys. Lett. 99, 153103 (2011)**

In[21]:

.. code:: python

    # numerical values for simulation
    
    alpha = 22.6274                              # logical 'one' amplitude
    
    numerical_vals = {
                      beta: -34.289-11.909j,     # bias input for pseudo-nands
                      kappa: 25.,                # Kerr-Cavity mirror couplings
                      Delta: 50.,                # Kerr-Cavity Detuning
                      chi : -50./60.,            # Kerr-Non-Linear coupling coefficient
                      theta: 0.891,              # pseudo-nand beamsplitter mixing angle
                      phi: 2.546,                # pseudo-nand corrective phase
        }

In[22]:

.. code:: python

    NSLHN = NSLH.substitute(numerical_vals)
    NSLHN

Out[22]:

.. math::

    \left( \begin{pmatrix} \frac{1}{2} \sqrt{2} & - \frac{1}{2} \sqrt{2} & 0 & 0 \\ \frac{1}{2} \sqrt{2} & \frac{1}{2} \sqrt{2} & 0 & 0 \\ 0 & 0 & 0.628634640249695 e^{2.546 \mathbf{\imath}} & - 0.777700770912654 e^{2.546 \mathbf{\imath}} \\ 0 & 0 & 0.777700770912654 & 0.628634640249695\end{pmatrix}, \begin{pmatrix} \frac{1}{2} \sqrt{2} A - \frac{1}{2} \sqrt{2} B \\  \left(\frac{1}{2} \sqrt{2} A + \frac{1}{2} \sqrt{2} B\right) +  5.0 {a_{{{\rm N.K}}}} \\  0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}} -  3.88850385456327 e^{2.546 \mathbf{\imath}} {a_{{{\rm N.K}}}} \\  - \left(26.666581733824 + 9.2616384807988 \mathbf{\imath}\right) +  3.14317320124847 {a_{{{\rm N.K}}}}\end{pmatrix},  \frac{1}{2} \mathbf{\imath} \left( - \left(2.5 \sqrt{2} A + 2.5 \sqrt{2} B\right) {a_{{{\rm N.K}}}^\dagger} +  \left(2.5 \sqrt{2} \overline{A} + 2.5 \sqrt{2} \overline{B}\right) {a_{{{\rm N.K}}}}\right) -  0.833333333333333 {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} {a_{{{\rm N.K}}}} +  50.0 {a_{{{\rm N.K}}}^\dagger} {a_{{{\rm N.K}}}} \right)

In[23]:

.. code:: python

    input_configs = [
                (0,0),
                (1, 0),
                (0, 1),
                (1, 1)
              ]

In[24]:

.. code:: python

    Lout = NSLHN.L[2,0]
    Loutqt = Lout.to_qutip()
    times = arange(0, 1., 0.01)
    psi0 = qutip.basis(N.space.dimension, 0)
    datasets = {}
    for ic in input_configs:
        H, Ls = NSLHN.substitute({A: ic[0]*alpha, B: ic[1]*alpha}).HL_to_qutip()
        data = qutip.mcsolve(H, psi0, times, Ls, [Loutqt], ntraj = 1)
        datasets[ic] = data.expect[0]


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00


In[25]:

.. code:: python

    figure(figsize=(10, 8))
    for ic in input_configs:
        plot(times, real(datasets[ic])/alpha, '-', label = str(ic) + ", real")
        plot(times, imag(datasets[ic])/alpha, '--', label = str(ic) + ", imag")
    legend()
    xlabel('Time $t$', size = 20)
    ylabel(r'$\langle L_out \rangle$ in logic level units', size = 20)
    title('Pseudo NAND logic, stochastically simulated time \n dependent output amplitudes for different inputs.', size = 20)

Out[25]:

.. parsed-literal::

    <matplotlib.text.Text at 0x1100b7dd0>

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_05.png


Pseudo NAND latch memory effect
*******************************

In[26]:

.. code:: python

    NLSLH.space

Out[26]:

.. math::

    {{\rm N_1.K}} \otimes {{\rm N_2.K}}

In[27]:

.. code:: python

    s1, s2 = NLSLH.space.operands
    s1.dimension = 75
    s2.dimension = 75
    NLSLH.space.dimension

Out[27]:

.. parsed-literal::

    5625

In[28]:

.. code:: python

    NLSLHN = NLSLH.substitute(numerical_vals)
    NLSLHN

Out[28]:

.. math::

    \left( \begin{pmatrix} - \frac{1}{2} \sqrt{2} & 0 & 0 & 0 & 0.314317320124847 \sqrt{2} e^{2.546 \mathbf{\imath}} & - 0.388850385456327 \sqrt{2} e^{2.546 \mathbf{\imath}} \\ \frac{1}{2} \sqrt{2} & 0 & 0 & 0 & 0.314317320124847 \sqrt{2} e^{2.546 \mathbf{\imath}} & - 0.388850385456327 \sqrt{2} e^{2.546 \mathbf{\imath}} \\ 0 & 0.777700770912654 & 0.628634640249695 & 0 & 0 & 0 \\ 0 & 0.314317320124847 \sqrt{2} e^{2.546 \mathbf{\imath}} & - 0.388850385456327 \sqrt{2} e^{2.546 \mathbf{\imath}} & - \frac{1}{2} \sqrt{2} & 0 & 0 \\ 0 & 0.314317320124847 \sqrt{2} e^{2.546 \mathbf{\imath}} & - 0.388850385456327 \sqrt{2} e^{2.546 \mathbf{\imath}} & \frac{1}{2} \sqrt{2} & 0 & 0 \\ 0 & 0 & 0 & 0 & 0.777700770912654 & 0.628634640249695\end{pmatrix}, \begin{pmatrix}  \frac{1}{2} \sqrt{2} \left(- A + 0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}}\right) -  1.94425192728164 \sqrt{2} e^{2.546 \mathbf{\imath}} {a_{{{\rm N_2.K}}}} \\  \frac{1}{2} \sqrt{2} \left(A + 0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}}\right) +  5.0 {a_{{{\rm N_1.K}}}} -  1.94425192728164 \sqrt{2} e^{2.546 \mathbf{\imath}} {a_{{{\rm N_2.K}}}} \\  - \left(26.666581733824 + 9.2616384807988 \mathbf{\imath}\right) +  3.14317320124847 {a_{{{\rm N_1.K}}}} \\  \frac{1}{2} \sqrt{2} \left(- B + 0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}}\right) -  1.94425192728164 \sqrt{2} e^{2.546 \mathbf{\imath}} {a_{{{\rm N_1.K}}}} \\  \frac{1}{2} \sqrt{2} \left(B + 0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}}\right) -  1.94425192728164 \sqrt{2} e^{2.546 \mathbf{\imath}} {a_{{{\rm N_1.K}}}} +  5.0 {a_{{{\rm N_2.K}}}} \\  - \left(26.666581733824 + 9.2616384807988 \mathbf{\imath}\right) +  3.14317320124847 {a_{{{\rm N_2.K}}}}\end{pmatrix},  1.25 \sqrt{2} \mathbf{\imath} \left(- A - 0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}}\right) {a_{{{\rm N_1.K}}}^\dagger} +  1.25 \sqrt{2} \mathbf{\imath} \left(- B - 0.628634640249695 \left(-34.289 - 11.909 \mathbf{\imath}\right) e^{2.546 \mathbf{\imath}}\right) {a_{{{\rm N_2.K}}}^\dagger} +  1.25 \frac{\sqrt{2} \mathbf{\imath} \left(e^{2.546 \mathbf{\imath}} \overline{A} -21.5552531795218 + 7.48640993073362 \mathbf{\imath}\right)}{e^{2.546 \mathbf{\imath}}} {a_{{{\rm N_1.K}}}} +  1.25 \frac{\sqrt{2} \mathbf{\imath} \left(e^{2.546 \mathbf{\imath}} \overline{B} -21.5552531795218 + 7.48640993073362 \mathbf{\imath}\right)}{e^{2.546 \mathbf{\imath}}} {a_{{{\rm N_2.K}}}} -  0.833333333333333 {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_1.K}}}} +  50.0 {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_1.K}}}} +  4.86062981820409 \frac{\sqrt{2} \mathbf{\imath} \left(-1 + e^{5.092 \mathbf{\imath}}\right)}{e^{2.546 \mathbf{\imath}}} {a_{{{\rm N_1.K}}}^\dagger} {a_{{{\rm N_2.K}}}} -  0.833333333333333 {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} {a_{{{\rm N_2.K}}}} +  50.0 {a_{{{\rm N_2.K}}}^\dagger} {a_{{{\rm N_2.K}}}} +  4.86062981820409 \frac{\sqrt{2} \mathbf{\imath} \left(-1 + e^{5.092 \mathbf{\imath}}\right)}{e^{2.546 \mathbf{\imath}}} {a_{{{\rm N_1.K}}}} {a_{{{\rm N_2.K}}}^\dagger} \right)

In[29]:

.. code:: python

    input_configs = {
                "SET": (1, 0), 
                "RESET": (0, 1),
                "HOLD": (1, 1)
              }
    
    models = {k: NLSLHN.substitute({A:v[0]*alpha, B:v[1]*alpha}).HL_to_qutip() for k, v in input_configs.items()}

In[30]:

.. code:: python

    a1, a2 = Destroy(s1), Destroy(s2)
    observables = [a1.dag()*a1, a2.dag()*a2]
    observables_qt = [o.to_qutip(full_space = NLSLH.space) for o in observables]

In[31]:

.. code:: python

    def model_sequence_single_trajectory(models, durations, initial_state, dt):
        """
        Solve a sequence of constant QuTiP open system models (H_i, [L_1_i, L_2_i, ...])
        via Quantum Monte-Carlo. Each model is valid for a duration deltaT_i and the initial state for
        is given by the previous model's final state.
        The function returns an array with the times and an array with the states at each time.
    
        :param models: Sequence of models given as tuples: (H_j, [L1j,L2j,...])
        :type models: Sequence of tuples
        :param durations: Sequence of times
        :type durations: Sequence of float
        :param initial_state: Overall initial state
        :type initial_state: qutip.Qobj
        :param dt: Sampling interval
        :type dt: float
        :return: times, states
        :rtype: tuple((numpy.ndarray, numpy.ndarray)
        """
        totalT = 0
        totalTimes = array([])
        totalStates = array([])
        current_state = initial_state
        
        for j, (model, deltaT) in enumerate(zip(models, durations)):
            print "Solving step {}/{} of model sequence".format(j + 1, len(models))
            HQobj, LQObjs = model
            times = arange(0, deltaT, dt)
            data = qutip.mcsolve(HQobj, current_state, times, LQObjs, [], ntraj = 1, options = qutip.Odeoptions(gui = False))
    
            # concatenate states
            totalStates = np.hstack((totalStates,data.states.flatten()))
            current_state = data.states.flatten()[-1]
            # concatenate times
            totalTimes = np.hstack((totalTimes, times + totalT))
            totalT += times[-1]
        
        return totalTimes, totalStates

In[32]:

.. code:: python

    durations = [.5, 1., .5, 1.]
    model_sequence = [models[v] for v in ['SET', 'HOLD', 'RESET', 'HOLD']]
    initial_state = qutip.tensor(qutip.basis(s1.dimension, 0), qutip.basis(s2.dimension, 0))

In[33]:

.. code:: python

    times, data = model_sequence_single_trajectory(model_sequence, durations, initial_state, 5e-3)

.. parsed-literal::

    Solving step 1/4 of model sequence


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00
    Solving step 2/4 of model sequence


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00
    Solving step 3/4 of model sequence


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00
    Solving step 4/4 of model sequence


.. parsed-literal::

    100.0%  (1/1)  Est. time remaining: 00:00:00:00


In[34]:

.. code:: python

    datan1 = qutip.expect(observables_qt[0], data)
    datan2 = qutip.expect(observables_qt[1], data)

In[36]:

.. code:: python

    figsize(10,6)
    plot(times, datan1)
    plot(times, datan2)
    for t in cumsum(durations):
        axvline(t, color = "r")
    xlabel("Time $t$", size = 20)
    ylabel("Intra-cavity Photon Numbers", size = 20)
    legend((r"$\langle n_1 \rangle $", r"$\langle n_2 \rangle $"), loc = 'lower right')
    title("SET - HOLD - RESET - HOLD sequence for $\overline{SR}$-latch", size = 20) 

Out[36]:

.. parsed-literal::

    <matplotlib.text.Text at 0x1121d8990>

.. image:: _static/PSeudoNANDAnalysis_files/PseudoNANDAnalysis_fig_06.png

