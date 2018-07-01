.. _library_structure:

=================
Library Structure
=================


Subpackage Organization
=======================


.. graphviz::

    digraph qnet {
        rankdir="TB";
        graph [pad="0", ranksep="0.25", nodesep="0.25"];
        node [penwidth=0.5, height=0.25, color=black, shape=box, fontsize=10, fontname="Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans"];
        edge [penwidth=0.5, arrowsize=0.5];
        compound=true;

        subgraph cluster_qnet {
            fontname="Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans";
            fontsize=12;
            label="qnet";
            tooltip="qnet";

            subgraph cluster_algebra {
                toolbox          [target="_top"; tooltip="qnet.algebra.toolbox",          href="../API/qnet.algebra.toolbox.html"];
                library          [target="_top"; tooltip="qnet.algebra.library",          href="../API/qnet.algebra.library.html"];
                core             [target="_top"; tooltip="qnet.algebra.core",             href="../API/qnet.algebra.core.html";             width=1.3];
                pattern_matching [target="_top"; tooltip="qnet.algebra.pattern_matching", href="../API/qnet.algebra.pattern_matching.html"; width=1.3];
                toolbox -> library;
                toolbox -> core;
                toolbox -> pattern_matching;
                library -> core;
                core    -> pattern_matching [weight=0, minlen=0];
                library -> pattern_matching;
                href = "../API/qnet.algebra.html"; target="_top";
                label="algebra";
                tooltip="qnet.algebra";
                graph[style=filled; fillcolor="#EEEEEE"];
            }

            convert          [target="_top"; tooltip="qnet.convert",       href="../API/qnet.convert.html"];
            visualization    [target="_top"; tooltip="qnet.visualization", href="../API/qnet.visualization.html"];
            printing         [target="_top"; tooltip="qnet.printing",      href="../API/qnet.printing.html"];
            utils            [target="_top"; tooltip="qnet.utils",         href="../API/qnet.utils.html"; width=1];

            { rank=same; convert[width=0.8]; visualization[width=0.8]; printing[width=0.8]; }
            convert       -> visualization [minlen=3, style=invis];
            visualization -> printing      [minlen=3];
            visualization -> toolbox  [minlen=2, lhead=cluster_algebra];
            printing      -> toolbox  [lhead=cluster_algebra];
            convert       -> toolbox  [lhead=cluster_algebra];

            core             -> utils [ltail=cluster_algebra];
            pattern_matching -> utils [ltail=cluster_algebra];
            convert          -> utils [minlen=6];
            printing         -> utils [minlen=6];
        }

    }


QNET is organized into the sub-packages outlined in the above diagram. Each
package may in turn contain several sub-modules. The arrows indicate which
package imports from which other package.

Every package exports all public symbol from all of its sub-packages/-modules
in a "flat" API. Thus, a user can directly import from the top-level `qnet`
package.

In order from high-level to low-level:

.. autosummary::
    qnet
    qnet.convert
    qnet.visualization
    qnet.printing
    qnet.algebra
    qnet.algebra.toolbox
    qnet.algebra.library
    qnet.algebra.core
    qnet.algebra.pattern_matching
    qnet.utils

See also the full :ref:`modindex`


Class Hierarchy
===============

The following is an inheritance diagram of *all* the classes defined in QNET
(this is best viewed as the full-page SVG):

.. inheritance-diagram:: qnet
   :parts: 1
   :cluster_modules:
