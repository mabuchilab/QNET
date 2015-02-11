# coding=utf-8
#This file is part of QNET.
#
#    QNET is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QNET is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QNET.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2013, Nikolas Tezak
#
###########################################################################


import sys


import pyx
import six
import qnet.misc.parse_circuit_strings as parse_circuit_strings
import qnet.algebra.circuit_algebra as ca
from qnet.circuit_components.component import Component, SubComponent

if six.PY2:
    from itertools import izip as zip
    texrunner = pyx.text.texrunner(mode='latex')
else:
    pyx.text.set(pyx.text.LatexRunner)


HUNIT = +4      # Basic unit for the width of a single Circuit object
                # the positive value corresponds to visualizing the channel
                # 'flow' from left to right

VUNIT = -1.     # Basic unit for the height of a single Circuit object,
                # the negative value makes the effective y-axis point downwards


RHMARGIN = .1   # Relative horizontal margin between gridline and Circuit object
RVMARGIN = .2   # Relative vertical margin between gridline and Circuit object
RPLENGTH = .4   # Relative width of a channel permutation



# helper function
def _curve(x1, y1, x2, y2, hunit = HUNIT, vunit = VUNIT):
    """
    Return a PyX curved path from (x1, y1) to (x2, y2),
    such that the slope at either end is zero.
    """
    ax1, ax2, axm = x1 * hunit, x2 * hunit, (x1 + x2) * hunit / 2
    ay1, ay2 = y1 * vunit, y2 * vunit
    return pyx.path.curve(ax1, ay1, axm, ay1, axm, ay2, ax2, ay2)


def draw_circuit_canvas(circuit, hunit = HUNIT, vunit = VUNIT, rhmargin = RHMARGIN, rvmargin = RVMARGIN, rpermutation_length = RPLENGTH, draw_boxes = True, permutation_arrows = False):
    """
    Generate a PyX graphical representation of a circuit expression object.

    :param circuit: The circuit expression
    :type circuit: ca.Circuit
    :param hunit: The horizontal length unit, default = ``HUNIT``
    :type hunit: float
    :param vunit: The vertical length unit, default = ``VUNIT``
    :type vunit: float
    :param rhmargin: relative horizontal margin, default = ``RHMARGIN``
    :type rhmargin: float
    :param rvmargin: relative vertical margin, default = ``RVMARGIN``
    :type rvmargin: float
    :param rpermutation_length: the relative length of a permutation circuit, default = ``RPLENGTH``
    :type rpermutation_length: float
    :param draw_boxes: Whether to draw indicator boxes to denote subexpressions (Concatenation, SeriesProduct, etc.), default = ``True``
    :type draw_boxes: bool
    :param permutation_arrows: Whether to draw arrows within the permutation visualization, default = ``False``
    :type permutation_arrows: bool
    :return: A PyX canvas object that can be further manipulated or printed to an output image.
    :rtype: pyx.canvas.canvas
    """

    if isinstance(circuit, str):
        try:
            parsed = parse_circuit_strings.parse_circuit_strings(circuit)
            if not isinstance(parsed, ca.Circuit):
                if len(parsed) > 1:
                    raise NotImplementedError('Can currently only process a single expression.')
                circuit = parsed[0]
            else:
                circuit = parsed
        except parse_circuit_strings.ParseCircuitStringError as e:
            raise ValueError('Could not parse {}'.format(circuit))


    if not isinstance(circuit, ca.Circuit):
        raise ValueError()
    
    nc = circuit.cdim
    c = pyx.canvas.canvas()
    
    
    if circuit is ca.CIdentity:
        # simply create a line going through
        c.stroke(pyx.path.line(0, vunit/2, hunit, vunit/2))
        return c, (1, 1), (.5,), (.5,)
    
    elif isinstance(circuit, (ca.CircuitSymbol, ca.SeriesInverse, ca.SLH, Component, SubComponent)):
        # draw box
        b = pyx.path.rect(rhmargin * hunit, rvmargin * vunit, hunit - 2 * rhmargin * hunit, nc * vunit - 2 * rvmargin * vunit)
        c.stroke(b)
        
        texstr = "${}$".format(circuit.tex() if not isinstance(circuit, ca.SLH) else r"{{\rm SLH}}_{{{}}}".format(circuit.space.tex()))

        # draw symbol name
        c.text(hunit/2., nc * vunit/2., texstr , [pyx.text.halign.boxcenter, pyx.text.valign.middle])
        
        # draw connectors at half-unit positions
        connector_positions = tuple((.5 + k) for k in range(nc))
        for y in connector_positions:
            c.stroke(pyx.path.line(0, y * vunit, rhmargin * hunit, y * vunit), [pyx.deco.earrow()])
            c.stroke(pyx.path.line(hunit * (1 - rhmargin), y * vunit, hunit, y * vunit))
        
        return c, (1, nc), connector_positions, connector_positions
    
    elif isinstance(circuit, ca.CPermutation):
        permutation = circuit.permutation
        
        connector_positions = tuple((k + 0.5) for k in range(nc))
        target_positions = [connector_positions[permutation[k]] for k in range(nc)]
        
        # draw curves
        for y1, y2 in zip(connector_positions, target_positions):
            if permutation_arrows:
                c.stroke(_curve(0, y1, rpermutation_length, y2, hunit = hunit, vunit = vunit), [pyx.deco.earrow()])
            else:
                c.stroke(_curve(0, y1, rpermutation_length, y2, hunit = hunit, vunit = vunit))
         
        if draw_boxes:
            b = pyx.path.rect(.5* rhmargin * hunit, .5* rvmargin * vunit, rpermutation_length * hunit -  rhmargin * hunit, nc * vunit -  rvmargin * vunit)
            c.stroke(b, [pyx.style.linewidth.thin, pyx.style.linestyle.dashed, pyx.color.rgb.green])
            
        
        return c, (rpermutation_length, nc), connector_positions, connector_positions
    
    elif isinstance(circuit, ca.SeriesProduct):
        assert len(circuit.operands) > 1
        
        # generate graphics of operad subsystems
        sub_graphics = [draw_circuit_canvas(op, hunit = hunit,
                                        vunit = vunit, rhmargin = rhmargin,
                                        rvmargin = rvmargin,
                                        rpermutation_length = rpermutation_length,
                                        draw_boxes = draw_boxes,
                                        permutation_arrows = permutation_arrows) for op in reversed(circuit.operands)]
        
        # set up first one
        previous_csub, previous_dims, previous_c_in, previous_c_out = sub_graphics[0]
        hoffset = 0
        c.insert(previous_csub)
        hoffset += previous_dims[0]
        
        max_height = nc
        
        # this will later become the full series in-port coordinate tuple
        first_c_in = previous_c_in
        
        # now add all other operand subsystems
        for csub, dims, c_in, c_out in sub_graphics[1:]:
            
            max_height = max(dims[1], max_height)
            
            if previous_c_out != c_in: # vertical port locations don't agree, map signals correspondingly
                
                x1 = hoffset
                x2 = hoffset + rpermutation_length
                
                # draw connection curves
                for y1, y2 in zip(previous_c_out, c_in):
                    c.stroke(_curve(x1, y1, x2, y2, hunit = hunit, vunit = vunit))
                
                hoffset += rpermutation_length
            
            previous_c_in, previous_c_out = c_in, c_out
            
            # now insert current system
            c.insert(csub, [pyx.trafo.translate(hunit * hoffset, 0)])
            hoffset += dims[0]
        if draw_boxes:
            b = pyx.path.rect(.5 * rhmargin * hunit, .5 * rvmargin * vunit, hoffset * hunit - 1. * rhmargin * hunit, max_height * vunit -  rvmargin * vunit)
            c.stroke(b, [pyx.style.linewidth.thin, pyx.style.linestyle.dashed, pyx.color.rgb.red])
        
        
        return c, (hoffset, max_height), first_c_in, c_out
    
    elif isinstance(circuit, ca.Concatenation):
        
        voffset = 0
        total_cin, total_cout = (), ()
        widths = [] # stores the component width for each channel(!)
        
        # generate all operand subsystem graphics and stack them vertically
        for op in circuit.operands:
            csub, dims, c_in, c_out = draw_circuit_canvas(op, hunit = hunit,
                                                    vunit = vunit, rhmargin = rhmargin,
                                                    rvmargin = rvmargin,
                                                    rpermutation_length = rpermutation_length,
                                                    draw_boxes = draw_boxes,
                                                    permutation_arrows = permutation_arrows)
            
            # add appropriatly offsets to vertical port coordinates
            total_cin += tuple(y + voffset for y in c_in)
            total_cout += tuple(y + voffset for y in c_out)
            
            
            c.insert(csub, [pyx.trafo.translate(0, vunit * voffset)])
            
            # keep track of width in all channel for this subsystem
            widths += [dims[0]] * op.cdim
            
            voffset += dims[1]
        
        max_width = max(widths)
        
        if max_width > min(widths): # components differ in width => we must extend the narrow component output lines
            
            for x,y in zip(widths, total_cout):
                if x == max_width:
                    continue
                
                
                ax, ax_to = x * hunit, max_width * hunit
                ay = y * vunit
                c.stroke(pyx.path.line(ax, ay, ax_to, ay))
                
        if draw_boxes:
            b = pyx.path.rect(.5 * rhmargin * hunit, .5 * rvmargin * vunit, max_width * hunit - 1. * rhmargin * hunit, voffset * vunit -  rvmargin * vunit)
            c.stroke(b, [pyx.style.linewidth.thin, pyx.style.linestyle.dashed, pyx.color.rgb.blue])
        
        return c, (max_width, voffset), total_cin, total_cout
    
    elif isinstance(circuit, ca.Feedback):
        
        # generate and insert graphics of subsystem
        csub, dims, c_in, c_out = draw_circuit_canvas(circuit.operand, hunit = hunit,
                                                vunit = vunit, rhmargin = rhmargin,
                                                rvmargin = rvmargin,
                                                rpermutation_length = rpermutation_length,
                                                draw_boxes = draw_boxes,
                                                permutation_arrows = permutation_arrows)
        
        c.insert(csub, [pyx.trafo.translate(hunit * .5 * rhmargin, 0)])
        width, height = dims
        
        # create feedback loop
        fb_out, fb_in = circuit.out_in_pair
        out_coords = (width + .5 * rhmargin) * hunit, c_out[fb_out] * vunit
        in_coords = .5 * rhmargin * hunit, c_in[fb_in] * vunit
        upper_y = (height) * vunit
        feedback_line = pyx.path.path(pyx.path.moveto(*out_coords), pyx.path.lineto(out_coords[0], upper_y),
                                        pyx.path.lineto(in_coords[0], upper_y), pyx.path.lineto(*in_coords))
        c.stroke(feedback_line)
        
        # remove feedback port coordinates
        new_c_in = c_in[:fb_in] + c_in[fb_in + 1 :]
        new_c_out = c_out[:fb_out] + c_out[fb_out + 1 :]
        
        # extend port connectors a little bit outward,
        # such that the feedback loop is not at the edge anymore
        for y in new_c_in:
            c.stroke(pyx.path.line(0, y * vunit, .5 * rhmargin * hunit, y * vunit))
        
        for y in new_c_out:
            c.stroke(pyx.path.line((width + .5 * rhmargin) * hunit, y * vunit, (width + rhmargin) * hunit, y * vunit))
        
        return c, (width + rhmargin, height + rhmargin), new_c_in, new_c_out
    
    raise Exception('Visualization not implemented for type %s' % type(circuit))


def draw_circuit(circuit, filename, direction = 'lr',
            hunit = HUNIT, vunit = VUNIT,
            rhmargin = RHMARGIN, rvmargin = RVMARGIN,
            rpermutation_length = RPLENGTH,
            draw_boxes = True,
            permutation_arrows = False):
    """
    Generate a graphic representation of circuit and store them in a file.
    The graphics format is determined from the file extension.

    :param circuit: The circuit expression
    :type circuit: ca.Circuit
    :param filename: A filepath to store the output image under. The file name suffix determines the output graphics format
    :type filename: str
    :param direction: The horizontal direction of laying out series products. One of ``'lr'`` and ``'rl'``. This option overrides a negative value for ``hunit``,
     default = ``'lr'``
    :param hunit: The horizontal length unit, default = ``HUNIT``
    :type hunit: float
    :param vunit: The vertical length unit, default = ``VUNIT``
    :type vunit: float
    :param rhmargin: relative horizontal margin, default = ``RHMARGIN``
    :type rhmargin: float
    :param rvmargin: relative vertical margin, default = ``RVMARGIN``
    :type rvmargin: float
    :param rpermutation_length: the relative length of a permutation circuit, default = ``RPLENGTH``
    :type rpermutation_length: float
    :param draw_boxes: Whether to draw indicator boxes to denote subexpressions (Concatenation, SeriesProduct, etc.), default = ``True``
    :type draw_boxes: bool
    :param permutation_arrows: Whether to draw arrows within the permutation visualization, default = ``False``
    :type permutation_arrows: bool
    :return: ``True`` if printing was successful, ``False`` if not.
    :rtype: bool
    """


    if direction == 'lr':
        hunit = abs(hunit)
    
    elif direction == 'rl':
        hunit = -abs(hunit)
    try:
        c, dims, c_in, c_out = draw_circuit_canvas(circuit, hunit = hunit, vunit = vunit, 
                                            rhmargin = rhmargin, rvmargin = rvmargin, 
                                            rpermutation_length = rpermutation_length, 
                                            draw_boxes = draw_boxes,
                                            permutation_arrows = permutation_arrows)
    except ValueError as e:
        print( ("No graphics returned for circuit {!r}".format(circuit)))
        return False
    if any(filename.endswith(suffix) for suffix in ('.pdf', '.eps', '.ps')):
        c.writetofile(filename)
    elif any(filename.endswith(suffix) for suffix in ('.png', '.jpg')):
        c.writeGSfile(filename)
    return True


#def display_circuit(circuit):
#    """
#    Under Mac OSX, bring up a visualization of a circuit on the screen.
#    """
#    import os, subprocess, tempfile
#
#    tmp_dir = tempfile.gettempdir()
#    i = 0
#    while(os.path.exists(tmp_dir + "/visualize_circuit_%d.pdf" % i)):
#        i += 1
#
#    fname = tmp_dir + "/visualize_circuit_%d.pdf" % i
#    logname = tmp_dir + "/visualize_circuit_%d-python.log" % i
#
#
#    if draw_circuit(circuit, fname):
#        with open(logname, 'w') as logfile:
#            subprocess.call(("qlmanage", "-p", fname), stdout = logfile, stderr = logfile)









##    display_circuit("(P_sigma(4, 0, 2, 3, 5, 1) << FB((((P_sigma(0, 3, 4, 5, 1, 2) << ((P_sigma(3, 1, 0, 2) << NAND2(4)) + cid(2))) + cid(1)) << (cid(2) + (P_sigma(0, 2, 3, 4, 1) << (((cid(2) + P_sigma(1, 0)) << NAND1(4)) + cid(1))))), 5, 4) << P_sigma(2, 3, 4, 0, 1, 5))")
#    def test():
#        # s = """
#        #       P_sigma(1,2,3,0)
#        #       a_nice_name(3)              # test comment
#        #       (a(3) + b(5))
#        #       (c(3) << d(3))
#        #       ((e(3) + cid(1)) << f(4))
#        #       [cid(1) + a(3)]_(1->2)
#        #       """
#        #   for q in filter(None, s.splitlines()):
#        #       display_circuit(q)
#        s2 = "(P_sigma(4, 0, 2, 3, 5, 1) << FB((((P_sigma(3, 4, 0, 5, 1, 2) << (NAND2(4) + cid(2))) + cid(1)) << (cid(2) + ((P_sigma(1, 0) + cid(1) + P_sigma(1, 0)) << (cid(1) + NAND1(4))))), 3, 5) << P_sigma(3, 4, 5, 0, 1, 2))"
#        s3 = "((cid(1) + (P_sigma(0, 2, 3, 4, 1) << (((cid(1) + FB(NAND2(4), 0, 0)) << P_sigma(0, 2, 3, 1) << NAND1(4)) + cid(1)))) << P_sigma(5, 0, 1, 2, 3, 4))"
#
#        # ps2 = parse_circuit_strings.parse_circuit_strings(s2).pop().simplify()
#        # display_circuit(s2)
#        # display_circuit(ps2)
#        # display_circuit(s3)
#        # display_circuit("((cid(2) + ((cid(3) + P_sigma(1, 0)) << ((P_sigma(0, 3, 2, 1) << NAND1(4)) + cid(1)))) << (cid(2) + P_sigma(1, 2, 3, 4, 0)))")
#        # display_circuit("(((P_sigma(0, 3, 4, 5, 1, 2) << ((P_sigma(1, 2, 0, 3) << NAND2(4)) + cid(2))) + cid(1)) << P_sigma(2, 4, 5, 0, 1, 3, 6)) << ((cid(2) + ((cid(3) + P_sigma(1, 0)) << ((P_sigma(0, 3, 2, 1) << NAND1(4)) + cid(1)))) << (cid(2) + P_sigma(1, 2, 3, 4, 0)))")
#        # display_circuit("(((P_sigma(0, 3, 4, 5, 1, 2) << ((P_sigma(1, 2, 0, 3) << NAND2(4)) + cid(2))) + cid(1)) << (cid(1) + ((cid(3) + P_sigma(1, 2, 0)) << (((cid(3) + P_sigma(1, 0)) << ((P_sigma(3, 1, 2, 0) << NAND1(4)) + cid(1))) + cid(1)))) << P_sigma(5, 6, 1, 2, 3, 4, 0))")
#        # display_circuit("((cid(1) + (P_sigma(0, 2, 3, 4, 1) << (((cid(1) + FB(NAND2(4), 0, 0)) << P_sigma(0, 2, 3, 1) << NAND1(4)) + cid(1)))) << P_sigma(5, 0, 1, 2, 3, 4))")
#        # display_circuit("((cid(1) + (P_sigma(0, 2, 3, 4, 1) << (((cid(1) + FB(NAND2(4), 0, 0)) << P_sigma(0, 2, 3, 1) << NAND1(4)) + cid(1)))) << P_sigma(5, 0, 1, 2, 3, 4))")
#        # display_circuit("FB(((BS1(2) + cid(1)) << (cid(1) + (BS2(2) << P_sigma(1, 0)))), 1, 2)")
#        # display_circuit("(P_sigma(4, 0, 2, 3, 5, 1) << FB(((cid(5) + P_sigma(1, 0)) << ((P_sigma(0, 3, 4, 5, 1, 2) << ((P_sigma(3, 1, 0, 2) << NAND2(4)) + cid(2))) + cid(1)) << (cid(1) + ((cid(3) + P_sigma(1, 2, 0)) << (((cid(3) + P_sigma(1, 0)) << ((P_sigma(3, 1, 2, 0) << NAND1(4)) + cid(1))) + cid(1)))) << P_sigma(5, 6, 1, 2, 3, 4, 0))))")
#    if __name__ == '__main__':
#    test()
#    pass
#
