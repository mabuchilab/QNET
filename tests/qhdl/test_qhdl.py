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
import os
import unittest
from qnet.qhdl.qhdl_parser import QHDLParser
from qnet.algebra.circuit_algebra import *

def parse(qhdl_string):
    p = QHDLParser()
    return p.parse(qhdl_string)

def parse_first_architecture_to_circuit(qhdl_string):
    data = parse(qhdl_string)
    return list(data['architectures'].values()).pop().to_circuit()


qhdl_example_simplest_feedback = \
"""
entity simple_feedback is
	port	(a: in fieldmode; b: out fieldmode);
end simple_feedback;

architecture simple_feedback_s of simple_feedback is

	component Beamsplitter
		port (s1, s2: in fieldmode; s3, s4: out fieldmode);
	end component;

	signal n: fieldmode;

begin
	BS: Beamsplitter
		port map (a,n,b,n);
end simple_feedback_s;
"""

qhdl_example_feedback_2 = \
"""
entity simple_feedback is
	port	(a: in fieldmode; b: out fieldmode);
end simple_feedback;

architecture simple_feedback_s of simple_feedback is

	component Beamsplitter
		port (s1, s2: in fieldmode; s3, s4: out fieldmode);
	end component;

	signal n: fieldmode;

begin
	BS: Beamsplitter
		port map (a,n,n,b);
end simple_feedback_s;
"""






qhdl_example_redheffer = \
"""
entity redheffer is
	port	(a, b: in fieldmode; c, d: out fieldmode);
end redheffer;

architecture redheffer_structure of redheffer is
	
	component Beamsplitter
		port (s1, s2: in fieldmode; s3, s4: out fieldmode);
	end component;

	signal n,t: fieldmode;

begin
	BS1:	Beamsplitter
		port map (a,t,c,n);
	BS2:	Beamsplitter
		port map (n,b,t,d);
end redheffer_structure;
"""

qhdl_example_Open1 = \
"""
entity redheffer is
    port    (a: in fieldmode; c: out fieldmode);
end redheffer;

architecture redheffer_structure of redheffer is
    
    component Beamsplitter
        port (s1, s2: in fieldmode; s3, s4: out fieldmode);
    end component;

    signal n,t: fieldmode;

begin
    BS1:    Beamsplitter
        port map (a,t,c,n);
    BS2:    Beamsplitter
        port map (n,OPEN,t,OPEN);
end redheffer_structure;
"""

qhdl_example_Open2 = \
"""
entity redheffer is
    port    (a: in fieldmode; c: out fieldmode);
end redheffer;

architecture redheffer_structure of redheffer is
    
    component Beamsplitter
        port (a, b: in fieldmode; c, d: out fieldmode);
    end component;

    signal n,t: fieldmode;

begin
    BS1:    Beamsplitter
        port map (a=>a,b=>t,c=>c,d=>n);
    BS2:    Beamsplitter
        port map (a=>n,b=>OPEN,c=>t,d=>OPEN);
end redheffer_structure;
"""

qhdl_example_Open3 = \
"""
entity redheffer is
    port    (a: in fieldmode; c: out fieldmode);
end redheffer;

architecture redheffer_structure of redheffer is
    
    component Beamsplitter
        port (a, b: in fieldmode; c, d: out fieldmode);
    end component;

    signal n,t: fieldmode;

begin
    BS1:    Beamsplitter
        port map (a=>a,b=>t,c=>OPEN,d=>n);
    BS2:    Beamsplitter
        port map (a=>n,b=>OPEN,c=>t,d=>c);
end redheffer_structure;
"""
    

class TestQHDLtoCircuit(unittest.TestCase):
    def testFeedback1(self):
        circuit, symbols, _ = parse_first_architecture_to_circuit(qhdl_example_simplest_feedback)
        self.assertEqual(circuit, FB(symbols['BS']) )
    
    def testFeedback2(self):
        circuit, symbols, _ = parse_first_architecture_to_circuit(qhdl_example_feedback_2)
        self.assertEqual(circuit, symbols['BS'].feedback(0 , 1))
        #print(circuit, all_symbols['BS'].feedback(0 , 1))
    
    def testRedheffer(self):
        
        circuit, symbols, _ = parse_first_architecture_to_circuit(qhdl_example_redheffer)
        BS1, BS2 = symbols['BS1'], symbols['BS2']
        # print(circuit)
        self.assertEqual(circuit, FB(((BS1 + cid(1)) << (cid(1) + BS2 )), 1, 1))
    
    def testOpenPorts1(self):
        circuit, symbols, _ = parse_first_architecture_to_circuit(qhdl_example_Open1)
        BS1, BS2 = symbols['BS1'], symbols['BS2']
        # print(circuit)
        self.assertEqual(circuit, FB(((BS1 + cid(1)) << (cid(1) + BS2 )), 1, 1))
            
    def testOpenPorts2(self):
        circuit, symbols, _ = parse_first_architecture_to_circuit(qhdl_example_Open2)
        BS1, BS2 = symbols['BS1'], symbols['BS2']
        # print(circuit)
        self.assertEqual(circuit, FB(((BS1 + cid(1)) << (cid(1) + BS2 )), 1, 1))

    def testOpenPorts3(self):
        circuit, symbols, _ = parse_first_architecture_to_circuit(qhdl_example_Open3)
        BS1, BS2 = symbols['BS1'], symbols['BS2']
        # print(circuit)
        self.assertEqual(circuit.series_inverse().series_inverse(), P_sigma(1,0) << FB(((BS1 + cid(1)) << (cid(1) + BS2 )), 1, 1))    

if __name__ == '__main__':
    unittest.main()

