#!/usr/bin/env python
# encoding: utf-8
"""
parse_qhdl.py

Created by Nikolas Tezak on 2011-03-11.
Copyright (c) 2011 . All rights reserved.
"""

import sys
import getopt


help_message = '''
Run as {executable} -f path/to/file.qhdl [options]
    When no additional option is specified, the program outputs a valid
    GraphViz dot-syntax graph file with the algebraic circuit expressions
    inserted as comments.
    
    The options are:

    --write-lib or -l : parse and install as new circuit library component
    --help or -h : display this message
    --graph-only or -g : do not attempt to parse into a circuit expression, 
                         but generate the GraphViz graph for the circuit
    

    
'''.format(executable = sys.argv[0])

parser = None

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

from qhdl_parser.qparse import QHDLParser


def set_up_parser(debug = False):
    global parser 
    parser = QHDLParser(debug = debug)
    
def keys_dict(dd):
    return dict(((k, v.keys()) for k,v in dd.items()))


def parse_qhdl_file(filename):
    # try:
    e_a_dict = parser.parse_file(filename)
    return keys_dict(e_a_dict)
    # except Exception, err:
        # raise Usage(str(err))

def parse_qhdl(qhdl_string):
    try:
        e_a_dict =  parser.parse(qhdl_string)
        return keys_dict(e_a_dict)
    except Exception, err:
        
        
        raise Usage(str(err))

def graphs_generator():
    archs = parser.architectures
    for name, a in archs.items():
        yield name, a.to_dot_graph(name + "_")


known_names = []

def id_for_name(name):
    try:
        return known_names.index(name) + 1
    except ValueError:
        known_names.append(name)
        return len(known_names)

def circuit_generator():
    archs = parser.architectures
    if len(archs) > 1:
        for ii, (name, a) in enumerate(archs.items()):
            yield name, a.to_circuit(identifier_postfix = "_%d" % ii)
    elif len(archs) == 1:
        name, a = archs.items().pop()
        yield name, a.to_circuit()
        
def write_modules():
    from circuit_components.library import write_component
    entities = parser.entities.keys()
    architectures_by_entity = dict(((e,{}) for e in entities))
    for a_name, a in parser.architectures.items():
        architectures_by_entity[a.entity.identifier][a_name] = a
    for e, e_archs in architectures_by_entity.items():
        yield write_component(parser.entities[e], e_archs)

        

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        if True:
            try:
                opts, args = getopt.getopt(argv[1:], "hgf:dl", ["help", "graph-only", "file=", "debug", "write-lib"])
            except getopt.error, msg:
                raise Usage(msg)
            debug = False
            input_file = False
            graph_only = False
            write_lib = False
            # option processing
            for option, value in opts:
            
            
                if option in ("-d", "--debug"):
                    debug = True
                
                if option in ("-h", "--help"):
                    raise Usage(help_message)
                
                if option in ("-f", "--file"):
                    input_file = value
            
                if option in ("-g", "--graph-only"):
                    graph_only = True
            
                if option in ("-l", "--write-lib"):
                    write_lib = True
                        

            set_up_parser(debug)
        
            if input_file:
                try:
                    p_dict = parse_qhdl_file(input_file)
                except Exception, e:
                    print "An Error occurred while parsing: ", str(e)
            else:
                 print "No input file supplied"
                 raise Usage(help_message)
             
            print "/* " + "-"*37
            for name, lst in p_dict.items():
                print "-"*40
                print "Found %s:" % name
                print "\t" + ", ".join(lst)
                print "-"*40


            if not graph_only:
                print "Creating algebraic Circuit expressions for all architectures..."
                for name, (circuit, circuit_symbols, assignments) in circuit_generator():
                    print "Circuit for entity '%s':" % name
                    print str(circuit)
                    print "-"*40
                    print "Python representation:"
                    print repr(circuit)
                    print "-"*40
                    print "LaTeX expression"
                    print circuit.tex()
                    print "-"*40
        
            if write_lib:
                print "Wrting library file for all entities..."
                for file_name in write_modules():
                    print "Wrote %s" % file_name
                print "-"*40            
            print "Creating graphviz/dot expressions for all architectures"
            print "-"*37+ "*/"
        
            print "// "+ "-"*37            
            for name, g in graphs_generator():
                print "//  Graph for entity '%s':" % name
                print "// "+ "-"*37
                print g
                print "// "+ "-"*37    
    
    except Usage, err:
        print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
        return 2
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
