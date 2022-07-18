import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Board():

    @staticmethod
    def _parse_qdimacs(filename):
        graph_e2a = nx.Graph()
        graph_a2e = nx.Graph()
        graph_l2c = nx.Graph()
        graph_ref = nx.Graph()
        variables = {}
        v_id = 0
        with open(filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while lines[i].strip().split(" ")[0] == "c":
            i += 1
        header = lines[i].strip().split(" ")
        assert(header[0] == "p")
        n_vars = int(header[2])
        n_clauses = int(header[3])
        i+=1
        prev_q = "e"
        prev_v = -1
        while lines[i].strip().split(" ")[0] == "e" or lines[i].strip().split(" ")[0] == "a":
            quant = lines[i].strip().split(" ")[:-1]
            variables[v_id]=(int(quant[1]),quant[0])
            v_id+=1
            graph_e2a.add_node(int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_e2a.add_node(-int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_a2e.add_node(int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_a2e.add_node(-int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_ref.add_node(int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_ref.add_node(-int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_l2c.add_node(int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_l2c.add_node(-int(quant[1]),n_type='elit' if quant[0] == 'e' else 'alit')
            graph_ref.add_edge(int(quant[1]),-int(quant[1]),e_type='ref')
            graph_ref.add_edge(-int(quant[1]),int(quant[1]),e_type='ref')
            if prev_v!=-1:
                if prev_q == 'e':
                    assert quant[0] == 'a'
                    graph_e2a.add_edge(prev_v,int(quant[1]),e_type='e2a')
                    graph_e2a.add_edge(-prev_v,int(quant[1]),e_type='e2a')
                    graph_e2a.add_edge(prev_v,-int(quant[1]),e_type='e2a')
                    graph_e2a.add_edge(-prev_v,-int(quant[1]),e_type='e2a')
                else:
                    assert prev_q == 'a' and quant[0] == 'e'
                    graph_a2e.add_edge(prev_v,int(quant[1]),e_type='a2e')
                    graph_a2e.add_edge(prev_v,-int(quant[1]),e_type='a2e')
                    graph_a2e.add_edge(-prev_v,int(quant[1]),e_type='a2e')
                    graph_a2e.add_edge(-prev_v,-int(quant[1]),e_type='a2e')
            prev_v = int(quant[1])
            prev_q = quant[0]
            i+=1
        assert len(graph_e2a.nodes())==n_vars*2
        clause_idx = 1
        clause_offset = n_vars
        for line in lines[i:]:
            graph_e2a.add_node(clause_idx+clause_offset,n_type='clause')
            graph_a2e.add_node(clause_idx+clause_offset,n_type='clause')
            graph_l2c.add_node(clause_idx+clause_offset,n_type='clause')
            graph_ref.add_node(clause_idx+clause_offset,n_type='clause')
            for lit in line.strip().split(" ")[:-1]:
                graph_l2c.add_edge(int(lit),clause_idx+clause_offset,e_type='l2c')
            clause_idx+=1

        return graph_e2a, graph_a2e, graph_l2c, graph_ref, n_vars, n_clauses, variables

    def __init__(self, in_obj, copy=False):
        "Set up initial board configuration."
        if copy:
            self.graphs = (in_obj.graphs[0].copy(),in_obj.graphs[1].copy(),in_obj.graphs[2].copy(),in_obj.graphs[3].copy())
            self.game_state = in_obj.game_state
            self.variables = in_obj.variables.copy()
            self.assign_seq = in_obj.assign_seq
        else:
            graph_e2a, graph_a2e, graph_l2c, graph_ref, n_vars, n_clauses, variables = Board._parse_qdimacs(in_obj)
            # Create the empty board array.
            self.graphs = (graph_e2a, graph_a2e, graph_l2c, graph_ref)
            self.game_state = 0        # 0 undergoing, -1 evaluate to False, 1 evaluate to True
            self.variables = variables
            self.assign_seq = ''

    def get_legal_moves(self):
        return [0,1]

    def has_legal_moves(self, player):
        # a player will not have legal move if
        # 1. the game is end, which means the value of the formula has already been confirmed
        # 2. the game is not end, but because of formula reduction, it is again another players move
        # !!! MUST keep current player always have legal move if game is not end 
        return self.game_state==0         

    def execute_move(self, move, player):
        assert move==0 or move==1
        value = False if move==0 else True

        g = self.graphs[0]
        nodelist =  sorted(g.nodes())                   # there is at least one variable in the graph
        assert nodelist!=[] and nodelist[0]<0

        variable = self.variables[sorted(self.variables.keys())[0]][0]
        (graph_e2a, graph_a2e, graph_l2c, graph_ref) = self.graphs
        graph_e2a, graph_a2e, graph_l2c, graph_ref, self.game_state = Board._proceed_graph(graph_e2a, graph_a2e, 
                                                                    graph_l2c, graph_ref, variable, value)
        self.graphs = (graph_e2a, graph_a2e, graph_l2c, graph_ref)
        self.assign_seq += str(move)

        # MUST return the information of next player! The reduction process will decide next player.
        if self.game_state==0:
            # re-count variables
            for v_id in list(self.variables.keys()):
                (v,t) = self.variables[v_id]
                if v not in graph_e2a.nodes:
                    del self.variables[v_id]
            g = self.graphs[0]
            nodelist =  sorted(g.nodes())
            assert nodelist!=[] and nodelist[0]<0        # there is at least one variable in the graph
            active_lit = self.variables[sorted(self.variables.keys())[0]][0]
            active_type = self.variables[sorted(self.variables.keys())[0]][1]

            if active_type=='e':
                return 1
            elif active_type=='a':
                return -1
            else:
                # should never come here!
                assert False
        else:
            # game end, no need to re-consider next player
            return -player



    @staticmethod
    def _proceed_graph2(graph_e2a, graph_a2e, graph_l2c, graph_ref, variable, value):
    
        g1,g2,g3,g4 = graph_e2a.copy(), graph_a2e.copy(), graph_l2c.copy(), graph_ref.copy()
        
        assert variable>0
        assert graph_e2a.nodes[variable]['n_type']!='clause'
        assert graph_a2e.nodes[variable]['n_type']!='clause'
        assert graph_l2c.nodes[variable]['n_type']!='clause'
        assert graph_ref.nodes[variable]['n_type']!='clause'
        
        # remove clause
        if value:
            for n in list(graph_l2c[variable]):
                assert n > 0
                graph_l2c.remove_node(n)
        else:
            for n in list(graph_l2c[-variable]):
                assert n > 0
                graph_l2c.remove_node(n)
        
        # remove head
        graph_l2c.remove_node(variable)
        graph_l2c.remove_node(-variable)

        # check if the formula is satisfied
        if len(graph_l2c.nodes)==0:
            # print("Formula is True, cannot preceed")
            return g1,g2,g3,g4,1

        # check clause orphans result from remove head (i.e. conflicts)
        for node in graph_l2c.nodes:
            if len(graph_l2c.to_undirected()[node])==0 and graph_l2c.nodes[node]['n_type']=='clause':
                # print("Formula is False, cannot preceed")
                return g1,g2,g3,g4,-1
        
        graph_l2c_cp = graph_l2c.copy()
        # remove lit orphans if both x and ~x are orphans
        for node in graph_l2c_cp.nodes:
            if node>0 and graph_l2c_cp.nodes[node]['n_type']!='clause'and len(graph_l2c_cp[node])==0 and len(graph_l2c_cp[-node])==0:
                graph_l2c.remove_node(node)
                graph_l2c.remove_node(-node)

        # check again if the formula is satisfied
        if len(graph_l2c.nodes)==0:
            # print("Formula is True, cannot preceed")
            return g1,g2,g3,g4,1

        nodelist =  sorted(graph_l2c.nodes())
        assert nodelist!=[] and nodelist[0]<0
        
        graph_l2c_cp = graph_l2c.copy()
        # remove clause orphans
        for node in graph_l2c_cp.nodes:
            if len(graph_l2c_cp.to_undirected()[node])==0 and graph_l2c_cp.nodes[node]['n_type']=='clause':
                graph_l2c.remove_node(node)

        nodelist =  sorted(graph_l2c.nodes())
        assert nodelist!=[] and nodelist[0]<0
        
        # consistency
        valid_nodes = set(list(graph_l2c.nodes()))
        
        graph_a2e_cp = graph_a2e.copy()
        graph_e2a_cp = graph_e2a.copy()
        graph_ref_cp = graph_ref.copy()
        
        for v in graph_a2e_cp.nodes:
            if v not in valid_nodes:
                graph_a2e.remove_node(v)
        
        for v in graph_e2a_cp.nodes:
            if v not in valid_nodes:
                graph_e2a.remove_node(v)
        
        for v in graph_ref_cp.nodes:
            if v not in valid_nodes:
                graph_ref.remove_node(v)

        nodelist =  sorted(graph_l2c.nodes())
        assert nodelist!=[] and nodelist[0]<0
                
        return graph_e2a, graph_a2e, graph_l2c, graph_ref, 0

    @staticmethod
    def _proceed_graph(graph_e2a, graph_a2e, graph_l2c, graph_ref, variable, value):
    
        g1,g2,g3,g4 = graph_e2a.copy(), graph_a2e.copy(), graph_l2c.copy(), graph_ref.copy()
        
        assert variable>0
        assert graph_e2a.nodes[variable]['n_type']!='clause'
        assert graph_a2e.nodes[variable]['n_type']!='clause'
        assert graph_l2c.nodes[variable]['n_type']!='clause'
        assert graph_ref.nodes[variable]['n_type']!='clause'
        
        # remove clause
        if value:
            for n in list(graph_l2c[variable]):
                assert n > 0
                graph_l2c.remove_node(n)
        else:
            for n in list(graph_l2c[-variable]):
                assert n > 0
                graph_l2c.remove_node(n)
        
        # remove head
        graph_l2c.remove_node(variable)
        graph_l2c.remove_node(-variable)

        # check if the formula is satisfied
        clause_count = 0
        for node in graph_l2c.nodes:
            if graph_l2c.nodes[node]['n_type']=='clause':
                clause_count += 1
        if clause_count==0:
            # print("Formula is True, cannot preceed")
            return g1,g2,g3,g4,1

        # check clause orphans result from remove head (i.e. conflicts)
        for node in graph_l2c.nodes:
            if len(graph_l2c.to_undirected()[node])==0 and graph_l2c.nodes[node]['n_type']=='clause':
                # print("Formula is False, cannot preceed")
                return g1,g2,g3,g4,-1

        nodelist =  sorted(graph_l2c.nodes())
        assert nodelist!=[] and nodelist[0]<0
        
        # consistency
        valid_nodes = set(list(graph_l2c.nodes()))
        
        graph_a2e_cp = graph_a2e.copy()
        graph_e2a_cp = graph_e2a.copy()
        graph_ref_cp = graph_ref.copy()
        
        for v in graph_a2e_cp.nodes:
            if v not in valid_nodes:
                graph_a2e.remove_node(v)
        
        for v in graph_e2a_cp.nodes:
            if v not in valid_nodes:
                graph_e2a.remove_node(v)
        
        for v in graph_ref_cp.nodes:
            if v not in valid_nodes:
                graph_ref.remove_node(v)
                
        return graph_e2a, graph_a2e, graph_l2c, graph_ref, 0

