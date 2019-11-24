import numpy as np
import pandas as pd
from igraph import Graph
import re
import igraph
import math
import random
import matplotlib.pyplot as plt
import pyedflib
import connectivipy as cp
import louvain 
import infomap
from collections import OrderedDict
from IPython.display import display, Image


class Connectivity_Graph:
    """
    A parent class with all the methods and toolbox needed to analyze the brain signals connectvity and format the associate graph.
    
    """
    def __init__(self,method='DTF'):
        """
        Initialize of the class
        Inputs:
            - method: Default DirectTransferFunction for estimating connectivity between signals OR PartialDirectedCoherence
        """
        self.method = method
        self.df = pd.DataFrame()
        self.channels  = None
        self.freq_sample = None
        self.values = None
        self.num_of_channels = None
        self.num_of_samples = None
        self.conn_algorithm = None
        self.p = None
        self.res = None
        self.G = None
        self.density = None
        self.connectivity_matrix = None
        self.binary_adjacency_matrix = None
        self.connectivity_matrix = None
    
    def import_data(self,path,channels = []):
        """
        Imports the data from a given path and creates a dataframe with all the signals during time
        Inputs:
            - path: given path of the .edf file
            - channels: list of the channels we want to extract for the rest of the procedure, if empty we use them all.
        """
        f = pyedflib.EdfReader(path)
        c = f.signals_in_file 
        signals = np.array([f.readSignal(k).tolist() for k in range(c)]).T.reshape(-1,c)
        self.df = pd.DataFrame(data=signals,columns=f.getSignalLabels(),)
        self.channels = list(map(lambda x:x.strip('.'),self.df.columns))
        self.df.columns = self.channels
        if len(channels)>0:
            self.df = self.df[channels]
            self.channels = channels
        self.freq_sample = f.getSampleFrequencies()[0]
        self.values = self.df.T.values
        self.num_of_channels, self.num_of_samples = self.values.shape
        f.close()
    
    def connectivity(self,freq,algorithm='yw',order=None,max_order=10,plot=False,resolution=100,threshold=None,mode=0,significance=[]):
        """
        Computes the connectivity matrix of a graph using a specific connectivity method (DTF or PDC) and MVar model fitting
        Inputs:
            -freq: sample frequency
            -algorithm: default Yule-Walker algorithm
            -order: MVAR model order
            -max_order: Maximum order to compute the best model's order
            -plot: (DEFAULT: FALSE) if TRUE, plotting of the model order respect to minimization of BIC critirion
            -resolution: frequency resolution
            -threshold: (float between 0,1) percentage of density threshold
            -mode: (default: 0 for Directed graph)
            -significance: list with the nodes we want to exclude from our analysis regarding their significance
            
        """
        self.conn_algorithm = algorithm
        if not order:
            #best,crit = cp.Mvar.order_schwartz(self.values,max_order)  ##BIC
            best,crit = cp.Mvar.order_akaike(self.values,max_order)   ##AIC
            if plot:
                plt.plot(1+np.arange(len(crit)), crit,marker='o', linestyle='dashed',markersize=8,markerfacecolor='yellow')
                plt.grid()
                plt.show()
            self.p = best
            print()
            print('best model order p: {}'.format(best))
            print()
        else:
            self.p = order
        data = cp.Data(self.values,chan_names=self.channels)
        data.fit_mvar(self.p, self.conn_algorithm)
        ar, vr = data.mvar_coefficients
        if self.method == 'DTF':
            conn_matrix = cp.conn.DTF()
        else:
            conn_matrix = cp.conn.PDC()
        Adj = conn_matrix.calculate(ar,vr,self.freq_sample,resolution)
        self.res = np.linspace(0,self.freq_sample/2,resolution)  ## matrix with resolution of frequencies
        ## choosing frequency equal to the sample_frequency
        Adj = Adj[np.where(self.res == self.find_closest_freq(freq)),:,:].reshape(self.num_of_channels,self.num_of_channels)
        ###############################################################################
        
        np.fill_diagonal(Adj,0)
        if len(significance) > 0:
            for a,b in significance:
                Adj[a,b] = 0
        ################################################################################
        self.G = Graph.Weighted_Adjacency(Adj.tolist(), mode = mode) # mode=0 is for directed / mode=1 is for indirected graph
    
    
        self.G.vs["label"] = list(map(lambda x: re.sub('\.', '', x), self.channels))
        locations = pd.read_csv("./data/channel_locations.csv")
        coords = {k[0]: (k[1],k[2]) for k in locations.values}
        self.G.vs["coords"] = [coords[k["label"]] for k in self.G.vs]

        A = np.array(self.G.get_adjacency(attribute = "weight").data)
        diag = np.diag(A)
        # set values of diagonal zero to avoid self-loops
        np.fill_diagonal(A,0)
        if threshold:
            while(self.G.density() > threshold):
                arg_min = np.argwhere(A == np.min(A[np.nonzero(A)]))
                i,j = arg_min[0][0],arg_min[0][1]
                self.G.delete_edges([(i,j)])
                A = np.array(self.G.get_adjacency(attribute = "weight").data)
                np.fill_diagonal(A,0)
            np.fill_diagonal(A,diag)
        
        self.density = self.G.density()
        self.connectivity_matrix = A.copy()
        A[A>0] = 1
        self.binary_adjacency_matrix = A
        self.G.vs['degree'] = self.G.degree()
        
    
    def find_closest_freq(self,f):
        """finds the closest frequency between of sample frequency and the resolution matrix"""
        idx = (np.abs(np.array(self.res) - f)).argmin()
        return(self.res[idx])
    
    def show_graph(self,name):
        """plots the network and exports a .png file with a name=name"""
        
        name = name + '.png'
        visual_style = {}
        visual_style["vertex_size"] = 25
        visual_style["vertex_color"] = "white"
        visual_style["vertex_label"] = self.G.vs["label"]
        visual_style["edge_width"] = [math.exp(weight)*0.5 for weight in self.G.es["weight"]]
        visual_style["layout"] = self.G.vs["coords"]
        
        graph = igraph.plot(self.G, bbox=(0, 0, 600, 600), **visual_style)
        graph.save(name)
        display(Image(filename=name))
        
        #return(graph)
    
    
    def significance(self,signf_threshold=None,channels=[],order=None,Nrep=200,alpha=0.05,visual=False,path=None,freq=10,name='significance'):
        """Computes the significance of the nodes using resampling method for the graph created
           Inputs:
               - signf_threshold: limits of significance under which we want to exclude specific nodes
               - channels: list of the channels we want to extract for the rest of the procedure, if empty we use them all.
               - order: order of the MVAR model
               - Nrep: number od resamples - number of repetitions for the resampling algorithm
               - alpha: (default 0.05) p-value - type I error rate (significance level)
               - visual: if TRUE plots the new network
               - path: path of .edf file we want to import the data if visual TRUE
               - freq: sample frequency if visual TRUE
               - name: name of the output plot of the network if visual TRUE
               """
        df = self.df
        if channels:
            df = df[channels]
        self.values = df.T.values
        self.channels = df.columns.values
        self.num_of_channels,self.num_of_samples = self.values.shape
        data = cp.Data(self.values,chan_names=self.channels)
        if order:
            self.p = order
        data.fit_mvar(self.p,self.conn_algorithm)
        if self.method == 'DTF':
            matrix_values = data.conn('dtf')
        else:
            matrix_values = data.conn('pdc')
        self.significance_matrix = data.significance(Nrep=Nrep, alpha=alpha,verbose=False)

        if signf_threshold:
            elim_indices = np.argwhere(self.significance_matrix>signf_threshold)
            try:
                self.import_data(path,channels = df.columns.values)
                self.connectivity(freq=freq,significance=elim_indices,order=self.p,threshold=round(self.density,1))

                if visual:

                        #self.import_data(path,channels = df.columns.values)
                        #self.connectivity(freq=freq,significance=elim_indices,order=self.p,threshold=round(self.density,1))
                        self.show_graph(name)
            except:
                print("Oops! That was not a valid significance threshold number. Try again...")
                    
                    
    def optimal_modularity_community_detection(self,visual=True,name='optimal_modularity'):
        """
        Community detection Function using Louvain algorithm and maximization of modularity.
        Inputs:
            - visual: (Default = True) Visualize the communities computed
            - name: name of the .png exported file
        """
        louvain.set_rng_seed(123456)
        partition = louvain.find_partition(self.G, louvain.ModularityVertexPartition,weights=self.G.es['weight'])
        self.G.vs['community_optimal_modularity'] = partition.membership
        
        print("The estimated number of communities is",len(set(partition.membership)))
        print('\n')
        print("Communities")
        for n in range(0,len(partition)):
            print('Community number', n, '- size:', len(partition[n]))

        #Create a dictionary whith keys as channels (names of our nodes) and values the community they belong
        comm_detect = dict(zip(self.G.vs['label'],self.G.vs['community_optimal_modularity']))
        print()
        print('The communities are:')
        print()
        comms = {}

        for item in comm_detect.items():
            if item[1] not in comms.keys():
                comms[item[1]] = []

            comms[item[1]].append(item[0])
            
        comms = OrderedDict(sorted(comms.items(), key=lambda t:t[0]))

        print(comms.items())
        
        if visual:
            visual_style = {}
            visual_style["vertex_size"] = 25
            #visual_style["vertex_color"] = "white"
            visual_style["vertex_label"] = self.G.vs["label"]
            #visual_style["edge_width"] = [math.exp(weight)*0.5 for weight in self.G.es["weight"]]
            visual_style["edge_width"] = 0.2
            visual_style["layout"] = self.G.vs["coords"]
            pal = igraph.drawing.colors.ClusterColoringPalette(len(set(self.G.vs['community_optimal_modularity'])))
            visual_style["vertex_color"] = pal.get_many(self.G.vs['community_optimal_modularity'])
            self.G.es['arrow_size'] = [0.1 for edge in self.G.es]



            graph = igraph.plot(self.G,bbox=(0, 0, 600, 600), **visual_style)
            graph.save(name + '.png')
            
            return(comms,graph)
        
        return(comms)
        
    def infomap_community_detection(self,visual=True,name='infomap'):
        """
        Community detection Function using Infomap algorithm and minimization of information flow.
        Inputs:
            - visual: (Default = True) Visualize the communities computed
            - name: name of the .png exported file
        """
        random.seed(123456)
        infomapSimple = infomap.Infomap("--two-level --directed")
        weights = self.G.es['weight']
        for k,e in enumerate(self.G.get_edgelist()):
            infomapSimple.addLink(*e,weights[k])


        infomapSimple.run()
        partition = [[] for _ in range(infomapSimple.numTopModules())]
        modules = []
        for node in infomapSimple.iterTree():
            if node.isLeaf():
                #print("{} {}".format(node.physicalId, node.moduleIndex()))
                modules.append(node.moduleIndex())
                partition[node.moduleIndex()].append(node.physicalId)
        self.G.vs['community_infomap'] = modules
        
        print("The estimated number of communities is",len(partition))
        print('\n')
        print("Communities")
        for n in range(0,len(partition)):
            print('Community number', n, '- size:', len(partition[n]))

        #Create a dictionary whith keys as channels (names of our nodes) and values the community they belong
        comm_detect = dict(zip(self.G.vs['label'],self.G.vs['community_infomap']))
        print()
        print('The communities are:')
        print()
        comms = {}

        for item in comm_detect.items():
            if item[1] not in comms.keys():
                comms[item[1]] = []

            comms[item[1]].append(item[0])

        comms = OrderedDict(sorted(comms.items(), key=lambda t:t[0]))

        print(comms.items())
        
        if visual:
            visual_style = {}
            visual_style["vertex_size"] = 25
            #visual_style["vertex_color"] = "white"
            visual_style["vertex_label"] = self.G.vs["label"]
            #visual_style["edge_width"] = [math.exp(weight)*0.5 for weight in self.G.es["weight"]]
            visual_style["edge_width"] = 0.2
            visual_style["layout"] = self.G.vs["coords"]
            pal = igraph.drawing.colors.ClusterColoringPalette(len(set(self.G.vs['community_infomap'])))
            visual_style["vertex_color"] = pal.get_many(self.G.vs['community_infomap'])
            self.G.es['arrow_size'] = [0.1 for edge in self.G.es]

            

            graph = igraph.plot(self.G,bbox=(0, 0, 600, 600), **visual_style)
            graph.save(name + '.png')
            
            return(comms,graph)
        
        return(comms)
    
    
class Graph_Theory_Indices(Connectivity_Graph):
    """
    A class with all the methods and toolbox needed to global and local indices in the graph associated to the brain connectivity.
    This class inherits the constructor and methods of the Connectivity_Graph class
    
    """
    
    def average_path_length(self, weights = False):
        """
        This function computes the Average Path Length in a directed graph in the weighted and unweighted case
        using the equation
        Inputs:
            - weights (boolean): if True considers the graph as weighted, otherwise it considers the graph as unweighted
        Outputs:
            - Average Path Length
        """

        # Initialize average path length, number of nodes, and list of shortest path lengths
        APL = 0
        N = self.G.vcount()
        delta_min_list = []
        
        # weighted case: compute weights list
        if weights == True:
            w = [l for l in self.connectivity_matrix.reshape(1,-1)[0] if l != 0]
            
        # This loop computes the shortest path between every pair of distinct nodes
        for v_i in range(N):
            for v_j in range(N):
                
                if v_i != v_j:
                    
                    # weighted case: compute weighted shortest path length
                    if weights == True:
                        delta_min = self.G.shortest_paths_dijkstra(source=v_i, target=v_j, weights=w, mode="IN")  
                        
                    # uniweighted case: compute unweighted shortest path length
                    if weights == False:
                        delta_min = self.G.shortest_paths_dijkstra(source=v_i, target=v_j, weights=None, mode="IN")
                        
                    delta_min = delta_min[0][0]
                    
                    # add the shortest path length to the list
                    # if the distance is infinite it will be ignored (i.e. set to 0)
                    if delta_min != np.inf:
                        delta_min_list.append(delta_min)
        
        # compute average path lenght
        APL = 1/(N*(N-1))*np.sum(delta_min_list)

        return APL
    
    def global_clustering_coefficient(self, weights = False):
        """
        This function computes the global clustering coefficient in a directed graph in the weighted and unweighted case
        using the equation GCC = 1/N * sum(LCC_i) where LCC_i =  is the local clustering coefficient of the i-th node.
        The computation of LCC depends on the type of graph (weighted or unweighted)
        Inputs:
            - weights (boolean): if True considers the graph as weighted, otherwise it considers the graph as unweighted
        Outputs:
            - Global clustering coefficient
        """
    
        # Initialize global clustering coefficient and number of nodes
        GCC = 0
        N = self.G.vcount()
        
        # weighted case
        if weights == True:
            
            LCC_list = []

            for v in range(N):

                # obtain W, the connectivity matrix, and compute WW
                W = self.connectivity_matrix
                WW = W**(1/3) + W.T**(1/3)

                num = np.dot(np.dot(WW, WW), WW)[v,v]

                # obtain the adjacency matrix and compute its square
                A = np.array([l for l in self.G.get_adjacency()])
                A2 = np.dot(A,A)
                
                lab = list(self.local_degree().keys())[v]
                # obtain total degree and double sided degree of node
                dt = self.local_degree()[lab]
                dd = A2[v,v]
                den = 2*(dt*(dt-1) -  2*dd)

                # compute LCC and add it to the list 
                # (if it is well defined, use the expression, otherwise set it to 0)
                if den != 0:
                    LCC = num/den
                else:
                    LCC = 0
                    
                LCC_list.append(LCC)

            # compute global clustering coefficient
            GCC = np.mean(LCC_list)
            
            return GCC      
        
        # unweighted case
        if weights == False:
        
            # Initialize local clustering coefficients list and get neighbours list for every node
            LCC_list = []
            neighbours = self.G.as_undirected().get_adjlist()

            # for every node
            for v in range(len(neighbours)):

                # compute subgraph involving the node and its neighbours
                G_sub = self.G.subgraph(neighbours[v], implementation="auto")
                # compute adjacency matrix for such subgraph
                G_sub_Adj = np.array([l for l in G_sub.get_adjacency()])
                # compute total number of edges in the subgraph
                # i.e. the number of edges between all the neighbours of the node
                e = np.sum(G_sub_Adj)
                # compute number of neighbours for the current node
                k = len(neighbours[v])
                
                # if the node has at least two neighbours,
                # compute the local clustering coefficient according to the equation
                if k > 1:
                    LCC = e/(k*(k-1))
                    LCC_list.append(LCC)
                # otherwise set the local clustering coefficient to 0 
                else:
                    LCC = 0
                    LCC_list.append(LCC)
            
            # compute global clustering coefficient
            GCC = np.mean(LCC_list)

            return GCC
        
    
    def local_indegree(self, weights = False):
        """
        This function computes the in-degree for all the nodes in a directed graph
        Inputs:
            - No input. The function operates on the self.G graph defined in the class
        Outputs:
            - Dictionary of nodes (keys) and associated in-degrees (values)
        """
        
        if weights == True:
            in_dict = {}
            N = self.G.vcount()
            # for every node, compute the sum of the in-weights
            # in the connectivity matrix
            for v in range(N):
                in_dict[self.G.vs["label"][v]] = round(np.sum(self.connectivity_matrix[:,v]), 3)
        if weights == False:
            in_dict = dict(zip(self.G.vs["label"],self.G.indegree()))
    
        return in_dict
    
    def local_outdegree(self, weights = False):
        """
        This function computes the out-degree for all the nodes in a directed graph
        Inputs:
            - No input. The function operates on the self.G graph defined in the class
        Outputs:
            - Dictionary of nodes (keys) and associated out-degrees (values)
        """
        
        # weighted case
        if weights == True:
            out_dict = {}
            N = self.G.vcount()
            # for every node, compute the sum of the out-weights
            # in the connectivity matrix
            for v in range(N):
                out_dict[self.G.vs["label"][v]] = round(np.sum(self.connectivity_matrix[v,:]), 3)
        # unweighted case
        if weights == False:
            out_dict = dict(zip(self.G.vs["label"],self.G.outdegree()))

        return out_dict
    
    def local_degree(self, weights = False):
        """
        This function computes the degree for all the nodes in a directed graph
        Inputs:
            - No input. The function operates on the self.G graph defined in the class
        Outputs:
            - Dictionary of nodes (keys) and associated degrees (values)
        """
        # weighted case
        if weights == True:
            deg_dict = {}
            N = self.G.vcount()
            # for every node, compute the sum of the in-weights and the out-weights
            # in the connectivity matrix
            for v in range(N):
                deg_dict[self.G.vs["label"][v]] = round(np.sum(self.connectivity_matrix[:,v]) + np.sum(self.connectivity_matrix[v,:]), 3)
        # unweighted case
        if weights == False:
            degree = np.array(self.G.indegree()) + np.array(self.G.outdegree())
            deg_dict = dict(zip(self.G.vs["label"],degree))

        return deg_dict
    
    pass



class Graph_for_motifs(Connectivity_Graph):        
    """
    Instantiate a Connectivity_Graph object given an input 
    (type of analysis, frequence, and density threhsold for the connectivity graph)
    The main reason of this class is to create a proper file 
    For mfinder program in order to find motifs
    Inputs:
       - method: DFT or PDC, default DFT
       - Analysis: 'EO' Open Eyes resting state or 'EC' Closed Eyes resting state,
                    default: 'EO'
       - freq: frequence of interest, default: 20Hz
       - density_threshold: threshold for the connectivity graph. default: 0.05
       
    """
    def __init__(self, method='DFT', Analysis='EO', freq=20, 
                 density_threshold=0.05, folder=''):
        
        # inheriting the class Connectivity Graph
        Connectivity_Graph.__init__(self, method)
        
        open_eyes_path = "./data/S004R01.edf"
        closed_eyes_path = "./data/S004R02.edf"
        
        if Analysis == 'EO':
            self.import_data(open_eyes_path)
            print('loaded EO resting state data')
        elif Analysis == 'EC':
            self.import_data(closed_eyes_path)
            print('loaded EC resting state data')
        else:
            print("Error: Analysis should be 'EO' or 'EC'")

        # create the path for preprocess_for_mfinder
        self.mfinder_path = folder  + Analysis + '_' + str(freq) + 'Hz_' \
                                    + str(density_threshold) + 'dt_Motifs.txt'

        # New graph with motifs
        self.G_motifs = None 
        
    def preprocess_for_mfinder(self):
        """
        Function which creates a .txt file suitable for mfinder program
        representing the binary adjacency matrix (directed/undirected graph)
        see https://www.weizmann.ac.il/mcb/UriAlon/sites/mcb.UriAlon/files/uploads/NetworkMotifsSW/mdraw/user_manual.pdf
        for the manual
        """
        
        cnt = 0 
        
        with open(self.mfinder_path, 'w') as f:
            
            # check if the matrix has true 
            for i in range (self.num_of_channels):
                for j in range(self.num_of_channels):
                    if self.binary_adjacency_matrix[i,j] != 0:
                        
                        f.write("{0:<3}{1:<2} 1\n".format(i,j))
                        #f.write('  '.join(chars) + '\n')
                        cnt +=1

        print("created text file into '{}'" .format(self.mfinder_path))
    

    def _load_motifs_list(self, filepath):
        """
        function used by plot_motifs():
        load the results of mfinder detecting the motif A->B<-C (id:36)
        input:
            - filepath
        output:
            - list of the nodes in the order: [A, C, B]
        """
        
        # creating the list
        lst = []
            
        # opening the file
        with open (filepath, "r") as myfile:
            for line in myfile:
                lst.append(line.strip()) 
        
        # skipping the first 5 lines (mfinder output file description)
        lst = lst[4:]
        # removing the empty lines and creating list of nodes from each line
        lst = [line.split('\t') for line in lst if len(line)!=0]
        
        # transform the nodes in integer for each motif in the list
        lst = [list(map(int, motif)) for motif in lst]
        
        return lst
    
    @staticmethod
    def filter_motifs(channel, motifs_list):
        """
        function used by motifs
        given a list of motifs, filter only the ones which contain 
        the given channel
        input:
        - channel: selected channel
        - motifs_list: list of channel
        """
        newlist = []
        
        for motif in motifs_list:
            
            if len(set(motif).intersection(set([channel]))) != 0:
                newlist.append(motif)
                
        return newlist


    def plot_motifs_36(self, filepath, name, single_channel=None, motifcolor=None, save=False, tipsize=0.1):
        """
        function which plot the graph enlighting all the motifs A->B<-C (id:36)
        
        inputs:
            - filepath: file.txt (output by mfinder) with all the motifs
            - name: name of the graph (needed only in case of saving the image)
            - single_channel: used only if we want to show only motifs 
                              of one particular node
            - motifcolor: enlight with a color the edges, default: None(black)
            - Save: wethere to save the graph (name.png) or not, default: False
            - tipsize: size of the tip of the arrows
        """
        G = self.G.copy()

        G.es["motif_edges"] = [1 for i in G.es]
        
        motifs_list = self._load_motifs_list(filepath)
        
        if single_channel is not None:
            motifs_list = Graph_for_motifs.filter_motifs(single_channel, motifs_list)

        for motif in motifs_list: 
            A = motif[0]
            B = motif[1]
            C = motif[2]

            G.es[G.get_eid(A, C)]["motif_edges"] += 4
            G.es[G.get_eid(B, C)]["motif_edges"] += 4
            
            if motifcolor is not None:
                G.es[G.get_eid(A, C)]['color'] = motifcolor
                G.es[G.get_eid(B, C)]['color'] = motifcolor

        # arrowsize
        visual_style = {}
        G.es["arrow_size"] = [tipsize for edge in G.es]
        visual_style["vertex_size"] = 30
        visual_style["vertex_color"] = "white"
        visual_style["vertex_label"] = G.vs["label"]
        visual_style["edge_width"] = [0.1 * edge for edge in G.es["motif_edges"]]
        

        G.vs["label"] = list(map(lambda x: re.sub('\.', '', x), self.channels))
        locations = pd.read_csv("./data/channel_locations.csv")
        coords = {k[0]: (k[1],k[2]) for k in locations.values}
        G.vs["coords"] = [coords[k["label"]] for k in G.vs]

        visual_style["layout"] = G.vs["coords"]
        graph = igraph.plot(G, bbox=(0, 0, 600, 600), **visual_style)
        
        if save==True:
            graph.save(name + '.png')
        return(graph)
    
    
    def find_motifs(self, channel='Po3'):
        '''
        Find the occurrencies of the Graph G given the channel
        '''
        
        # list of motif according to igraph library 
        # None values are referred to triples which are non motifs 
        # for more information go to 
        # https://igraph.org/python/doc/igraph.GraphBase-class.html#motifs_randesu_no
        motifs = ['None1', 'None2', 'A->B<-C', 'None3', 'A->B->C', 'A<->B<-C', 
                 'A<-B->C', 'A->B<-C, A->C', 'A<-B->C, A<->C', 'A<->B->C',
                 'A<->B<->C', 'A<-B<-C, A->C', 'A->B->C, A<->C', 'A->B<-C, A<->C',
                 'A->B<->C, A<->C', 'A<->B<->C, A<->C']
        
        subgraph = self.G.subgraph(self.G.neighborhood(self.G.vs.select(label = channel)[0].index))
        
        # occurrencies values
        values = subgraph.motifs_randesu(3)

        # zipping the results with the proper list of motifs 
        res = list(zip(motifs, values))

        # filtering out the nans
        res = [x for x in res if not np.isnan(x[1])]
        res = pd.DataFrame(res, columns=['motif', 'occurrencies'])
        res.sort_values("occurrencies", ascending = False, inplace = True)
        
        return res




def average_path_length(G, weights = False):

    APL = 0
    N = G.vcount()
    delta_min_list = []

    if weights == True:
        w = [l for l in np.array([r for r in G.get_adjacency()]).reshape(1,-1)[0] if l != 0]

    for v_i in range(N):
        for v_j in range(N):

            if v_i != v_j:

                if weights == True:
                    delta_min = G.shortest_paths_dijkstra(source=v_i, target=v_j, weights=w, mode="IN")  

                if weights == False:
                    delta_min = G.shortest_paths_dijkstra(source=v_i, target=v_j, weights=None, mode="IN")

                delta_min = delta_min[0][0]

                if delta_min != np.inf:
                    delta_min_list.append(delta_min)

    APL = 1/(N*(N-1))*np.sum(delta_min_list)

    return APL

def global_clustering_coefficient(G, weights = False):

    GCC = 0
    N = G.vcount()

    if weights == True:

        LCC_list = []

        for v in range(N):

            W = np.array([r for r in G.get_adjacency()])
            WW = W**(1/3) + W.T**(1/3)

            num = np.dot(np.dot(WW, WW), WW)[v,v]

            A = np.array([l for l in G.get_adjacency()])
            A2 = np.dot(A,A)

            dt = G.degree()[v]
            dd = A2[v,v]
            den = 2*(dt*(dt-1) -  2*dd)

            if den != 0:
                LCC = num/den
            else:
                LCC = 0

            LCC_list.append(LCC)

        GCC = np.mean(LCC_list)

        return GCC      

    if weights == False:

        neighbours = G.as_undirected().get_adjlist()
        LCC_list = []

        for v in range(len(neighbours)):

            G_sub = G.subgraph(neighbours[v], implementation="auto")
            G_sub_Adj = np.array([l for l in G_sub.get_adjacency()])
            e = np.sum(G_sub_Adj)
            k = len(neighbours[v])

            if k > 1:
                LCC = e/(k*(k-1))
                LCC_list.append(LCC)

        GCC = np.mean(LCC_list)

        return GCC

def top10(local_degree_dict):

    top_dict = {}
    
    local_degree_list = sorted(local_degree_dict.values())[::-1]

    for i in range(len(local_degree_list)):

        for v in local_degree_dict:
            if local_degree_dict[v] == local_degree_list[i] and len(top_dict) < 10:
                top_dict[v] = local_degree_list[i]

    return(top_dict)