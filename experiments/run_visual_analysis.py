import sys
sys.path.append('../')
from data import TupleGenerator,  get_xkcd_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import itertools
import ipdb
from pprint import pprint 
from sklearn.neighbors import NearestNeighbors

tuple_generator = TupleGenerator()
num_ref_sample= 1000

for ref, comp_tgt in itertools.groupby(   # comp_tgt = comp, tgt
    tuple_generator.train_tuples, lambda x: x[0]):
    redu = TSNE(n_components=2)
    
    cont_mean = []
    
    data_ref = get_xkcd_data([ref+".train"])
    data_ref_points = data_ref[ref][:num_ref_sample]
    mean_ref = list(map(int, np.mean(data_ref_points,axis=0).tolist()))
    
    print("reference color :" , ref, "points selected: ", len(data_ref_points) )

    total_ref = list(zip( data_ref_points, ["ref" for i in data_ref_points ])) 
    total_ref = total_ref + [(  mean_ref , "meanr"  )]
    
    for i, comp_tgt_ex in enumerate(comp_tgt):
        
        # separate data: comparative and  tgt
        comp_name = comp_tgt_ex[1][1]   #"".join(comp_tgt_ex[1]) 
        tgt  = comp_tgt_ex[2]
        
        # get tgt points
        data_tgt = get_xkcd_data([tgt+".train"])
        data_tgt_points = data_tgt[tgt]
        
        # get mean color value        
        cont_temp_mean = list(map(int, np.mean(data_tgt_points,axis=0).tolist()))
        cont_mean.append( (cont_temp_mean, comp_name))
    
    #for i in cont_mean:
    #    print(i)
    
    print("----")
    
    total = []
    total = total_ref  + cont_mean
    
    total_colors, total_labels = zip(*total)
    print("reducing")
    total_colors_2d = redu.fit_transform(total_colors).tolist()
    
    total_consolidated = list(zip ( total_colors,total_colors_2d,  total_labels))
    
    ref_data = [i for i in total_consolidated if i[2] == "ref"]
    mean_ref_data = [i for i in total_consolidated if i[2] == "meanr"]
    mean_data = [i for i in total_consolidated 
                 if i[2] != "ref" and i[2] != "meanr"]
    
    ref_data_colors, ref_data_xy, _ = zip(*ref_data)
    ref_data_x, ref_data_y = zip(*ref_data_xy)
    
    mean_ref_data_colors, mean_ref_data_xy, _ = zip(*mean_ref_data)
    mean_ref_data_x, mean_ref_data_y = zip(*mean_ref_data_xy)
    
    mean_data_colors, mean_data_xy, mean_data_labels = zip(*mean_data)
    mean_data_x, mean_data_y = zip(*mean_data_xy)
    
    #plt.figure(figsize=(10,10))
    #plt.figure(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    
    plt.scatter(ref_data_x, ref_data_y, 
                c = np.asarray(ref_data_colors)/255, 
                alpha=0.5, 
                s= 50)
    plt.scatter(mean_ref_data_x, mean_ref_data_y, 
                c = np.asarray(mean_ref_data_colors)/255)
    plt.annotate("average "+ ref,
                xy=(mean_ref_data_x[0],mean_ref_data_y[0]), 
                xycoords='data',
                xytext=(-50, 50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))
    
    # plot the mean values of the target colors
    for k in range(len(mean_data_labels)):
        plt.scatter(mean_data_x[k], mean_data_y[k], c=np.asarray(
            mean_data_colors[k]).reshape(1,3)/255)
        
        plt.annotate(mean_data_labels[k],
                xy=(mean_data_x[k],mean_data_y[k]), xycoords='data',
                xytext=(-50, 50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))
    
    plt.savefig("output_visual_analysis/visual_analysis_"+ref+".png")
    plt.close()