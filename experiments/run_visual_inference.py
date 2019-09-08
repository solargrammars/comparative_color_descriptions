import sys
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint 
from numpy import linalg 
from mpl_toolkits import mplot3d
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

sys.path.append('../')
from settings import OUTPUT_PATH
from inference import get_results

def rgb2lab(r,g,b):
    return list(convert_color( 
        sRGBColor(r/255,g/255,b/255), LabColor).get_value_tuple())

def lab2rgb(l,a,b):
    return list(convert_color( 
        LabColor(l,a,b), sRGBColor).get_value_tuple())

def get_visual_representation(ref, wg, tgt, save_path=None, format="RGB"):
    """
    generate a 3d plot comparing the the line
    associated to the learned vector from the comparative
    and  the actual line between ref and tgt
    """

    # wg vector. a bit of care here:
    # there is a corner case if the resulting wg point
    # is out of the boudaries of the RGb space . for the moment
    # im not considering it.
    w_norm = linalg.norm(wg)
    w_dir   = wg / w_norm
    wg_point = ref + int(w_norm) * w_dir
    wg_point = list(map(int, wg_point))
    
    
    ref_color = np.asarray(ref)/255
    tgt_color = np.asarray(tgt)/255 
    wg_color  = np.asarray(wg_point)/255
    
    
    if format == "LAB":
        ref = rgb2lab(ref[0], ref[1], ref[2])
        tgt = rgb2lab(tgt[0], tgt[1], tgt[2])
        wg_point  = rgb2lab(wg_point[0], wg_point[1],  wg_point[2])

        
    #ipdb.set_trace()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    
    # ref point
    ax.scatter([ref[0]], 
               [ref[1]], 
               [ref[2]], 
               label='reference color',
                color=ref_color, s=100)
    # tgt point
    ax.scatter([tgt[0]], 
               [tgt[1]], 
               [tgt[2]], 
               label='target color',
                color=tgt_color, s=100)
    # tr vector
    ax.plot([ref[0], tgt[0]], 
            [ref[1], tgt[1]], 
            [ref[2], tgt[2]], 
            label='tr vector',
            color="black")
    
    # line from ref to the last point of vector wg 
    ax.plot([ref[0], wg_point[0]],
            [ref[1], wg_point[1]],
            [ref[2], wg_point[2]], 
            label='wg vector',
            color="black",
            linestyle='dashed')

    # wg point
    ax.scatter([wg_point[0]], 
               [wg_point[1]], 
               [wg_point[2]], 
                color=wg_color, s=100, marker='^')
    
    if format == "LAB":
        ax.set_xlabel('L')
        ax.set_ylabel('a')
        ax.set_zlabel('b')
    else:
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        
        
    """
    another option is to use a vector form (x,y,z) = (x0,y0,z0) + t(dx,dy,dz)
    and then do something like:
    
    t = np.linspace(1, 10, 100)  
    xg = ref[0] + t*wg[0] 
    yg = ref[1] + t*wg[1] 
    zg = ref[2] + t*wg[2] 
    ax.plot(xg, yg, zg, label='generated line v2')
    """

    plt.legend()
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    
def visualize(comparative):
    sorted_list = sorted(results_dic[comparative], key= lambda x:x[3])
    top = sorted_list[0]
    print(top)
    
    ref = top[0].astype(int)
    tgt = top[1].astype(int)
    w   = top[2].astype(int)
    
    print("ref", ref)
    print("tgt", tgt)
    print("w"  , w)
    
    get_visual_representation(ref, w, tgt, "output_visual_inference/"+test_category+"_vector_visualization_"+comparative+".png")
    
    #https://math.stackexchange.com/a/83419    
    # gradient between ref and tgt
    tr = tgt - ref
    tr_norm =  linalg.norm(tr)
    tr_dir = tr / tr_norm
    tr_points = []
    for j in range(1, int(tr_norm)  ):
        point = ref + j * tr_dir
        point = list(map(int, point))
        if all([i <=255 and i >= 0 for i in point]): 
            tr_points.append(point)
            
    # gradient from ref following wg direction
    w_norm = linalg.norm(w)
    w_dir   = w / w_norm
    w_points = []
    for j in range(1,int(w_norm)):
        point = ref + j * w_dir
        point = list(map(int, point))
        if all([i <=255 and i >= 0 for i in point]): 
            w_points.append(point)
    
    #some numbers 
    print("norm w :", w_norm)
    print("norm tr:", tr_norm)
    
    # plotting
    fig, axs = plt.subplots(2, 1, sharex =True,  figsize=(5,5))
    axs[0].set_yticks([], [])
    axs[1].set_yticks([], [])
    
    
    #plt.subplot(212)
    axs[0].set_title("tr vector:")
    for x in range(len(tr_points)):
        axs[0].axvline(x, color=np.asarray(tr_points[x])/255, linewidth=10)
 
    #plt.subplot(211)
    axs[1].set_title("wg vector:")
    for x in range(len(w_points)):
        axs[1].axvline(x, color=np.asarray(w_points[x])/255, linewidth=10)

    plt.savefig("output_visual_inference/"+test_category+"_color_gradients_"+comparative+".png")
    

    
if __name__ == "__main__":
    
    # choose one test set category: 'seen_pairings', 'unseen_pairings', 'unseen_ref', 'unseen_comp', 'fully_unseen'
    test_category = "fully_unseen"
    
    
    results  = get_results(OUTPUT_PATH+"/1565967650/model.pth", test_category=test_category)  #   1563072627
    results_dic = {}
    for str_tuple, data in results:
        results_dic[str_tuple[2]]= data

    print("available comparatives:")
    pprint(results_dic.keys())
    print("-----------------------")
    
    #ipdb.set_trace()
    
    instances = list(results_dic.keys())

    for i in instances:
        visualize(i)