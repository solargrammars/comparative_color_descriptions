from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def get_visual_representation(ref, wg, tgt, save_path=None):
    """
    generate a 3d plot comparing the the line
    associated to the learned vector from the comparative
    and  the actual line between ref and tgt
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter([ref[0]], [ref[1]], [ref[2]], label='reference color')
    ax.scatter([tgt[0]], [tgt[1]], [tgt[2]], label='target color')
    ax.plot([ref[0], tgt[0]], [ref[1], tgt[1]], [ref[2], tgt[2]], 
            label='ref-tgt line')

    ax.plot([ref[0], wg[0]] ,[ref[1], wg[1]] ,[ref[2], wg[2]], 
            label='generated comparative direction')

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
    

def get_color_vis(l):
    """
    receives a RGB tuple and generate a html block with the
    associated color as background 
    to be used in jupyter for visualization
    """
    hex_rep  =  rgb2hex(l[0], l[1], l[2])
    s =  "<div style='float:left;background:"+ hex_rep +";padding:10px;width:300px;'></div>"
    return s

def get_gradient_vis(color_list):

    s = "<div style='width:300px;'>"
    for color in color_list:
        hex_rep  =  rgb2hex(color[0], color[1], color[2])

        s = s + "<div style='float:left;background:"+ hex_rep +";padding:5px;width:300px;'></div>"
     
    s= s + "</div>"

    return s


def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))
