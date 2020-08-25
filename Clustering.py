import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation
from matplotlib.cbook import get_sample_data

from sklearn.preprocessing import normalize
from sklearn import decomposition
from sklearn.manifold import TSNE 

from auxiliary_functions import get_correlation_matrix, heatmap, annotate_heatmap,annotate_cluster

import imageio


#https://plot.ly/python/
#https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot

dimensionality = 3 #3
model = 'PCA'
n_components = max(dimensionality+3,3)
normalize_input = 0

random_state = np.random.seed()

#civ_data=pd.read_csv('AOE 2 DE Civ Megasheet with grades - New Civ Grades - General.csv', sep=',', index_col=0) 
civ_data=pd.read_csv('AOE 2 DE Civ Megasheet with grades - New Civ Grades - Specific.csv', sep=',', skiprows = 2, index_col=1)
civ_data.drop(columns='Icon',inplace=True)
civ_data.fillna(0,inplace=True)

X = civ_data.to_numpy()
if normalize_input:
    X = normalize(X, axis=0, norm='max')
    
if model=='tSNE':
    model = TSNE(n_components=3, random_state=random_state)

elif model=='PCA':
    model = decomposition.PCA(n_components,random_state=random_state)


X = model.fit_transform(X)

Colorshift = min(X[:,dimensionality])
Colorrange = max(X[:,dimensionality])-Colorshift

fig1,ax1 = plt.subplots(figsize=(25,25))

if dimensionality==2:
    ax1.scatter(X[:,0], X[:,1], s=750, c=tuple((X[:,dimensionality]-Colorshift)/Colorrange), cmap=plt.cm.get_cmap('PiYG',10), marker='o',edgecolors='black')
elif dimensionality==3:
    ax1 = Axes3D(fig1, rect=[0, 0, .95, 1], elev=48, azim=134,facecolor='black')
    ax1.scatter(X[:,0], X[:,1], X[:,2], s=750, c=tuple((X[:,dimensionality]-Colorshift)/Colorrange), cmap=plt.cm.get_cmap('PiYG',10), marker='o',edgecolors='black')
    
for index, row in enumerate(civ_data.iterrows()):
    if dimensionality==2:
        plt.text(X[index,0],X[index,1],civ_data.index[index], size = 32,
          horizontalalignment='center', verticalalignment='top',
          bbox=dict(alpha=0, edgecolor='w', facecolor='w'))    
    elif dimensionality==3:    
        ax1.text3D(X[index,0],X[index,1],X[index,2]+0.1+0.6*(1-normalize_input),civ_data.index[index],
                  horizontalalignment='center', verticalalignment='top', size=18,
                  color='white',
                  bbox=dict(alpha=0, edgecolor='w', facecolor='w'))
        ax1.w_zaxis.set_ticklabels([])
        
    ax1.w_xaxis.set_ticklabels([])
    ax1.w_yaxis.set_ticklabels([])
    
#    with get_sample_data('./CivIcon-'+civ_data.index[index]+'.png') as file:
#        Icon = plt.imread(file, format='png')
#
#    imagebox = AnnotationBbox(OffsetImage(Icon, cmap="binary"), X[index,:3])
#    ax1.add_artist(imagebox)

#     icon = mpimg.imread('CivIcon-'+civ_data.index[index]+'.png')
#     imagebox = OffsetImage(icon, zoom=0.2)
#     ab = AnnotationBbox(imagebox, (0, 0,0))
#     ax.add_artist(ab)

clusters = {
		"Gunpowder gang" : ['Turks','Italians','Spanish','Portuguese'],
        "American bois": ['Mayans','Incas','Aztecs'],
		"Infantry bros" : ['Teutons','Goths','Franks','Celts','Slavs','Ethiopians','Burmese','Aztecs','Japanese','Malay', 'Bulgarians','Slavs','Vikings'],
        "Siege dudemeisters": ['Celts','Slavs','Ethiopians','Mongols'],
		"Camel troupe" : ['Indians','Berbers','Byzantines','Persians','Malians', 'Saracens'],
		"Archer gang" : ['Mayans','Incas','Britons','Chinese','Japanese','Malay','Incas','Ethiopians', 'Vietnamese','Vikings','Koreans','Italians', 'Saracens'],
		"Cav archer enthusiasts" : ['Mongols','Huns','Cumans','Magyars','Tatars'],
		"Horse lovers" :  ['Mongols','Huns','Cumans','Magyars','Tatars', 'Slavs','Bulgarians', 'Lithuanians','Franks','Burmese'],
		"Monk maniacs" :  ['Aztecs','Spanish','Burmese','Lithuanians', 'Teutons'],
		"Navy nerds" :  ['Vikings', 'Italians','Portuguese', 'Spanish','Koreans','Malay','Japanese',]
	}

fig = px.scatter_3d(X[0:3], x=civ_data.index)

#ax1.set_axis_off()

# Load the cluster that each civ belongs to
#civ_clusters = dict.fromkeys(civ_data.index)
#for civ in civ_clusters.keys():
#    civ_clusters[civ] = set()
#    
#for cluster, civs in clusters.items():
#        for civ in civs:
#            civ_clusters[civ].add(cluster)
#

#i = 0
#ini = 25
#prev = 15
#ax1.view_init(25,0)

#annotate_cluster("Gunpowder gang",clusters["Gunpowder gang"],civ_data,X,ax1)
#ax1.texts[-1].remove()
#ax1.collections[-1].remove()
#ax1.collections[-1].remove()
#plt.draw()
#
#for angle1 in [ini,prev]:
#    for angle2 in range(0, 360, 5):
#        ax1.view_init(angle1, angle2)
#        plt.draw()
##        plt.pause(1)
#        print(angle1,angle2)
#        if angle1==25 and  angle2==5:
#            annotate_cluster("Gunpowder gang",clusters["Gunpowder gang"],civ_data,X,ax1)
#        elif angle1==25 and  angle2==65:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()
#        elif angle1==25 and  angle2==70:
#            annotate_cluster("American bois",clusters["American bois"],civ_data,X,ax1)
#        elif angle1==25 and  angle2==120:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()
#        elif angle1==25 and  angle2==125:
#            annotate_cluster("Camel troupe",clusters["Camel troupe"],civ_data,X,ax1)
#        elif angle1==25 and  angle2==185:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()
#        elif angle1==25 and  angle2==295:
#            annotate_cluster("Infantry bros",clusters["Infantry bros"],civ_data,X,ax1)
#        elif angle1==25 and  angle2==355:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()
#        elif angle1==15 and  angle2==0:
#            annotate_cluster("Archer gang",clusters["Archer gang"],civ_data,X,ax1)
#        elif angle1==15 and  angle2==60:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()
#        elif angle1==15 and  angle2==140:
#            annotate_cluster("Cav archer enthusiasts",clusters["Cav archer enthusiasts"],civ_data,X,ax1)
#        elif angle1==15 and  angle2==200:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()
#        elif angle1==15 and  angle2==205:
#            annotate_cluster("Horse lovers",clusters["Horse lovers"],civ_data,X,ax1)
#        elif angle1==15 and  angle2==265:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw()  
#        elif angle1==15 and  angle2==290:
#                annotate_cluster("Siege dudemeisters",clusters["Siege dudemeisters"],civ_data,X,ax1)
#        elif angle1==15 and  angle2==350:
#            ax1.texts[-1].remove()
#            ax1.collections[-1].remove()
#            ax1.collections[-1].remove()
#            plt.draw() 
#        filename='Animation/Fig1-'+str(angle1)+'-'+str(angle2)+'.png'
#        plt.savefig(filename, dpi=96)
#        images.append(imageio.imread(filename))
#        plt.gca()
#                        
#    for angle3 in np.linspace(angle1,prev,np.abs(angle1-prev)+1):
#        ax1.view_init(angle3, angle2)
#        plt.draw()
##        plt.pause(1)
#        
#        filename='Animation/Fig1-'+str(angle1)+'-'+str(angle2)+'.png'
#        plt.savefig(filename, dpi=96)
#        images.append(imageio.imread(filename))
#        plt.gca()
#    prev=angle1


#corr_matrix = np.array(get_correlation_matrix(civ_data.index,X[:,0:dimensionality+3],n_components))
#corr_matrix = 1 - corr_matrix/(np.max(corr_matrix)-np.min(corr_matrix)) - np.min(corr_matrix)
#
#fig2, ax2 = plt.subplots(figsize=(25,25))
#
#im, _ = heatmap(corr_matrix, civ_data.index, civ_data.index, ax=ax2,
#                cmap=plt.cm.get_cmap('PiYG', 10), vmin=0, vmax=1,
#                cbarlabel="correlation coeff.")
#
#def func_annotate(x, pos):
#    return "{:.2f}".format(x)
#
#annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func_annotate), size=10)
#plt.tight_layout()
#plt.show()