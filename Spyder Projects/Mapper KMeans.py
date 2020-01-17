#import os
#import sys
#import base64
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import ensemble
from sklearn import manifold
import kmapper as km
import pandas as pd
from scipy.misc import imsave, toimage #pip3 install scipy==1.2.0     imsave

###########################################
###Cargamos los datos#### Importar datos###         
###########################################

df = pd.read_csv("C://Users/ServW10/Documents/Datasets/betti0AbnormalMessidor.csv")#,encoding='latin-1')
feature_names = [c for c in df.columns if c not in ["70"]]
#Funcion lambda para clasificacion de datos df["70"] = df["70"].apply(lambda x: 1 if x == "M" else 0)
X = np.array(df["70"])#.fillna(0))
y = np.array(df["70"])
print(X.shape)
X = X.reshape(-1, 1)
print(X.shape)
data = X

###########################################################################################################
### Creamos las imagenes para un array tooltip personalizado                                            ###
tooltip_s = np.array(y)  # need to make sure to feed it as a NumPy array, not a list                    ###
#tooltip = How to display simple text hints above widget when holding mouse over them.                  ###
###########################################################################################################
###Inicializar para usar t-SNE con 2 componentes(reduce los datos a 2 dimensiones,                      ###
###Hay que tener en cuenta en cuenta el alto porcentaje de superposición.                               ###
#######################################################################################################################
# Create a custom 1-D lens with **Isolation Forest**                                                                ###
#Return the anomaly score of each sample using the IsolationForest algorithm                                        ###
#The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly                      ###
#selecting a split value between the maximum and minimum values of the selected feature.                            ###
#Since recursive partitioning can be represented by a tree structure, the number of splittings                      ###
#required to isolate a sample is equivalent to the path length from the root node to the terminating node.          ###
#This path length, averaged over a forest of such random trees, is a measure of normality and our decisionfunction. ###
#Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees          ###
#collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.          ###
#######################################################################################################################

# Create a custom 1-D lens with Isolation Forest
model = ensemble.IsolationForest(random_state=1729) #If int, random_state is the seed used by the random number generator;
model.fit(X)
lens1 = model.decision_function(X).reshape((X.shape[0], 1))

# Create another 1-D lens with L2-norm
mapper = km.KeplerMapper(verbose=0)
lens2 = mapper.fit_transform(X, projection="l2norm")

# Combine both lenses to get a 2-D [Isolation Forest, L^2-Norm] lens
lens = np.c_[lens1, lens2]

##############################################################################################################
### Aplicacion del cluster KMeans proveniente de la libreria SKLearn                                       ###
##############################################################################################################
# Define the simplicial complex
graph = mapper.map(lens,
                      X,
                      nr_cubes=15,#15
                      overlap_perc=0.7,#0.7
                      clusterer=sklearn.cluster.KMeans(n_clusters=2,#module gathers popular unsupervised clustering algorithms.
                                                       random_state=3471))#Determines random number generation for centroid initialization.
                                                                          #Use an int to make the randomness deterministic


##############################################################################################################
###creacion de las vizualizaciones(Incrementado el graph_gravity para una apariencia gráfica más ajustada.)###
##############################################################################################################
print("Output: Grafo de ejemplo para HTML" )

### Tooltip con datos de imagen para cada miembro del cluster
mapper.visualize(graph,
                 title="Algoritmo Mapper en digitos escritos a mano",
                 path_html="C:\\Users\ServW10\Documents\Spyder Projects\Mapper_KMeans_dataCluster.html",
                 #color_function=labels,
                 custom_tooltips=tooltip_s)

### Toolptips con el target y-labels para cada miembro del cluster
mapper.visualize(graph,
                 title="Algoritmo Mapper en digitos escritos a mano",
                 path_html="C:\\Users\ServW10\Documents\Spyder Projects\Mapper_Kmeans_labelsCluster.html",
                 custom_tooltips=y)

# Matplotlib ejemplo para mostrar en consola y comparar
km.draw_matplotlib(graph)#, layout="spring"
plt.show()


