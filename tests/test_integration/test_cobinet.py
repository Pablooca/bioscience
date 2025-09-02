import bioscience as bs

###################
# 1) Load dataset 
###################
# Usa tu ruta real en Windows:
DATASET = r"C:\Users\ivan0\Documents\Libreria\bioscience\datasets\Spellman_v5.csv"

dataset = bs.load(
    db=DATASET,
    separator=",",
    index_gene=0,
    head=1
)

####################
# 2) Data mining 
####################
# CoBiNet biclustering
# mode=1 → secuencial (implementado)
# mode=2 → CPU paralelo (pendiente)
# mode=3 → GPU (pendiente)

# Si expusiste el WRAPPER (recomendado):
#listModels = bs.processCobinet(dataset, mode=1, deviceCount=1, debug=True)




cobinet = bs.processCobinetBC(dataset, mode=1, deviceCount=1, debug=True)


#print(cobinet)