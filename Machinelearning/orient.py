import sys
import Adaboost_image
import knn
import imagenn

if sys.argv[3] == 'nearest':
    knn.read_data(sys.argv[1], sys.argv[2])
elif sys.argv[3] == 'adaboost':
    Adaboost_image.read_data(sys.argv[1], sys.argv[2], int(sys.argv[4]))
elif sys.argv[3] == 'nnet':
    imagenn.read_data(sys.argv[1], sys.argv[2], int(sys.argv[4]))
elif sys.argv[3] == 'best':
    Adaboost_image.read_data(sys.argv[1], sys.argv[2], 70)
