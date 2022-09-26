import numpy as np
import scipy

#Computes the optimal diffeomorphism between the domain and the image using the model as a map. Assumes model takes in input as [domain_i, image_i]
def optimal_diffeomorphism_LSA(domain, image, model):

    model_image = model(domain)

    #model_image_expanded = np.expand_dims(model_image, axis = 0)
    #ideal_image_expanded = np.expand_dims(image, axis = 1)

    net_graph = scipy.spatial.distance.cdist(model_image.detach().numpy(), image)
    assignments = scipy.optimize.linear_sum_assignment(net_graph)

    return assignments[1]


