import numpy as np
import matplotlib.pyplot as plt



def normalize(filter):
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            filter[i,j,:,:] = (filter[i,j,:,:] - np.min(filter[i,j,:,:]))/(np.max(filter[i,j,:,:])-np.min(filter[i,j,:,:]))

    return filter

a = kernels
a = normalize(a)
a = a.transpose(0,2,3,1)

fig=plt.figure(figsize=(6, 6))
for i in range(1, 6*6+1):
    fig.add_subplot(6, 6, i)
    plt.imshow(a[i-1,:,:,:])
    plt.xticks([])
    plt.yticks([])

plt.show()



