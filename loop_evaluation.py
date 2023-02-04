import numpy as np
from InstanceSHAP.kernel_sampling import INSTANCEBASEDSHAP


if __name__ == '__main__':
    instance_based_class = INSTANCEBASEDSHAP()
    data = instance_based_class.read_data()
    improvement = []
    for i in range(10):
        gini_vals = instance_based_class.Compare_explanations()
        improvement.append(gini_vals[1]-gini_vals[0])

    print(np.mean(improvement))