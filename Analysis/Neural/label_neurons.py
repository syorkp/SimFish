

def check_feature_selectivity(vector):
    """Check responsiveness to prey or predator."""


def check_size_selectivity(vector, feature):
    """For a given feature, check which sizes responsive to"""



def label_all_units_distances(response_vectors):
    """Returns a neuron dimensional list of lists, each of which contains all selectivity properties of neurons.
    Higher level properties could then be inferred by combinations of selectivities"""
    selectivities = [[] for i in response_vectors]
    for i, unit in enumerate(response_vectors):
        # TODO: Check if responsive ot prey or predators. Add to that units selectivity.
        # TODO: For each identified response, check size selectivity.
        # TODO: For each identified response, check motion selectivity.
        # TODO: Return the angle for which has a significant response.
        # TODO: Add whether all of these responses are robust to red background.
        ...


def label_all_units_tsne(response_vectors, archetypes):
    """Based on the archetypes, clusters the nearest units and assigns this manual category to them."""
    ...

