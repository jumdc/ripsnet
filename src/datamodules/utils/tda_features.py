import numpy as np
import gudhi as gd


def alpha_pi(X, maxd):
    pd = []
    st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
    st.persistence()
    dg = st.persistence_intervals_in_dimension(1)
    if len(dg) == 0:
        dg = np.empty([0,2])
    pd.append(dg)

    clean_pd = gd.representations.DiagramSelector(use=True).fit_transform(pd)
    