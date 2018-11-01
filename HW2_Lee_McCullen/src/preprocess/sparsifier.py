from scipy.sparse import csr_matrix

"""
    Converts a matrix into a compressed sparse row (csr) matrix.
"""
def sparsify(matrix):
    csrTup = ([], [])
    csrData = []
    for index, row in enumerate(matrix):
        for idxToken in row:
            csrTup[0].append(index)
            csrTup[1].append(int(idxToken))
            csrData.append(1)
    return csr_matrix((csrData, csrTup))