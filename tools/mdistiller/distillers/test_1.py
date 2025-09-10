import torch

def merge_matrix(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        return
    matrix = torch.zeros_like(matrix1)

    for i in range(matrix1.shape[0]):
    	   
        for j in range(matrix1.shape[1]):
            a = matrix1[i, j].item()
            b = matrix2[i, j].item()
            matrix[i, a] = b
    return matrix

matrix1 = torch.tensor([
    [1, 2, 4, 0, 3],
    [2, 4, 3, 1, 0],
    [1, 2, 3, 4, 0]
])

matrix2 = torch.tensor([
    [True, True, False, False, True],
    [False, False, True, True, False],
    [True, True, True, False, False]
])

print(merge_matrix(matrix1, matrix2)) 