import math
import torch
import torch.nn.functional as F

def fft_3d(A):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float64)
    return torch.fft.fft(A, dim=2)

def ifft_3d(A):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.complex128)
    return torch.fft.ifft(A, dim=2)

def I_tensor(n, n3):
    I = torch.zeros(n, n, n3, dtype=torch.float64)
    I[:, :, 0] = torch.eye(n, dtype=torch.float64)
    return I

def block_diag_matrix(matrices):
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = torch.block_diag(result, matrices[i])
    return result

def t_product(A, B):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.complex128)
    if not isinstance(B, torch.Tensor):
        B = torch.tensor(B, dtype=torch.complex128)
    n3 = A.shape[2]
    A_bar, B_bar = fft_3d(A), fft_3d(B)
    C_bar = []
    t = math.ceil((n3 + 1) / 2) - 1
    for i in range(0, n3):
        if i <= t:
            C_bar_i = torch.matmul(A_bar[:, :, i], B_bar[:, :, i])
        else:
            C_bar_i = torch.conj(C_bar[n3 - i])
        C_bar.append(C_bar_i)
    C_bar = torch.stack(C_bar, dim=2)
    C = ifft_3d(C_bar)
    return C

def t_transpose(V):
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch.complex128)
    n3 = V.shape[2]
    V_t = V.permute(1, 0, 2).clone()
    for j in range(1, n3):
        V_t[:, :, j] = V_t[:, :, n3 - j]
    return V_t

def t_conjTranspose(V):
    if not isinstance(V, torch.Tensor):
        V = torch.tensor(V, dtype=torch.complex128)
    n3 = V.shape[2]
    V_t = V.permute(1, 0, 2).conj().clone()
    for j in range(1, n3):
        V_t[:, :, j] = V_t[:, :, n3 - j]
    return V_t

def t_inverse(A):
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.complex128)
    n3, n, _ = A.shape[2], A.shape[0], A.shape[1]
    I = torch.zeros(n, n, n3, dtype=A.dtype, device=A.device)
    I[:, :, 0] = torch.eye(n, dtype=A.dtype, device=A.device)
    A_bar, I_bar = fft_3d(A), fft_3d(I)
    B_bar = []
    for i in range(0, n3):
        A_bar_i_inv = torch.linalg.inv(A_bar[:, :, i])
        B_bar_i = torch.matmul(A_bar_i_inv, I_bar[:, :, i])
        B_bar.append(B_bar_i)
    B_bar = torch.stack(B_bar, dim=2)
    B = ifft_3d(B_bar)
    return B

def t_plus(S, tao):
    S = S - tao
    S = torch.clamp(S, min=0)
    return S

def t_svt(Y, tao):
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.complex128)
    Y_bar = fft_3d(Y)
    n1, n2, n3 = Y.shape
    t = math.ceil((n3 + 1) / 2) - 1
    W_bar = []
    for i in range(0, t + 1):
        U, S, Vh = torch.linalg.svd(Y_bar[:, :, i], full_matrices=False)
        s = torch.diag(S)
        if n1 == n2:
            S_mat = s
        elif n1 < n2:
            S_mat = F.pad(s, (0, n2 - s.shape[1]))
        else:
            S_mat = F.pad(s, (0, 0, 0, n1 - s.shape[0]))
        S_tao = t_plus(S_mat, tao)
        W_bar_i = U @ S_tao @ Vh
        W_bar.append(W_bar_i)
    for j in range(t + 1, n3):
        W_bar_j = torch.conj(W_bar[n3 - j])
        W_bar.append(W_bar_j)
    W_bar = torch.stack(W_bar, dim=2)
    Y_svt = ifft_3d(W_bar)
    return Y_svt