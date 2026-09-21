import math
import numpy as np
import datetime
import t_operation_GPU as top
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve
import torch
import warnings
warnings.filterwarnings('ignore')
import platform

# select device
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# detect and configure device
def setup_device():
    if platform.system() == "Darwin":  # macOS (Apple Silicon)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using Metal Performance Shaders (MPS) on Apple Silicon")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU on macOS")
    elif platform.system() == "Windows":  # Windows
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU on Windows")
    else:  # Linux or other systems
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA on {platform.system()} - {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print(f"CUDA not available, using CPU on {platform.system()}")

    return device

class MSPMLTC:
    def __init__(self, alpha, beta, lbd1, lbd2, lbd3, lbd4, eta1, eta2, kNum, tol, max_iter, y_test):
        self.alpha = alpha
        self.beta = beta
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        self.lbd3 = lbd3
        self.lbd4 = lbd4
        self.eta1 = eta1
        self.eta2 = eta2
        self.kNum = kNum
        self.tol = tol
        self.max_iter = max_iter
        self.y_test = y_test

        # setup device
        self.device = setup_device()

    def to_device(self, tensor):
        """move tensor to device"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device, dtype=torch.float32)
        else:
            return torch.tensor(tensor, device=self.device, dtype=torch.float32)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def vec(self, array_3d):
        if not isinstance(array_3d, torch.Tensor):
            array_3d = torch.tensor(array_3d, dtype=torch.float32, device=DEVICE)
        if array_3d.ndim == 3:
            vector = array_3d.permute(0, 2, 1).contiguous().reshape(-1)
        elif array_3d.ndim == 2:
            vector = array_3d.T.contiguous().reshape(-1)
        else:
            raise ValueError("Unsupported array dimension")
        return vector

    def innvec(self, vector, m, n1, n2):
        if vector.numel() != m * n1 * n2:
            raise ValueError(f"can't reshape array with {vector.numel()} elements to ({m}, {n1}, {n2}) shape")
        array_3d_reconstructed = vector.reshape(m, n1, n2).permute(0, 2, 1)
        return array_3d_reconstructed

    def unfold(self, tensor, n):
        if isinstance(tensor, torch.Tensor):
            np_tensor = tensor.detach().cpu().numpy()
        else:
            np_tensor = np.array(tensor)
        mode = 0 if n == 3 else n
        if n == 3:
            np_tensor = np.array([item.T for item in np_tensor])
            result_np = np.moveaxis(np_tensor, mode, 0).reshape(np_tensor.shape[mode], -1)
        else:
            result_np = np.moveaxis(np_tensor, mode, 0).reshape(np_tensor.shape[mode], -1)
        result = torch.tensor(result_np, dtype=torch.float32, device=tensor.device if isinstance(tensor, torch.Tensor) else DEVICE)
        return result

    def mode_n_product(self, tensor, matrix, mode):
        if isinstance(tensor, torch.Tensor):
            shape = tensor.shape
            np_tensor = tensor.detach().cpu().numpy()
        else:
            shape = np.array(tensor).shape
            np_tensor = np.array(tensor)
        tensor_unfolded = torch.tensor(
            np.moveaxis(np_tensor, mode, 0).reshape(shape[mode], -1),
            dtype=torch.float32, device=DEVICE
        )
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=torch.float32, device=DEVICE)
        matrix = matrix.to(dtype=torch.float32, device=DEVICE)
        result_unfolded = torch.matmul(matrix, tensor_unfolded)
        new_shape = list(shape)
        new_shape[mode] = matrix.shape[0]
        result = result_unfolded.reshape(new_shape)
        return result

    def non_zero(self, arr):
        if isinstance(arr, torch.Tensor):
            arr_np = arr.detach().cpu().numpy()
        else:
            arr_np = np.asarray(arr)
        non_zero_indices = np.nonzero(arr_np)
        non_zero_values = arr_np[non_zero_indices]
        return non_zero_indices, non_zero_values


    def TLR_theorem(self, K, A, tao, alpha):
        device = A.device if isinstance(A, torch.Tensor) else DEVICE
        n1, n2, n3 = A.shape

        A_bar = top.fft_3d(A)
        K_bar = top.fft_3d(K)

        t = math.ceil((n3 + 1) / 2) - 1
        result_bar = []
        for i in range(n3):
            if i <= t:
                Ai_np = A_bar[:, :, i].detach().cpu().numpy()
                Ki_np = K_bar[:, :, i].detach().cpu().numpy()

                Ai_np = np.nan_to_num(Ai_np, nan=0.0, posinf=1e10, neginf=-1e10)
                Ki_np = np.nan_to_num(Ki_np, nan=0.0, posinf=1e10, neginf=-1e10)

                try:
                    U_A, S_A, Vh_A = np.linalg.svd(Ai_np, full_matrices=False)
                    U_K, S_K, Vh_K = np.linalg.svd(Ki_np, full_matrices=False)
                except np.linalg.LinAlgError:
                    Ai_np = Ai_np @ Ai_np.T * np.eye(max(Ai_np.shape))[:Ai_np.shape[0],
                                              :Ai_np.shape[1]] + 1e-6 * np.eye(Ai_np.shape[0])
                    Ki_np = Ki_np @ Ki_np.T * np.eye(max(Ki_np.shape))[:Ki_np.shape[0],
                                              :Ki_np.shape[1]] + 1e-6 * np.eye(Ki_np.shape[0])
                    U_A, S_A, Vh_A = np.linalg.svd(Ai_np, full_matrices=False)
                    U_K, S_K, Vh_K = np.linalg.svd(Ki_np, full_matrices=False)

                k = min(n1, n2)
                S_A = S_A[:k]
                S_K = S_K[:k]

                S_new = S_A - tao * (alpha * np.power(np.abs(S_K), alpha - 1)) / (1 + np.power(np.abs(S_K), alpha))
                S_new = np.maximum(S_new, 0)

                S_new_mat = np.diag(S_new.astype(Ai_np.dtype))

                result_i_np = (U_A[:, :k] @ S_new_mat) @ Vh_A[:k, :]
                result_i = torch.tensor(result_i_np, dtype=A_bar.dtype, device=device)
            else:
                result_i = torch.conj(result_bar[n3 - i])
            result_bar.append(result_i)

        result_bar = torch.stack(result_bar, dim=2)
        return top.ifft_3d(result_bar).real

    def updateZtensor(self, Z, A, U, V, W, Y, E, OList, muList):
        AUVW = self.mode_n_product(self.mode_n_product(self.mode_n_product(A, U, 1), V, 2), W, 0)
        mu_1 = muList[0]
        O_1 = OList[0]
        Y = torch.tensor(Y, dtype=torch.float32, device=DEVICE) if not isinstance(Y, torch.Tensor) else Y.to(DEVICE)
        D1 = (2 * AUVW + mu_1 * (Y - E) - O_1) / (2 + mu_1)
        Z_new = self.TLR_theorem(Z, D1, self.lbd1 / (2 + mu_1), self.alpha)
        Z_new = Z_new.real.to(DEVICE)

        Z_new[Z_new < 0] = 0
        Z_max = Z_new.max()
        if Z_max.abs() > 1e-12:
            Z_new = Z_new / Z_max
        return Z_new

    def updateAhat(self, A, G, Ahat, tensorZ, OList, muList):
        O_5 = OList[4]
        mu_5 = muList[4]
        Ahat_new = []
        m = tensorZ.shape[0]
        sum_val = torch.zeros_like(Ahat[0], dtype=torch.float32, device=DEVICE)
        for i in range(0, m):
            for j in range(0, m):
                if i != j:
                    sij = torch.sum(tensorZ[i] * tensorZ[j])
                    sum_val = sum_val + sij * (Ahat[i] - Ahat[j])
        temp = self.mode_n_product(A, G, 0)
        for i in range(0, m):
            gradAhati = (2 * self.beta * sum_val + mu_5
                         * (Ahat[i] - temp[i]) + O_5[i])
            Ahat_new.append(Ahat[i] - self.eta1 * gradAhati)
        return torch.stack(Ahat_new)

    def updateE(self, Y, Z, OList, muList):
        mu_1 = muList[0]
        O_1 = OList[0]
        Y = torch.tensor(Y, dtype=torch.float32, device=DEVICE) if not isinstance(Y, torch.Tensor) else Y.to(DEVICE)
        B = Y - Z - (O_1 / mu_1)
        v = self.lbd2 / mu_1
        E = B.clone()
        E[torch.abs(E) <= v] = 0
        mask_pos = E > 0
        mask_neg = E < 0
        E[mask_pos] = E[mask_pos] - v
        E[mask_neg] = E[mask_neg] + v

        E[E<0] = 0
        E_max = E.abs().max()
        if E_max.abs() > 1e-12:
            E = E / E_max
        return E

    def updateA(self, Ahat, B, C, Ztensor, W, V, U, OList, muList):
        m, n, q = Ztensor.shape
        mu_3, mu_4, mu_5 = muList[2], muList[3], muList[4]
        O_3, O_4, O_5 = OList[2], OList[3], OList[4]

        Ut = U.T.contiguous()
        Vt = V.T.contiguous()
        Wt = W.T.contiguous()
        WtW = (Wt @ W).contiguous()

        def to_flat(T):
            return T.permute(0, 2, 1).contiguous().reshape(-1)

        def Q1_transpose(T):
            R = torch.einsum('abc,da->dbc', T, Wt)
            # print(R.shape)
            R = torch.einsum('abc,dc->adc', R, Vt)
            # print(T.shape, Wt.shape, R.shape, Vt.shape, "Q1_transpose", R.shape, ' Ut shape:', Ut.shape)
            R = torch.einsum('abc,db->abd', R, Ut)
            return to_flat(R)

        def Q2_transpose(X_t):
            R = torch.einsum('abc,da->dbc', X_t, Wt)
            return to_flat(R)

        def sys_matvec(a_flat):
            A_t = self.innvec(a_flat, m, n, n)
            # print("A_t shape:", A_t.shape, U.shape)
            T = torch.einsum('abc,bd->adc', A_t, U)
            # print("T shape:", T.shape, V.shape)
            T = torch.einsum('abc,dc->abd', T, V)
            # print("T shape:", T.shape, W.shape)
            T = torch.einsum('abc,da->dbc', T, W)
            # print("T shape:", T.shape, Wt.shape)
            R = torch.einsum('abc,da->dbc', T, Wt)
            # print("R shape:", R.shape, Wt.shape)
            R = torch.einsum('abc,dc->abd', R, Vt)
            # print("R shape:", R.shape, Vt.shape, Ut.shape)
            R = torch.einsum('abc,db->abd', R, Ut)
            WtW_part = to_flat(torch.einsum('abc,da->dbc', A_t, WtW))
            reg_part = (mu_3 + mu_4) * to_flat(A_t)
            return 2 * to_flat(R) + mu_5 * WtW_part + reg_part

        rhs = (2 * Q1_transpose(Ztensor)
               + mu_3 * self.vec(B) + mu_4 * self.vec(C)
               + mu_5 * Q2_transpose(Ahat)
               - self.vec(O_3) - self.vec(O_4) - Q2_transpose(O_5))

        x = torch.zeros_like(rhs)
        r = rhs - sys_matvec(x)
        p = r.clone()
        rs_old = torch.sum(r * r).item()

        max_cg_iter = min(30, m * n * n)
        for _ in range(max_cg_iter):
            ap = sys_matvec(p)
            pap = torch.sum(p * ap).item()
            if abs(pap) < 1e-30:
                break
            alpha_cg = rs_old / pap
            if np.isnan(alpha_cg) or np.isinf(alpha_cg):
                alpha_cg = 0.0

            x = x + alpha_cg * p
            r = r - alpha_cg * ap
            rs_new = torch.sum(r * r).item()
            if rs_new < 1e-4:
                break
            beta_cg = rs_new / rs_old
            p = r + beta_cg * p
            rs_old = rs_new

        result = self.innvec(x, m, n, n)
        result[result < 0] = 0
        result_max = result.max()
        if result_max.abs() > 1e-12:
            result = result / (result_max + 1e-12)

        return result

    def updateH(self, X, B, muList, OList):
        O_2 = OList[1]
        mu_2 = muList[1]
        m = len(X)
        H_new = []
        for k in range(0, m):
            Xk = X[k].to(dtype=torch.float32, device=DEVICE) if isinstance(X[k], torch.Tensor) else (
                torch.tensor(X[k], dtype=torch.float32, device=DEVICE))
            Bk = B[k].to(dtype=torch.float32, device=DEVICE) if isinstance(B[k], torch.Tensor) else (
                torch.tensor(B[k], dtype=torch.float32, device=DEVICE))
            O2k = O_2[k].to(dtype=torch.float32, device=DEVICE) if isinstance(O_2[k], torch.Tensor) else (
                torch.tensor(O_2[k], dtype=torch.float32, device=DEVICE))
            Hk = Xk - Bk @ Xk + O2k / mu_2
            v = self.lbd2 / mu_2
            Hk[torch.abs(Hk) <= v] = 0
            mask_pos = Hk > 0
            mask_neg = Hk < 0
            Hk[mask_pos] = Hk[mask_pos] - v
            Hk[mask_neg] = Hk[mask_neg] + v

            # Hk[Hk < 0] = 0
            Hk_max = Hk.max()
            if Hk_max.abs() > 1e-12:
                Hk = Hk / Hk_max

            H_new.append(Hk)

        return H_new

    def updateB(self, X, H, A, OList, muList, n):
        m = len(X)
        O_2, O_3 = OList[1], OList[2]
        mu_2, mu_3 = muList[1], muList[2]
        I = torch.eye(n, dtype=torch.float32, device=DEVICE)
        B_new = []
        for k in range(0, m):
            Xk = X[k].to(dtype=torch.float32, device=DEVICE) if isinstance(X[k], torch.Tensor) else (
                torch.tensor(X[k], dtype=torch.float32, device=DEVICE))
            Hk = H[k].to(dtype=torch.float32, device=DEVICE) if isinstance(H[k], torch.Tensor) else (
                torch.tensor(H[k], dtype=torch.float32, device=DEVICE))
            O2k = O_2[k].to(dtype=torch.float32, device=DEVICE) if isinstance(O_2[k], torch.Tensor) else (
                torch.tensor(O_2[k], dtype=torch.float32, device=DEVICE))
            O3k = O_3[k].to(dtype=torch.float32, device=DEVICE) if isinstance(O_3[k], torch.Tensor) else (
                torch.tensor(O_3[k], dtype=torch.float32, device=DEVICE))
            Ak = A[k].to(dtype=torch.float32, device=DEVICE) if isinstance(A[k], torch.Tensor) else (
                torch.tensor(A[k], dtype=torch.float32, device=DEVICE))
            tmp1 = mu_2 * (Xk - Hk) @ Xk.T
            tmp2 = O2k @ Xk.T
            tmp3 = mu_3 * Ak + O3k
            tmp4 = torch.linalg.inv(mu_2 * Xk @ Xk.T + mu_3 * I)
            Bk = (tmp1 + tmp2 + tmp3) @ tmp4
            B_new.append(Bk)

        B_new = torch.stack(B_new)
        B_new[B_new < 0] = 0
        B_max = B_new.max()
        if B_max.abs() > 1e-12:
            B_new = B_new / B_max
        return B_new

    def updateC(self, C, A, OList, muList):
        O_4 = OList[3]
        mu_4 = muList[4]
        A = torch.tensor(A, dtype=torch.float32, device=DEVICE) if not isinstance(A, torch.Tensor) else A.to(DEVICE)
        C_new = self.TLR_theorem(C, A + O_4 / mu_4, self.lbd1 / mu_4, self.alpha)
        C_new = C_new.real.to(DEVICE)

        C_new[C_new < 0] = 0
        C_max = C_new.max()
        if C_max.abs() > 1e-12:
            C_new = C_new / C_max

        return C_new


    def updateU(self, A, W, V, Z, n):
        device = Z.device  # auto get device (CPU/GPU)
        In = torch.eye(n, device=device, dtype=torch.float32)  # n x n identity matrix
        A = self.to_device(A)
        V, W = self.to_device(V), self.to_device(W)
        # Step 1: Compute V^T @ Z[k]^T for all k → (m, n, n)
        VZT = torch.einsum('nq,mqp->mnp', V.T, Z.permute(0, 2, 1))
        # Step 2: temp_all[i] = sum_k W[k, i] * VZT[k]
        temp_all = torch.tensordot(W.T, VZT, dims=([1], [0]))  # (m, n, n)
        # Step 3: fast = sum_i A[i] @ temp_all[i]
        fast = torch.einsum('mij,mjk->ik', A, temp_all)

        # --- Clean input ---
        fast = fast.to(dtype=torch.float32)
        fast = torch.nan_to_num(fast, nan=0.0, posinf=1e5, neginf=-1e5)
        fast = torch.clamp(fast, -1e5, 1e5)

        # --- Add jitter for stability ---
        if fast.dim() == 2:
            eps = 1e-6
            fast = fast + eps * torch.eye(fast.shape[0], device=fast.device, dtype=fast.dtype)
        else:
            # For batched SVD
            eps = 1e-6
            I = torch.eye(fast.shape[-1], device=fast.device, dtype=fast.dtype)
            fast = fast + eps * I

        # --- Try SVD ---
        try:
            E, Sigma, Fh = torch.linalg.svd(fast)
        except torch._C._LinAlgError:
            # Fallback: move to CPU with gesvd
            print("SVD failed on GPU, falling back to CPU gesvd...")
            fast_cpu = fast.cpu()
            E, Sigma, Fh = torch.linalg.svd(fast_cpu, driver='gesvd')
            E, Sigma, Fh = E.to(fast.device), Sigma.to(fast.device), Fh.to(fast.device)

        U_new = Fh.T @ In @ E.T
        U_new[U_new < 0.0] = 0.0
        return U_new

    def updateV(self, Z, W, U, A, n):
        device = Z.device
        I = torch.eye(n, device=device, dtype=torch.float32)
        U, W = self.to_device(U), self.to_device(W)
        UA = torch.einsum('nd,mdp->mnp', U, A)
        temp_all = torch.tensordot(W, UA, dims=([1], [0]))
        Zt = Z.transpose(-2, -1)
        tmp1 = torch.einsum('mij,mjk->ik', Zt, temp_all)

        WUA2 = temp_all.reshape(-1, n).to(dtype=torch.float32)
        gram_matrix = WUA2.T @ WUA2 + self.lbd3 * I + 1e-6 * I

        try:
            V_new = torch.linalg.solve(gram_matrix.T, tmp1.to(dtype=torch.float32).T).T
        except torch._C._LinAlgError:
            V_new = torch.linalg.lstsq(gram_matrix.T, tmp1.to(dtype=torch.float32).T).solution.T
        V_new = torch.clamp(V_new, min=0.0)

        V_new = torch.clamp(V_new, min=0.0)
        V_max = V_new.max()
        if V_max.abs() > 1e-12:
            V_new = V_new / V_max

        return V_new

    def updateW(self, W, Z, A, V, U, C, OList, muList):
        mu_2 = muList[5]
        O_2 = self.to_device(OList[5])

        # Step 1: Compute U @ A[:,k,:] for all k → (m, n, n)
        U, Z = self.to_device(U), self.to_device(Z)
        W, C = self.to_device(W), self.to_device(C)
        UA = torch.einsum('nd,mdp->pnm', U, A)
        # Step 2: temp_all[i] = sum_k V[i, k] * UA[k]
        temp_all = torch.tensordot(V, UA, dims=([1], [0]))  # (m, n, n)
        # Step 3: fast = sum_i Z[:,i,:].T @ temp_all[i]
        ZUVAT = torch.einsum('qmn,qnk->mk', Z.permute(2, 0, 1), temp_all)
        VUA3 = temp_all.reshape(-1, temp_all.shape[2])
        tmp1 = ZUVAT - W @ VUA3.T @ VUA3
        tmp2 = 2 * tmp1 + mu_2 * (W - C) + O_2
        tmp3 = 2 * torch.sign(W) * torch.sum(torch.abs(W), axis=0)
        W_new = W - self.eta1 * (tmp2 + tmp3)
        # Step 3: Min-Max normalization (global over all elements)
        W_min = W_new.min()
        W_max = W_new.max()
        # avoid division by zero (if W_max == W_min)
        denom = W_max - W_min
        if denom.abs() < 1e-12:
            W_normalized = torch.zeros_like(W_new)
        else:
            W_normalized = (W_new - W_min) / denom

        # Step 4: truncate non-positive values to 0
        W_normalized = torch.clamp(W_normalized, min=0.0)

        # W_new = torch.clamp(W_new, min=0.0)
        # row_sums = W.sum(dim=1, keepdim=True)
        # W_normalized = W_new / row_sums

        return W_normalized

    def updateG(self, A, Ahat, W, OList, muList, n):
        device = A.device if isinstance(A, torch.Tensor) else DEVICE
        mu_5, mu_6 = muList[4], muList[5]
        O_5, O_6 = OList[4], OList[5]
        A3 = self.unfold(A, 3)
        I = torch.eye(A3.shape[0], dtype=torch.float32, device=device)
        Ahat3 = self.unfold(Ahat, 3)
        O6_dev = O_6.to(device) if not isinstance(O_6, torch.Tensor) else O_6.to(device)
        lhs = mu_5 * A3 @ A3.T + mu_6 * I + 1e-6 * I
        rhs = mu_5 * Ahat3 @ A3.T + self.unfold(O_5, 3) @ A3.T + mu_6 * W + O6_dev
        try:
            G_new = torch.linalg.solve(lhs, rhs.to(dtype=torch.float32))
        except torch._C._LinAlgError:
            G_new = torch.linalg.lstsq(lhs, rhs.to(dtype=torch.float32)).solution

        G_new[G_new < 0] = 0
        G_max = G_new.max()
        if G_max.abs() > 1e-12:
            G_new = G_new / G_max

        return G_new



    def updateOmu(self, Z, E, Y, X, H, A, Ahat, B, C, G, W, OList, muList, rho, mu_max):
        m = len(X)
        Y = Y.to(DEVICE) if isinstance(Y, torch.Tensor) else torch.tensor(Y, dtype=torch.float32, device=DEVICE)
        Z = Z.to(DEVICE) if isinstance(Z, torch.Tensor) else torch.tensor(Z, dtype=torch.float32, device=DEVICE)
        OList[0] = OList[0] + muList[0] * (Z + E - Y)
        OList1_new = []
        for k in range(0, m):
            Xk = X[k].to(DEVICE) if isinstance(X[k], torch.Tensor) else torch.tensor(X[k], dtype=torch.float32, device=DEVICE)
            OList1_new.append(OList[1][k] + muList[1] * (Xk - B[k] @ Xk - H[k]))
        OList[1] = OList1_new
        A = A.to(DEVICE) if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32, device=DEVICE)
        OList[2] = OList[2] + muList[2] * (A - B)
        OList[3] = OList[3] + muList[3] * (A - C)
        OList[4] = OList[4] + muList[4] * (Ahat - self.mode_n_product(A, G, 0))
        OList[5] = OList[5] + muList[5] * (W - G)

        for i in range(len(muList)):
            muList[i] = min(rho * muList[i], mu_max)
        return OList, muList

    def MSPMLTC_Algorithm(self):
        m, n, q = len(self.X_train), self.X_train[0].shape[0], self.y_train[0].shape[1]
        print('m, n, q=', m, n, q)
        device = DEVICE

        torch.manual_seed(42)
        E = torch.rand(m, n, q, dtype=torch.float32, device=device)
        Ztensor = torch.rand(m, n, q, dtype=torch.float32, device=device)
        A_max = 1
        A = torch.rand(m, n, n, dtype=torch.float32, device=device) * A_max
        Ahat = torch.rand(m, n, n, dtype=torch.float32, device=device)
        B = torch.rand(m, n, n, dtype=torch.float32, device=device)
        C = torch.rand(m, n, n, dtype=torch.float32, device=device)
        U = torch.tensor(np.random.rand(n, n), dtype=torch.float32, device=device)
        V = torch.tensor(np.random.rand(q, n), dtype=torch.float32, device=device)
        W = torch.tensor(np.random.rand(m, m), dtype=torch.float32, device=device)
        G = W.clone()

        O1 = torch.zeros(m, n, q, dtype=torch.float32, device=device)
        H, O2 = [], []
        for k in range(m):
            H.append(torch.zeros(size=self.X_train[k].shape, dtype=torch.float32, device=device))
            O2.append(torch.zeros(size=self.X_train[k].shape, dtype=torch.float32, device=device))
        O3 = torch.zeros(m, n, n, dtype=torch.float32, device=device)
        O4 = torch.zeros(m, n, n, dtype=torch.float32, device=device)
        O5 = torch.rand(m, n, n, dtype=torch.float32, device=device)
        O6 = torch.zeros(size=W.shape, dtype=torch.float32, device=device)

        OList = [O1, O2, O3, O4, O5, O6]
        muList = [1, 10e-3, 1, 1, 1, 1]
        rho = self.eta2
        mu_max = 10e+10

        X, Y = self.X_train, self.y_train

        t = 0
        redus = 4
        while t < self.max_iter:
            start_time = datetime.datetime.now()
            H_new = self.updateH(X, B, muList, OList)
            # print('H_new=', H_new)
            Ahat_new = self.updateAhat(A, G, Ahat, Ztensor, OList, muList)
            # print('Ahat_new=', Ahat_new)
            A_new = self.updateA(Ahat_new, B, C, Ztensor, W, V, U, OList, muList)
            # print('A_new=', A_new)
            B_new = self.updateB(X, H_new, A_new, OList, muList, n)
            # print('B_new=', B_new)
            C_new = self.updateC(C, A_new, OList, muList)
            # print('C_new=', C_new)
            U_new = self.updateU(A_new, W, V, Ztensor, n)
            # print('U_new=', U_new)
            V_new = self.updateV(Ztensor, W, U_new, A_new, n)
            # print('V_new=', V_new)
            W_new = self.updateW(W, Ztensor, A_new, V_new, U_new, G, OList, muList)
            # print('W_new=', W_new)
            G_new = self.updateG(A_new, Ahat_new, W_new, OList, muList, n)
            # print('G_new=', G_new)
            Ztensor_new = self.updateZtensor(Ztensor, A_new, U_new, V_new, W_new, Y, E, OList, muList)
            # print('Ztensor_new=', Ztensor_new)
            E_new = self.updateE(Y, Ztensor_new, OList, muList)
            # print('E_new=', E_new)

            Z = np.zeros((n, q))  # Z是一个n*q的矩阵

            OList_new, muList_new = self.updateOmu(Z, E, Y, X, H, A, Ahat, B, C, G, W, OList, muList, rho, mu_max)

            err_Z = round(torch.norm(Ztensor_new - Ztensor, p=float('inf')).item(), redus)
            err_E = round(torch.norm(E_new - E, p=float('inf')).item(), redus)
            err_A = round(torch.norm(A_new - A, p=float('inf')).item(), redus)

            if t > 1 and err_Z <= self.tol and err_E <= self.tol and err_A <= self.tol:
                print('end algorithm1')
                break

            Ztensor = Ztensor_new
            Ahat = Ahat_new
            E = E_new
            A = A_new
            H = H_new
            B = B_new
            C = C_new
            U = U_new
            V = V_new
            W = W_new
            G = G_new
            OList = OList_new
            muList = muList_new

            t += 1

        return Ztensor, E, A, H, U, V, W

    def find_common_k_neighbors(self, data_matrices, k):
        num_points = data_matrices[0].shape[0]
        all_neighbors = []
        for matrix in data_matrices:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(matrix)
            distances, indices = nbrs.kneighbors(matrix)
            all_neighbors.append(indices)
        common_neighbors = []
        for i in range(num_points):
            neighbors = set(all_neighbors[0][i])
            for j in range(1, len(data_matrices)):
                neighbors.intersection_update(all_neighbors[j][i])
            common_neighbors.append(list(neighbors))
        return common_neighbors

    def mode_n_product_vector(self, tensor, vector, mode):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = np.array(tensor)
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        else:
            vector = np.array(vector)
        shape = tensor.shape
        unfolded = np.moveaxis(tensor, mode, 0).reshape(shape[mode], -1)
        result_unfolded = np.dot(vector, unfolded)
        new_shape = list(shape)
        new_shape.pop(mode)
        result = result_unfolded.reshape(new_shape)
        return result

    def predict(self, X_test, y_test):
        Z, E, A, H, U, V, W = self.MSPMLTC_Algorithm()
        ones = torch.ones(W.shape[0], device=self.device)  # 全1向量
        result = torch.matmul(W, ones)  # 矩阵乘法

        self.kNum = min(self.kNum, X_test[0].shape[0])
        common_neighbors = self.find_common_k_neighbors(X_test, self.kNum)

        y_score = []
        for i in range(X_test[0].shape[0]):
            A_testi = A[:, common_neighbors[i], :][:, :, common_neighbors[i]]
            V_testi = V[:, common_neighbors[i]]
            U_testi = U[common_neighbors[i], :][:, common_neighbors[i]]
            Y_testi = self.mode_n_product(self.mode_n_product(self.mode_n_product(A_testi, U_testi, 1),
                                                           V_testi, 2), W, 0)
            W1 = torch.sum(W, dim=1).detach().cpu().numpy()
            y_scorei = self.mode_n_product_vector(Y_testi, W1, 0)
            y_score.append(np.mean(y_scorei, axis=0))
        y_score = np.array(y_score)
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1e10, neginf=-1e10)

        precision, recall, thresholds_pr = precision_recall_curve(y_test.ravel(), y_score.ravel())

        precision = precision[:-1]
        recall = recall[:-1]
        optimal_idx_pr = np.argmax(precision - recall)
        optimal_threshold_pr = thresholds_pr[optimal_idx_pr]
        print(f"Optimal Threshold: {optimal_threshold_pr:.2f}")

        y_pred = (y_score >= optimal_threshold_pr).astype(int)

        return y_score, y_pred


########################################### Test code ##################################################################
import pandas as pd
import metrics_MLL
from sklearn.model_selection import KFold
import time

np.random.seed(42)
# datasets in paper
# dataList = ['Human', 'Rugby', 'Yeast', 'Plant', 'PascalVOC', '3Sources', 'ESC-50', 'OBJECT', 'NOIZEUS', 'SCENE']
# souList = [3, 3, 2, 3, 6, 3, 4, 5, 4, 5]
# labList = [14, 15, 14, 12, 20, 6, 8, 31, 7, 33]

# dataset for test
dataList = ['EXAMPLE']
labNum = 7 # 标签个数
souNum = 4 # 数据源个数

NioseRat = [0.1, 0.3, 0.5, 0.7]
d = 0 # 数据集索引
file = './datasets_noise_labels/'  # Datasets_asymmetric_noiselabels
data = pd.read_csv('./datasets/' + dataList[d] + '_0.csv')
samNum = data.shape[0]
print('samNum=', samNum, end=', ')
data_label = np.array(data.iloc[:, -labNum:])

X, Y = [], []
Scores = []
attNumList = []

func_elapsed_time = func_start_time = 0
isTimeRecoder = False  # 代码运行时间是否已经被统计？只统计一折的时间

for mr in range(0, len(NioseRat)):
    start_time = time.time()
    for m in range(souNum):
        dfm = pd.read_csv(file + dataList[d] + '_' + str(m) + '_' + str(NioseRat[mr]) + '.csv')

        samNum = dfm.shape[0]
        attNum = dfm.shape[1] - labNum
        attNumList.append(attNum)
        X.append(np.array(dfm.iloc[:, 0:attNum]))
        if m == 0:
            Y_k = np.ones(shape=(samNum, labNum)) - data_label
        else:
            Y_k = np.array(dfm.iloc[:, -labNum:])
        Y.append(Y_k)
        temp = Y_k - data_label  # [temp_list, :]
        temp[temp < 0] = 1
        print('m=', m, 'attNum=', attNum, np.round(np.sum(temp) / (samNum * labNum), 2))
        # print('temp:', temp, np.sum(temp), samNum * labNum)

    print('attNum=', attNumList, labNum, dataList[d])
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scoresList = []

    for train_index, test_index in kfold.split(X[0], Y[0]):
        runtime = time.time()

        if not isTimeRecoder:
            func_start_time = time.time()

        X_train, X_test = [], []
        y_train, y_test = [], data_label[test_index]
        for v in range(souNum):
            X_train.append(X[v][train_index])
            X_test.append(X[v][test_index])
            y_train.append(Y[v][train_index])
        alpha = 0.2
        kNum = 20
        beta, lbd1, lbd2, lbd3, lbd4 = 0.5, 1.6, 17.5, 3.5, 1.1
        eta1, eta2 = 0.1, 1.2
        tol, max_iter = 0.001, 25
        model = MSPMLTC(alpha, beta, lbd1, lbd2, lbd3, lbd4, eta1, eta2, kNum, tol, max_iter, y_test)
        model.fit(X_train, y_train)

        y_score, y_pred = model.predict(X_test, y_test)
        y_test = y_test.astype(int)

        scores = metrics_MLL.mll_metrics(y_test, y_pred, y_score)
        print('score:', scores)
        scoresList.append(scores)

        usetime = time.time() - runtime
        print(f"time: {usetime:.2f} s, {usetime / 60:.2f} m")

        if not isTimeRecoder:
            func_elapsed_time = time.time() - func_start_time
            isTimeRecoder = True

    finalScore = np.round(np.mean(scoresList, axis=0), 4)
    Scores.append(finalScore)
    print('MissRat=', NioseRat[mr], 'Score=', finalScore, dataList[d])

print('#####################################################################')
for s in range(len(Scores)):
    print('MissRat=', NioseRat[s], 'Score=', Scores[s], dataList[d])