import numpy as np


class M:
    Ex = 0
    Ey = 1
    Ez = 2
    Hx = 3
    Hy = 4
    Hz = 5
    n = 6


class OverlapTools:
    @staticmethod
    def mode_to_numpy(mode):
        num_freqs = 1
        np_mode = np.zeros((num_freqs, 7, mode.x.shape[0], mode.y.shape[0]), dtype=complex)
        np_mode[:, M.Ex, :, :] = mode.Ex[:, :]
        np_mode[:, M.Ey, :, :] = mode.Ey[:, :]
        np_mode[:, M.Ez, :, :] = mode.Ez[:, :]
        np_mode[:, M.Hx, :, :] = mode.Hx[:, :]
        np_mode[:, M.Hy, :, :] = mode.Hy[:, :]
        np_mode[:, M.Hz, :, :] = mode.Hz[:, :]
        np_mode[:, M.n, :, :] = mode.n[:, :]
        return np.array(np_mode)

    @staticmethod
    def meep_overlap(E1x, E1y, H1x, H1y, E2x, E2y, H2x, H2y, x, y):
        term1 = np.conj(E1x) * H2y - np.conj(E1y) * H2x
        term2 = E2x * np.conj(H1y) - E2y * np.conj(H1x)
        return np.trapz(np.trapz((term1 + term2), x, axis=-2), y, axis=-1)

    @staticmethod
    def eme_overlap(E1x, E1y, H1x, H1y, E2x, E2y, H2x, H2y, x, y):
        cross = E1x * H2y - E1y * H2x
        return np.trapz(np.trapz(cross, x, axis=-2), y, axis=-1)

    @staticmethod
    def fom_overlap(E1x, E1y, H1x, H1y, E2x, E2y, H2x, H2y, x, y):
        term1 = E1x * H2y - E1y * H2x
        term2 = E2x * H1y - E2y * H1x
        return 0.5 * np.trapz(np.trapz((term1 + term2), x, axis=-2), y, axis=-1)

    @staticmethod
    def lumerical_complex(E1x, E1y, H1x, H1y, E2x, E2y, H2x, H2y, x, y):
        term12 = np.trapz(np.trapz(E1x * np.conj(H2y) - E1y * np.conj(H2x), x, axis=-2), y, axis=-1)
        term21 = np.trapz(np.trapz(E2x * np.conj(H1y) - E2y * np.conj(H1x), x, axis=-2), y, axis=-1)
        term11 = np.trapz(np.trapz(E1x * np.conj(H1y) - E1y * np.conj(H1x), x, axis=-2), y, axis=-1)
        term22 = np.trapz(np.trapz(E2x * np.conj(H2y) - E2y * np.conj(H2x), x, axis=-2), y, axis=-1)
        return (term12 * term21 / term11) * 1 / (term22)

    @staticmethod
    def lumerical_overlap(E1x, E1y, H1x, H1y, E2x, E2y, H2x, H2y, x, y):
        term12 = np.trapz(np.trapz(E1x * np.conj(H2y) - E1y * np.conj(H2x), x, axis=-2), y, axis=-1)
        term21 = np.trapz(np.trapz(E2x * np.conj(H1y) - E2y * np.conj(H1x), x, axis=-2), y, axis=-1)
        term11 = np.trapz(np.trapz(E1x * np.conj(H1y) - E1y * np.conj(H1x), x, axis=-2), y, axis=-1)
        term22 = np.trapz(np.trapz(E2x * np.conj(H2y) - E2y * np.conj(H2x), x, axis=-2), y, axis=-1)
        return np.abs(np.real(term12 * term21 / term11) * 1 / np.real(term22))

    # Class attribute - change for different overlaps
    inner_product = eme_overlap

    @staticmethod
    def inner(left, right, x, y):
        """
        left: (f, M, n, Nx, Ny)
        right: (f, M, n, Nx, Ny)
        x: (Nx)
        y: (Ny)
        inner: (f, M)
        """
        return OverlapTools.inner_product(
            E1x=left[:, :, M.Ex],
            E1y=left[:, :, M.Ey],
            H1x=left[:, :, M.Hx],
            H1y=left[:, :, M.Hy],
            E2x=right[:, :, M.Ex],
            E2y=right[:, :, M.Ey],
            H2x=right[:, :, M.Hx],
            H2y=right[:, :, M.Hy],
            x=x,
            y=y,
        )

    @staticmethod
    def massinner(left, right, x, y):
        """
        left: (f, M2, M1, n, Nx, Ny)
        right: (f, M2, M1, n, Nx, Ny)
        x: (Nx)
        y: (Ny)
        inner: (f, M2, M1)
        """
        return OverlapTools.inner_product(
            E1x=left[:, :, :, M.Ex],
            E1y=left[:, :, :, M.Ey],
            H1x=left[:, :, :, M.Hx],
            H1y=left[:, :, :, M.Hy],
            E2x=right[:, :, :, M.Ex],
            E2y=right[:, :, :, M.Ey],
            H2x=right[:, :, :, M.Hx],
            H2y=right[:, :, :, M.Hy],
            x=x,
            y=y,
        )


class InterfaceSolver(object):

    """
    LM: (f, M, 7, Nx, Ny)
    RM: (f, M, 7, Nx, Ny)
    """

    def __init__(self, left_layer, right_layer):
        self.LM = np.stack([OverlapTools.mode_to_numpy(mode) for mode in left_layer.modes], axis=1)
        self.RM = np.stack([OverlapTools.mode_to_numpy(mode) for mode in right_layer.modes], axis=1)
        self.x = left_layer.modes[0].x
        self.y = left_layer.modes[0].y

        # Normalize LM
        norm_LM = OverlapTools.inner(self.LM, self.LM, self.x, self.y)[:, :, np.newaxis, np.newaxis, np.newaxis]
        norm_LM = np.repeat(norm_LM, 7, axis=2)
        norm_LM = np.repeat(norm_LM, self.LM.shape[3], axis=3)
        norm_LM = np.repeat(norm_LM, self.LM.shape[4], axis=4)
        self.LM /= np.sqrt(np.abs(norm_LM))

        # Normalize RM
        norm_RM = OverlapTools.inner(self.RM, self.RM, self.x, self.y)[:, :, np.newaxis, np.newaxis, np.newaxis]
        norm_RM = np.repeat(norm_RM, 7, axis=2)
        norm_RM = np.repeat(norm_RM, self.RM.shape[3], axis=3)
        norm_RM = np.repeat(norm_RM, self.RM.shape[4], axis=4)
        self.RM /= np.sqrt(np.abs(norm_RM))

    def solve(self):
        """
        S: (f, M1+M2, M1+M2)
        """

        # Get mode array shape
        f, M1, _, Nx, Ny = self.LM.shape
        _, M2, _, _, _ = self.RM.shape

        # Initialize S in tensor form
        S = np.zeros((f, M1 + M2, M1 + M2), dtype=complex)

        # Get T
        T1 = self.get_T(self.LM, self.RM)  # Shape (f, M1, M2)
        T2 = self.get_T(self.RM, self.LM)  # Shape (f, M2, M1)

        # Get R
        R1 = self.get_R(T1, self.LM, self.RM)  # Shape (f, M1, M1)
        R2 = self.get_R(T2, self.RM, self.LM)  # Shape (f, M2, M2)

        # Create meshgrids for indices
        _M1 = np.arange(M1)
        _M2 = np.arange(M2)

        # Create proper mappings for forward indices
        _out_T1 = M1 + _M2
        _in_T1 = _M1
        _out_R1 = _M1
        _in_R1 = _M1

        # Create proper mappings for backward indices
        _out_T2 = _M1
        _in_T2 = M1 + _M2
        _out_R2 = M1 + _M2
        _in_R2 = M1 + _M2

        # Create linspaces from mappings
        _out_T1_, _in_T1_ = np.meshgrid(_out_T1, _in_T1)
        _out_T2_, _in_T2_ = np.meshgrid(_out_T2, _in_T2)
        _out_R1_, _in_R1_ = np.meshgrid(_out_R1, _in_R1)
        _out_R2_, _in_R2_ = np.meshgrid(_out_R2, _in_R2)

        # Properly place T and R
        S[:, _out_T1_, _in_T1_] = T1
        S[:, _out_T2_, _in_T2_] = T2
        S[:, _out_R1_, _in_R1_] = R1
        S[:, _out_R2_, _in_R2_] = R2

        return S

    def get_T(self, LM, RM):
        """
        LM: (f, M1, 7, Nx, Ny)
        RM: (f, M2, 7, Nx, Ny)
        T: (f, M1, M2)
        """
        f, M1, _, Nx, Ny = LM.shape
        _, M2, _, _, _ = RM.shape
        x = self.x
        y = self.y

        # Initialize out
        T = np.zeros((f, M1, M2), dtype=complex)

        # Create linspaces for indices
        _i = np.arange(M1)
        _j = np.arange(M2)
        _i_, _j_ = np.meshgrid(_i, _j)

        # Define X
        X = OverlapTools.massinner(LM[:, _i_, :, :, :], RM[:, _j_, :, :, :], x, y) + OverlapTools.massinner(
            RM[:, _j_, :, :, :], LM[:, _i_, :, :, :], x, y
        )
        X_inv = np.transpose(np.linalg.pinv(X), axes=(0, 2, 1))

        # Define Y
        Y = np.zeros((f, M1, M1), dtype=complex)
        Y[:, _i, _i] = 2 * OverlapTools.inner(LM[:, _i, :, :, :], LM[:, _i, :, :, :], x, y)

        # Solve for T
        T[:, _i_, _j_] = np.matmul(X_inv, Y)

        return T

    def get_R(self, T, LM, RM):
        """
        T: (f, M1, M2)
        LM: (f, M1, 7, Nx, Ny)
        RM: (f, M2, 7, Nx, Ny)
        out: (f, M1, M1)
        """
        f, M1, _, Nx, Ny = LM.shape
        _, M2, _, _, _ = RM.shape
        x = self.x
        y = self.y

        # Initialize out
        R = np.zeros((f, M1, M1), dtype=complex)

        # Create linspaces for indices
        _i = np.arange(M1)
        _j = np.arange(M2)
        _i_, _j_ = np.meshgrid(_i, _j)
        _i1_, _i2_ = np.meshgrid(_i, _i)

        # Define X
        X = OverlapTools.massinner(RM[:, _j_, :, :, :], LM[:, _i_, :, :, :], x, y) - OverlapTools.massinner(
            LM[:, _i_, :, :, :], RM[:, _j_, :, :, :], x, y
        )

        # Define Y
        Y = np.zeros((f, M1, M1), dtype=complex)
        _in = 2 * OverlapTools.inner(LM[:, :, :, :, :], LM[:, :, :, :, :], x, y)
        Y[:, :, :] = np.repeat(
            _in[:, np.newaxis, :], M1, axis=1
        )  # Might need to swap these, but i believe row is j and column is i, should change as a function of i

        # Assign new product to R
        R[:, _i1_, _i2_] = 1 / Y * (T @ X)

        return R
