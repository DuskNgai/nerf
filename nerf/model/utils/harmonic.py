import torch

__all__ = ["SphericalHarmonics"]


class SphericalHarmonics(object):
    # The real form of spherical harmonics,
    # see https://dl.acm.org/doi/pdf/10.1145/3197517.3201291.
    # Here we generate the coefficients by `C++` `long double` type.

    C = (
        # 0
        0.28209479177387814346,  # 1 / (2 * sqrt(pi))
        # 1
        0.48860251190291992160,  # sqrt(3) / (2 * sqrt(pi))
        0.48860251190291992160,  # sqrt(3) / (2 * sqrt(pi))
        0.48860251190291992160,  # sqrt(3) / (2 * sqrt(pi))
        # 2
        1.09254843059207907056,  # sqrt(15) / (2 * sqrt(pi))
        1.09254843059207907056,  # sqrt(15) / (2 * sqrt(pi))
        0.31539156525252000604,  # sqrt(5) / (4 * sqrt(pi))
        1.09254843059207907056,  # sqrt(15) / (2 * sqrt(pi))
        0.54627421529603953528,  # sqrt(15) / (4 * sqrt(pi))
        # 3
        0.59004358992664351035,  # sqrt(70) / (8 * sqrt(pi))
        2.89061144264055405544,  # sqrt(105) / (2 * sqrt(pi))
        0.45704579946446573615,  # sqrt(42) / (8 * sqrt(pi))
        0.37317633259011539140,  # sqrt(7) / (4 * sqrt(pi))
        0.45704579946446573615,  # sqrt(42) / (8 * sqrt(pi))
        1.44530572132027702772,  # sqrt(105) / (4 * sqrt(pi))
        0.59004358992664351035,  # sqrt(70) / (8 * sqrt(pi))
        # 4
        2.50334294179670453838,  # 3 * sqrt(35) / (4 * sqrt(pi))
        1.77013076977993053104,  # 3 * sqrt(70) / (8 * sqrt(pi))
        0.94617469575756001803,  # 3 * sqrt(5) / (4 * sqrt(pi))
        0.66904654355728916795,  # 3 * sqrt(10) / (8 * sqrt(pi))
        0.10578554691520430380,  # 3 / (16 * sqrt(pi))
        0.66904654355728916795,  # 3 * sqrt(10) / (8 * sqrt(pi))
        0.47308734787878000902,  # 3 * sqrt(5) / (8 * sqrt(pi))
        1.77013076977993053104,  # 3 * sqrt(70) / (8 * sqrt(pi))
        0.62583573544917613459,  # 3 * sqrt(35) / (16 * sqrt(pi))
        # 5
        0.65638205684017010284,  # 3 * sqrt(154) / (32 * sqrt(pi))
        8.30264925952416511616,  # 3 * sqrt(385) / (4 * sqrt(pi))
        0.48923829943525038767,  # sqrt(770) / (32 * sqrt(pi))
        4.79353678497332375474,  # sqrt(1155) / (4 * sqrt(pi))
        0.45294665119569692130,  # sqrt(165) / (16 * sqrt(pi))
        0.11695032245342359643,  # sqrt(11) / (16 * sqrt(pi))
        0.45294665119569692130,  # sqrt(165) / (16 * sqrt(pi))
        2.39676839248666187737,  # sqrt(1155) / (8 * sqrt(pi))
        0.48923829943525038767,  # sqrt(770) / (32 * sqrt(pi))
        2.07566231488104127904,  # 3 * sqrt(385) / (16 * sqrt(pi))
        0.65638205684017010284,  # 3 * sqrt(154) / (32 * sqrt(pi))
        # 6
        1.36636821038382864392,  # sqrt(6006) / (32 * sqrt(pi))
        2.36661916223175203180,  # 3 * sqrt(2002) / (32 * sqrt(pi))
        2.01825960291489663696,  # 3 * sqrt(91) / (8 * sqrt(pi))
        0.92120525951492349912,  # sqrt(2730) / (32 * sqrt(pi))
        0.92120525951492349912,  # sqrt(2730) / (32 * sqrt(pi))
        0.58262136251873138879,  # sqrt(273) / (16 * sqrt(pi))
        0.06356920226762842593,  # sqrt(13) / (32 * sqrt(pi))
        0.58262136251873138879,  # sqrt(273) / (16 * sqrt(pi))
        0.46060262975746174956,  # sqrt(2730) / (64 * sqrt(pi))
        0.92120525951492349912,  # sqrt(2730) / (32 * sqrt(pi))
        0.50456490072872415924,  # 3 * sqrt(91) / (32 * sqrt(pi))
        2.36661916223175203180,  # 3 * sqrt(2002) / (32 * sqrt(pi))
        0.68318410519191432196,  # sqrt(6006) / (64 * sqrt(pi))
        # 7
        0.70716273252459617822,  # 3 * sqrt(715) / (64 * sqrt(pi))
        5.29192132360380044402,  # 3 * sqrt(10010) / (32 * sqrt(pi))
        0.51891557872026031976,  # 3 * sqrt(385) / (64 * sqrt(pi))
        4.15132462976208255808,  # 3 * sqrt(385) / (8 * sqrt(pi))
        0.15645893386229403365,  # 3 * sqrt(35) / (64 * sqrt(pi))
        0.44253269244498263276,  # 3 * sqrt(70) / (32 * sqrt(pi))
        0.09033160758251731423,  # sqrt(105) / (64 * sqrt(pi))
        0.06828427691200494191,  # sqrt(15) / (32 * sqrt(pi))
        0.09033160758251731423,  # sqrt(105) / (64 * sqrt(pi))
        0.22126634622249131638,  # 3 * sqrt(70) / (64 * sqrt(pi))
        0.15645893386229403365,  # 3 * sqrt(35) / (64 * sqrt(pi))
        1.03783115744052063952,  # 3 * sqrt(385) / (32 * sqrt(pi))
        0.51891557872026031976,  # 3 * sqrt(385) / (64 * sqrt(pi))
        2.64596066180190022201,  # 3 * sqrt(10010) / (64 * sqrt(pi))
        0.70716273252459617822,  # 3 * sqrt(715) / (64 * sqrt(pi))
    )

    def __init__(self, n_degrees: int) -> None:
        assert isinstance(n_degrees, int), "n_degrees must be a integer."
        assert 1 <= n_degrees <= 8, "only first 1 ~ 8 degree of spherical harmonics are supported."

        self.n_degrees = n_degrees

    def __call__(self, coeffs: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        The `xyz` should be normalized to the unit sphere before passing to this function.
        """

        x, y, z = xyz.unbind(dim=-1)
        x, y, z = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)

        tensor_out = coeffs[..., 0] * torch.full_like(x, self.C[0])
        if self.n_degrees <= 1:
            return tensor_out

        tensor_out = tensor_out \
                   - coeffs[..., 1] * self.C[1] * y \
                   + coeffs[..., 2] * self.C[2] * z \
                   - coeffs[..., 3] * self.C[3] * x
        if self.n_degrees <= 2:
            return tensor_out

        x2, y2, z2, xy, xz, yz = x * x, y * y, z * z, x * y, x * z, y * z
        tensor_out = tensor_out \
                   + coeffs[..., 4] * self.C[4] * xy               \
                   - coeffs[..., 5] * self.C[5] * yz               \
                   + coeffs[..., 6] * self.C[6] * (3.0 * z2 - 1.0) \
                   - coeffs[..., 7] * self.C[7] * xz               \
                   + coeffs[..., 8] * self.C[8] * (x2 - y2)
        if self.n_degrees <= 3:
            return tensor_out

        tensor_out = tensor_out \
                   - coeffs[...,  9] * self.C[ 9] * (3.0 * x2 - y2) * y  \
                   + coeffs[..., 10] * self.C[10] * xy * z               \
                   - coeffs[..., 11] * self.C[11] * (5.0 * z2 - 1.0) * y \
                   + coeffs[..., 12] * self.C[12] * (5.0 * z2 - 3.0) * z \
                   - coeffs[..., 13] * self.C[13] * (5.0 * z2 - 1.0) * x \
                   + coeffs[..., 14] * self.C[14] * (x2 - y2) * z        \
                   - coeffs[..., 15] * self.C[15] * (x2 - 3.0 * y2) * x
        if self.n_degrees <= 4:
            return tensor_out

        x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
        tensor_out = tensor_out \
                   + coeffs[..., 16] * self.C[16] * xy * (x2 - y2)                \
                   - coeffs[..., 17] * self.C[17] * yz * (3.0 * x2 - y2)          \
                   + coeffs[..., 18] * self.C[18] * xy * (7.0 * z2 - 1.0)         \
                   - coeffs[..., 19] * self.C[19] * yz * (7.0 * z2 - 3.0)         \
                   + coeffs[..., 20] * self.C[20] * (35.0 * z4 - 30.0 * z2 + 3.0) \
                   - coeffs[..., 21] * self.C[21] * xz * (7.0 * z2 - 3.0)         \
                   + coeffs[..., 22] * self.C[22] * (x2 - y2) * (7.0 * z2 - 1.0)  \
                   - coeffs[..., 23] * self.C[23] * xz * (x2 - 3.0 * y2)          \
                   + coeffs[..., 24] * self.C[24] * (x4 - 6.0 * x2 * y2 + y4)
        if self.n_degrees <= 5:
            return tensor_out

        xyz = x * y * z
        tensor_out = tensor_out \
                   - coeffs[..., 25] * self.C[25] * y * (5.0 * x4 - 10.0 * x2 * y2 + y4)   \
                   + coeffs[..., 26] * self.C[26] * xyz * (x2 - y2)                        \
                   - coeffs[..., 27] * self.C[27] * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0) \
                   + coeffs[..., 28] * self.C[28] * xyz * (3.0 * z2 - 1.0)                 \
                   - coeffs[..., 29] * self.C[29] * y * (21.0 * z4 - 14.0 * z2 + 1.0)      \
                   + coeffs[..., 30] * self.C[30] * z * (63.0 * z4 - 70.0 * z2 + 15.0)     \
                   - coeffs[..., 31] * self.C[31] * x * (21.0 * z4 - 14.0 * z2 + 1.0)      \
                   + coeffs[..., 32] * self.C[32] * z * (x2 - y2) * (3.0 * z2 - 1.0)       \
                   - coeffs[..., 33] * self.C[33] * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0) \
                   + coeffs[..., 34] * self.C[34] * z * (x4 - 6.0 * x2 * y2 + y4)          \
                   - coeffs[..., 35] * self.C[35] * x * (x4 - 10.0 * x2 * y2 + 5.0 * y4)
        if self.n_degrees <= 6:
            return tensor_out

        x6, y6, z6 = x2 * x4, y2 * y4, z2 * z4
        tensor_out = tensor_out \
                   + coeffs[..., 36] * self.C[36] * xy * (3.0 * x4 - 10.0 * x2 * y2 + 3.0 * y4)   \
                   - coeffs[..., 37] * self.C[37] * yz * (5.0 * x4 - 10.0 * x2 * y2 + y4)         \
                   + coeffs[..., 38] * self.C[38] * xy * (x2 - y2) * (11.0 * z2 - 1.0)            \
                   - coeffs[..., 39] * self.C[39] * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0)      \
                   + coeffs[..., 40] * self.C[40] * xy * (33.0 * z4 - 18.0 * z2 + 1.0)            \
                   - coeffs[..., 41] * self.C[41] * yz * (33.0 * z4 - 30.0 * z2 + 5.0)            \
                   + coeffs[..., 42] * self.C[42] * (231.0 * z6 - 315.0 * z4 + 105.0 * z2 - 5.0)  \
                   - coeffs[..., 43] * self.C[43] * xz * (33.0 * z4 - 30.0 * z2 + 5.0)            \
                   + coeffs[..., 44] * self.C[44] * (x2 - y2) * (33.0 * z4 - 18.0 * z2 + 1.0)     \
                   - coeffs[..., 45] * self.C[45] * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0)      \
                   + coeffs[..., 46] * self.C[46] * (x4 - 6.0 * x2 * y2 + y4) * (11.0 * z2 - 1.0) \
                   - coeffs[..., 47] * self.C[47] * xz * (x4 - 10.0 * x2 * y2 + 5.0 * y4)         \
                   + coeffs[..., 48] * self.C[48] * (x6 - 15.0 * x4 * y2 + 15.0 * x2 * y4 - y6)
        if self.n_degrees <= 7:
            return tensor_out

        xyz = x * y * z
        tensor_out = tensor_out \
                   - coeffs[..., 49] * self.C[49] * y * (7.0 * x6 - 35.0 * x4 * y2 + 21.0 * x2 * y4 - y6)    \
                   + coeffs[..., 50] * self.C[50] * xyz * (3.0 * x4 - 10.0 * x2 * y2 + 3.0 * y4)             \
                   - coeffs[..., 51] * self.C[51] * y * (13.0 * z2 - 1.0) * (5.0 * x4 - 10.0 * x2 * y2 + y4) \
                   + coeffs[..., 52] * self.C[52] * xyz * (13.0 * z2 - 3.0) * (x2 - y2)                      \
                   - coeffs[..., 53] * self.C[53] * y * (143.0 * z4 - 66.0 * z2 + 3.0) * (3.0 * x2 - y2)     \
                   + coeffs[..., 54] * self.C[54] * xyz * (143.0 * z4 - 110.0 * z2 + 15.0)                   \
                   - coeffs[..., 55] * self.C[55] * y * (429.0 * z6 - 495.0 * z4 + 135.0 * z2 - 5.0)         \
                   + coeffs[..., 56] * self.C[56] * z * (429.0 * z6 - 693.0 * z4 + 315.0 * z2 - 35.0)        \
                   - coeffs[..., 57] * self.C[57] * x * (429.0 * z6 - 495.0 * z4 + 135.0 * z2 - 5.0)         \
                   + coeffs[..., 58] * self.C[58] * z * (143.0 * z4 - 110.0 * z2 + 15.0) * (x2 - y2)         \
                   - coeffs[..., 59] * self.C[59] * x * (143.0 * z4 - 66.0 * z2 + 3.0) * (x2 - 3.0 * y4)     \
                   + coeffs[..., 60] * self.C[60] * z * (13.0 * z2 - 3.0) * (x4 - 6.0 * x2 * y2 + y4)        \
                   - coeffs[..., 61] * self.C[61] * x * (13.0 * z2 - 1.0) * (x4 - 10.0 * x2 * y2 - 5.0 * y4) \
                   + coeffs[..., 62] * self.C[62] * z * (x6 - 15.0 * x4 * y2 + 15.0 * x2 * y4 - y6)          \
                   - coeffs[..., 63] * self.C[63] * x * (x6 - 21.0 * x4 * y2 + 35.0 * x2 * y4 - 7.0 * y6)
        if self.n_degrees <= 8:
            return tensor_out

    def __len__(self) -> int:
        return (self.n_degrees) ** 2
