# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.reciprocal(np.exp(-x) + 1.0)


# TODO: Vectorize this, esp for diag_only=True
def joint_probability(
    alpha: np.ndarray, beta: np.ndarray, diag_only: bool = False
) -> dict:
    """
    Computes 2D joint probability table related to marginal contributions in
    games based on d-way categorical distributions. We return a dictionary
    with:

    * "joint": Joint probability matrix (if ``diag_only == False``)
    * "diag": Diagonal of joint probability matrix (if ``diag_only == True``)
    * "marg_left": Marginal probability vector over left variable
    * "marg_right": Marginal probability vector over left variable

    Note that the matrix has a simple representation in terms of ``O(d)``
    memory and compute, and many useful functions of the full table can
    likely be computed in less than ``O(d^2)`` space and time.

    :param alpha: Input vectors, shape ``(n, d)``
    :param beta: Input vectors, shape ``(n, d)``
    :param diag_only: If ``True``, only the diagonal of the joint probability
        matrix is computed, which costs ``O(d)`` only. Defaults to ``False``
    :return: Result dictionary (see above)
    """

    if alpha.ndim == 1:
        alpha = alpha.reshape((1, -1))
    assert alpha.size == beta.size
    beta = beta.reshape(alpha.shape)
    num_cases, num_categs = alpha.shape
    assert num_categs >= 3, "Must have at least 3 categories"
    # Reorder categories so that ``delta`` is nonincreasing
    delta = alpha - beta
    # HIER!!
    order_ind = np.argsort(-delta)
    alpha_ord = alpha[order_ind]
    beta_ord = beta[order_ind]
    delta = delta[order_ind]
    # Compute :code:`log(sum(exp(alpha[:k])))` for all k<d. This is done by
    # applying :code:`accumulate` (which generalizes cumsum and cumprod) to the
    # ufunc
    #    ``(a, b) -> log(exp(a) + exp(b))``
    bar_alpha = np.logaddexp.accumulate(alpha_ord)
    bar_alpha_d = bar_alpha[-1]  # Needed for diagonal below
    bar_alpha = bar_alpha[:-1]
    bar_beta = np.logaddexp.accumulate(np.flip(beta_ord))
    bar_beta_0 = bar_beta[-1]
    bar_beta = np.flip(bar_beta[:-1])
    bar_diff = bar_beta - bar_alpha
    sigma_delta_k = sigmoid(bar_diff + delta[:-1])
    prob_diag_ord = np.concatenate(
        (
            sigma_delta_k * np.exp(beta_ord[:-1] - bar_beta),
            np.array([np.exp(alpha_ord[-1] - bar_alpha_d)]),
        )
    )
    if not diag_only:
        sigma_delta_kp1 = sigmoid(bar_diff + delta[1:])
        cvec = np.exp(-bar_alpha - bar_beta) * (sigma_delta_k - sigma_delta_kp1)
        cumsum_c_ord = np.concatenate((np.zeros(1), np.cumsum(cvec)))
    # Undo the reordering
    if not diag_only:
        cumsum_c = np.empty((num_categs,))
        cumsum_c[order_ind] = cumsum_c_ord
    prob_diag = np.empty((num_categs,))
    prob_diag[order_ind] = prob_diag_ord
    result = {
        "marg_left": np.exp(alpha - bar_alpha_d),
        "marg_right": np.exp(beta - bar_beta_0),
    }
    if diag_only:
        result["diag"] = prob_diag
    else:
        # Compose probability matrix. Note that up to here, all computations are
        # O(d)
        inv_order_ind = np.empty((num_categs,))
        inv_order_ind[order_ind] = np.arange(num_categs)
        r_shp = (-1, 1)
        s_shp = (1, -1)
        prob_mat = (
            np.reshape(np.exp(alpha), r_shp)
            * (cumsum_c.reshape(s_shp) - cumsum_c.reshape(r_shp))
            * np.reshape(np.exp(beta), s_shp)
            * (inv_order_ind.reshape(r_shp) < inv_order_ind.reshape(s_shp))
        )
        prob_mat[np.diag_indices_from(prob_mat)] = prob_diag
        result["joint"] = prob_mat
    return result


def query_max_probability_of_change(
    alpha: np.ndarray, beta: np.ndarray
) -> (float, int):
    """
    Computes the score for the query functional

    .. math::

       \mathrm{max}_s \mathbb{P}( v(S) = s, v(S\cup i) \ne s )

    :param alpha: Input vector, shape ``(d,)``
    :param beta: Input vector, shape ``(d,)``
    :return: Tuple of score value and argmax
    """
    result = joint_probability(alpha, beta, diag_only=True)
    argument = result["marg_right"] - result["diag"]
    pos = np.argmax(argument)
    return argument[pos], pos


if __name__ == "__main__":
    num_categs = 10
    alpha = np.random.normal(size=num_categs)
    beta = np.random.normal(size=num_categs)
    prob_mat = joint_probability(alpha, beta)["joint"]
    print(prob_mat)
    print(f"Sum of all entries = {np.sum(prob_mat)}")
