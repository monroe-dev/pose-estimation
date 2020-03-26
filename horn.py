import numpy as np
import transformations as tf
import kin

def absoluteOrientation(A, B):

    centroid_a = np.mean(A, axis=0)
    centroid_b = np.mean(B, axis=0)
    A_wrt_centroid = A - centroid_a
    B_wrt_centroid = B - centroid_b
    M = 0

    for a_, b_ in zip(A_wrt_centroid, B_wrt_centroid):
        M_ = np.dot(np.expand_dims(a_, axis=1), np.expand_dims(b_, axis=0))
        M = M + M_

    Sxx = M[0, 0]
    Syx = M[1, 0]
    Szx = M[2, 0]
    Sxy = M[0, 1]
    Syy = M[1, 1]
    Szy = M[2, 1]
    Sxz = M[0, 2]
    Syz = M[1, 2]
    Szz = M[2, 2]

    N = np.array([[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
                  [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
                  [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
                  [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]])

    eig_val, eig_vec = np.linalg.eig(N)
    q = eig_vec[:, np.argmax(eig_val)]

    # print('eig_val', eig_val)
    # print('max eig val', max(eig_val))
    # print('eig vec', eig_vec)
    # print('eig vec corresponding to max eig val', eig_vec[:, np.argmax(eig_val)])
    # print('norm ', np.linalg.norm(q))
    # print('q', q)

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    rot1 = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                     [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                     [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])

    rot2 = tf.quaternion_matrix(q)
    rot2 = kin.rotationFromHmgMatrix(rot2)

    v = q[1:4]
    v_out = np.outer(v, v)
    Z = np.array([[qw, -qz, qy],
                  [qz, qw, -qx],
                  [-qy, qx, qw]])
    rot3 = v_out + np.dot(Z, Z)

    rot = rot1
    t = centroid_b - np.dot(rot, centroid_a)
    return rot, t

if __name__ == '__main__':
    A = kin.generatePointSet(4, 3, 0, 10)
    T_ab = kin.homogeneousMatrix(0, 50, 30, 0, 90, 0, True)
    # A = np.array([[2,	6,	6,	0,	8],
    #               [5,	9,	2,	6,	9],
    #               [9,	8,	7,	7,	10]])
    # A = A.transpose()
    # T_ab = np.array([[0, -1, 0, 10],
    #                 [1, 0, 0, 20],
    #                 [0, 0, 1, 30],
    #                 [0, 0, 0, 1]])
    B = kin.transform(A, T_ab)
    # print('A', A)
    # f_A = np.flip(A, 0)
    # print('T_ab', T_ab)
    # print('B', B)
    # f_B = np.flip(B, 0)
    # print('flip B', f_B)
    R, t = absoluteOrientation(A, B)
    R_original = kin.rotationFromHmgMatrix(T_ab)
    t_original = kin.translationFromHmgMatrix(T_ab)
    e_ori = R_original - R
    e_pos = t_original - t
    print('e_ori', np.linalg.norm(e_ori))
    print('e_pos', np.linalg.norm(e_pos))