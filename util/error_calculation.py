"""Utility methods for gaze angle and error calculations."""
import cv2 as cv
import numpy as np

radians_to_degrees = 180.0 / np.pi

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def classFeature2value(input, range_angle, num_class):
    input = np.asarray(input)
    input = input.astype('float')
    max_index = np.argmax(input, axis=1)
    max_index = max_index / num_class * range_angle
    output = max_index - (range_angle / 2.0)
    return output

def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return np.mean(angular_error(a, b))

def angular_error_zhang(pred_theta, pred_phi, gt_theta, gt_phi):
    data_x = -1 * np.cos(pred_theta) * np.sin(pred_phi)
    data_y = -1 * np.sin(pred_theta)
    data_z = -1 * np.cos(pred_theta) * np.cos(pred_phi)
    norm_data = np.sqrt(data_x * data_x + data_y * data_y + data_z * data_z)

    label_x = -1 * np.cos(gt_theta) * np.sin(gt_phi)
    label_y = -1 * np.sin(gt_theta)
    label_z = -1 * np.cos(gt_theta) * np.cos(gt_phi)
    norm_label = np.sqrt(label_x * label_x + label_y * label_y + label_z * label_z)

    angle_value = (data_x * label_x + data_y * label_y + data_z * label_z) / (norm_data * norm_label)

    angle_value = np.clip(angle_value, a_min=-0.99999, a_max=0.99999)

    output = np.arccos(angle_value) / np.pi * 180.0

    return output

