import cv2
import numpy
import torch


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def generate_skeleton(target, h, w):
    target1 = target.cpu().detach().numpy()
    preds, maxval = get_max_preds(target1)
    neck_point = (preds[:, 5, :]+preds[:, 6, :])/2
    neck_point = neck_point[:, np.newaxis, :]
    preds = np.concatenate((preds, neck_point), axis=1)
    num_clips = preds.shape[0]
    skeleton = np.zeros([num_clips, h, w, 1], dtype=np.float32)
    for i in range(num_clips):
        skeleton_i = np.zeros([h, w, 3], dtype=np.float32)
        for c in connection:
            point1 = tuple(preds[i][c[0]]*4)
            point2 = tuple(preds[i][c[1]]*4)
            cv2.line(skeleton_i, point1, point2, color=(0, 255, 0), thickness=2)
            cv2.circle(skeleton_i, point1, 1, (255, 0, 0), 3)
            cv2.circle(skeleton_i, point2, 1, (255, 0, 0), 3)
        skeleton_i = cv2.cvtColor(skeleton_i, cv2.COLOR_RGB2GRAY)
        skeleton_i = skeleton_i[:, :, np.newaxis]
        skeleton[i] = skeleton_i
    return skeleton/255.


