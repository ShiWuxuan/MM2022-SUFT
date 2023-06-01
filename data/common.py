import random

def arugment(img, depth, gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    if hflip: 
        img = img[:, ::-1, :].copy()
        depth = depth[:, ::-1, :].copy()
        gt = gt[:, ::-1, :].copy()
    if vflip: 
        img = img[::-1, :, :].copy()
        depth = depth[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()

    return img, depth, gt


def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    # lr_tx = tx // scale
    # lr_ty = ty // scale

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]