
def align_sfm_poses(corr_dict, out_dict):
    R = corr_dict['R'] @ out_dict['R'].T
    return R