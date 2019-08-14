# embedded here for now, to avoid input files floating around, but this must move out!
# Keys: MOUS ID. The values are in GBs, summed for all the ASDMs included in that MOUS.
mous_sizes = {'uid___A001_X2fa_X187': 12,
              'uid___A002_Xc24c3f_X287': 1.435,
              'uid___A001_X121_X524': 5.8,
              'uid___A002_Xc24c3f_X1b3': 1.6,
              'uid___A001_X1272_X23': 1.4,
              'uid___A001_X2fb_X2b9': 45,
              'uid___A001_X87d_Xc7b': 24,
              'uid___A001_X87c_X61c': 9.6,
              'uid___A001_X88f_X128': 7.1,
              'uid___A002_Xc24c3f_X154': 13,
              'uid___A002_Xc24c3f_Xb3': 67,
              'uid___A001_X87c_X453': 23,
              'uid___A002_Xc24c3f_Xcd': 114,
              'uid___A002_Xc24c3f_Xd6': 232,
              'uid___A001_X2fa_Xf0': 104,
              'uid___A001_Xbd4641_X23': 28,
              'uid___A001_X12a_X40': 5.5,
              'uid___A001_X1271_Xa': 5.8,
              'uid___A002_Xc24c3f_X2dd': 15,
              'uid___A001_X2fb_X2bb': 3.6,
              'uid___A002_Xc24c3f_X219': 8.9,
              'uid___A002_Xc24c3f_X1c2': 1.934,
              'uid___A001_X879_X6d1': 2.4,
              'uid___A002_Xad2d3f_X17': 12,
              'uid___A002_Xc24c3f_X199': 0.555,
              'uid___A001_X88f_X250': 36,
              'uid___A002_Xc24c3f_X2ba': 1.2,
              'uid___A002_Xc24c3f_X1a4': 3.4,
              'uid___A002_Xc24c3f_X1e9': 1.1,
              'uid___A001_X2f6_X265': 46,
              'uid___A001_X88a_Xb': 3.1,
              'uid___A002_Xc24c3f_X203': 0.677,
              'uid___A001_X885_X3ff': 1.4,
              'uid___A002_Xc24c3f_Xc6': 13,
              'uid___A001_X879_X426': 21.5,
              'uid___A001_X2fe_X116': 3.3,
              'uid___A001_X2fb_X112': 102,
              'uid___A002_Xc24c3f_X245': 59,
              'uid___A002_Xc24c3f_X15f': 3.8,
              'uid___A001_X2f7_X1eb': 17,
              'uid___A001_X87c_X10d': 7.5,
              'uid___A002_Xc24c3f_X8d': 0.387,
              'uid___A002_Xc24c3f_Xbc': 20,
              'uid___A002_Xac065b_X48': 30,
              'uid___A001_X879_X1cf': 63,
              'uid___A001_X87c_X81d': 0.998,
              'uid___A001_X2fa_X1fc': 6.1,
              'uid___A001_X2d8_X2c5': 0.546,
              'uid___A001_X2d8_X158': 9.0,
              'uid___A002_Xb5270a_X1c': 3.3,

              'uid___A001_X2d8_X317': 305,
              'uid___A001_X2d2_X6e': 126,
              'uid___A002_Xb020f7_X9e2': 9,
              'uid___A002_Xb65918_X10b': 1.6,
              'uid___A002_Xb66884_X1258': 1.6,
              # CAS-11578, PRTSPR-34036: 142 targets, 5 spws, 710 cubes
              'uid___A001_X87d_X29e': 130, # ASDM uid___A002_Xcde1cf_X3966
              # CAS-11660, not the most demanding MOUS in some respects:
              # 4 EBs and 68 targets, but only 4 TDM spws.
              'uid___A001_X12a3_X7b2': 53, # 4 ASDMs
              # ? add 2017.1.00129.S_2018_08_18T22_31_45.201 MOUS uid://A001/X1284/X505
              # (>16d on old_lustre)

              # 2017
              'uid___A001_X1284_X15d3': 7.5, # 5
              'uid___A001_X12a2_X293': 17, # 9,
              'uid___A001_X1284_X505': 82, # 33 EBs (35 targets)
              'uid___A001_X1296_X8ef': 14,
              'uid___A001_X1273_X2e3': 19, # 9
              'uid___A001_X1296_X7b1': 31,
              'uid___A001_X12a3_X3be': 320, # 3
              'uid___A001_X1288_Xf93': 2,
              'uid___A001_X1284_X95f': 11,              
              'uid___A001_X1296_X1d1': 13,
              'uid___A001_X12fc_X108' : 1.5,
              # 'uid___A001_X1288_X647': 262, # 5 EBs, # CAS-12166

              # E2E6
              'uid___A002_Xcff05c_Xc2': 1.3,
              'uid___A002_Xcff05c_X1fb': 0.8,
              'uid___A001_X131f_X11': 8.3
}
# uid___A001_X133d_X3987: 25 uid___A002_Xd3c7c2_X5cc (12 GB) + uid___A002_Xd3e89f_X8c97 (13 GB)



# Key: MOUS ID. Value: short name for test dataset.
mous_short_names = {'uid___A001_X2fa_X187': 'T024',
                    'uid___A002_Xc24c3f_X287': 'T037',
                    'uid___A001_X121_X524': 'T059',
                    'uid___A002_Xc24c3f_X1b3': 'T039',
                    'uid___A001_X1272_X23': 'T033',
                    'uid___A001_X2fb_X2b9': 'T008',
                    'uid___A001_X87d_Xc7b': 'T013',
                    'uid___A001_X87c_X61c': 'T012',
                    'uid___A001_X88f_X128': 'T016',
                    'uid___A002_Xc24c3f_X154': 'T050',
                    'uid___A002_Xc24c3f_Xb3': 'T051',
                    'uid___A001_X87c_X453': 'T014',
                    'uid___A002_Xc24c3f_Xcd': 'T045',
                    'uid___A002_Xc24c3f_Xd6': 'T048',
                    'uid___A001_X2fa_Xf0': 'T027',
                    'uid___A001_Xbd4641_X23': 'T004',
                    'uid___A001_X12a_X40': 'T058',
                    'uid___A001_X1271_Xa': 'T043',
                    'uid___A002_Xc24c3f_X2dd': 'T053',
                    'uid___A001_X2fb_X2bb': 'T020',
                    'uid___A002_Xc24c3f_X219': 'T042',
                    'uid___A002_Xc24c3f_X1c2': 'T035',
                    'uid___A001_X879_X6d1': 'T015',
                    'uid___A002_Xad2d3f_X17': 'T026',
                    'uid___A002_Xc24c3f_X199': 'T032',
                    'uid___A001_X88f_X250': 1,
                    'uid___A002_Xc24c3f_X2ba': 'T036',
                    'uid___A002_Xc24c3f_X1a4': 'T034',
                    'uid___A002_Xc24c3f_X1e9': 'T040',
                    'uid___A001_X2f6_X265': 'T002',
                    'uid___A001_X88a_Xb': 'T018',
                    'uid___A002_Xc24c3f_X203': 'T038',
                    'uid___A001_X885_X3ff': 'T019',
                    'uid___A002_Xc24c3f_Xc6': 'T046',
                    'uid___A001_X879_X426': 'T003',
                    'uid___A001_X2fe_X116': 'T022',
                    'uid___A001_X2fb_X112': 'T017',
                    'uid___A002_Xc24c3f_X245': 'T049',
                    'uid___A002_Xc24c3f_X15f': 'T044',
                    'uid___A001_X2f7_X1eb': 'T023',
                    'uid___A001_X87c_X10d': 'T010',
                    'uid___A002_Xc24c3f_X8d': 'T009',
                    'uid___A002_Xc24c3f_Xbc': 'T052',
                    'uid___A002_Xac065b_X48': 'T025',
                    'uid___A001_X879_X1cf': 'T060',
                    'uid___A001_X87c_X81d': 'T061',
                    'uid___A001_X2fa_X1fc': 'T062=TEC003', # also 
                    'uid___A001_X2d8_X2c5': 'T063=TEC004',
                    'uid___A001_X2d8_X158': 'T064=TEC001',
                    'uid___A002_Xb5270a_X1c': 'T065=TEC002',
                    'uid___A001_X87d_X29e': 'CAS-11578, PRTSPR-34036',
                    'uid___A001_X12a3_X7b2': 'CAS-11660',
                    'uid___A001_X2d8_X317': '---',   # fails mitigation
                    'uid___A001_X2d2_X6e': '---' ,  # fails mitigation

                    # 2017.
                    'uid___A001_X1284_X15d3': '2017.1.00015.S',
                    'uid___A001_X12a2_X293': '2017.1.00019.S',
                    'uid___A001_X1284_X505': '2017.1.00129.S',
                    # 'uid___A001_X1288_X647': '2017.1.00138.S', # CAS-12166
                    'uid___A001_X1296_X8ef': '2017.1.00236.S',
                    'uid___A001_X1273_X2e3': '2017.1.00271.S',
                    'uid___A001_X1296_X7b1': '2017.1.00884.S',
                    'uid___A001_X12a3_X3be': '2017.1.00983.S',
                    'uid___A001_X1288_Xf93': '2017.1.01053.S',
                    'uid___A001_X1284_X95f': '2017.1.01085.S',
                    'uid___A001_X1296_X1d1': '2017.1.01355.L',
                    'uid___A001_X12fc_X108': '2017.A.00044.S',
                    # E2E6.
                    'uid___A002_Xcff05c_Xc2': 'E2E6.1.00018.S',
                    'uid___A002_Xcff05c_X1fb': 'E2E6.1.00035.S',
                    'uid___A001_X131f_X11': 'E2E6.1.00051.S'
}

ebs_cnt = {'uid___A001_X2fa_X187': 1,
                    'uid___A002_Xc24c3f_X287': 2,
                    'uid___A001_X121_X524': 1,
                    'uid___A002_Xc24c3f_X1b3': 1,
                    'uid___A001_X1272_X23': 1,
                    'uid___A001_X2fb_X2b9': 1,
                    'uid___A001_X87d_Xc7b': 1,
                    'uid___A001_X87c_X61c': 1,
                    'uid___A001_X88f_X128': 4,
                    'uid___A002_Xc24c3f_X154': 1,
                    'uid___A002_Xc24c3f_Xb3': 1,
                    'uid___A001_X87c_X453': 1,
                    'uid___A002_Xc24c3f_Xcd': 1,
                    'uid___A002_Xc24c3f_Xd6': 1,
                    'uid___A001_X2fa_Xf0': 1,
                    'uid___A001_Xbd4641_X23': 1,
                    'uid___A001_X12a_X40': 1,
                    'uid___A001_X1271_Xa': 1,
                    'uid___A002_Xc24c3f_X2dd': 1,
                    'uid___A001_X2fb_X2bb': 3,
                    'uid___A002_Xc24c3f_X219': 1,
                    'uid___A002_Xc24c3f_X1c2': 2,
                    'uid___A001_X879_X6d1': 3,
                    'uid___A002_Xad2d3f_X17': 1,
                    'uid___A002_Xc24c3f_X199': 2,
                    'uid___A001_X88f_X250': 1,
                    'uid___A002_Xc24c3f_X2ba': 1,
                    'uid___A002_Xc24c3f_X1a4': 1,
                    'uid___A002_Xc24c3f_X1e9': 1,
                    'uid___A001_X2f6_X265': 1,
                    'uid___A001_X88a_Xb': 2,
                    'uid___A002_Xc24c3f_X203': 1,
                    'uid___A001_X885_X3ff': 1,
                    'uid___A002_Xc24c3f_Xc6': 'T046',
                    'uid___A001_X879_X426': 5,
                    'uid___A001_X2fe_X116': 3,
                    'uid___A001_X2fb_X112': 2,
                    'uid___A002_Xc24c3f_X245': 1,
                    'uid___A002_Xc24c3f_X15f': 1,
                    'uid___A001_X2f7_X1eb': 1,
                    'uid___A001_X87c_X10d': 4,
                    'uid___A002_Xc24c3f_X8d': 1,
                    'uid___A002_Xc24c3f_Xbc': 1,
                    'uid___A002_Xac065b_X48': 1,
                    'uid___A001_X879_X1cf': 1,
                    'uid___A001_X87c_X81d': 1,
                    'uid___A001_X2fa_X1fc': 1, # also 
                    'uid___A001_X2d8_X2c5': 2,
                    'uid___A001_X2d8_X158': 1,
                    'uid___A002_Xb5270a_X1c': 'T065=TEC002',
                    'uid___A001_X87d_X29e': 'CAS-11578, PRTSPR-34036',
                    'uid___A001_X12a3_X7b2': 'CAS-11660',
                    'uid___A001_X2d8_X317': '---',   # fails mitigation
                    'uid___A001_X2d2_X6e': '---' ,  # fails mitigation

                    # 2017.
                    'uid___A001_X1284_X15d3': 5,
                    'uid___A001_X12a2_X293': 9,
                    'uid___A001_X1284_X505': 33,
                    # 'uid___A001_X1288_X647': '2017.1.00138.S', # CAS-12166
                    'uid___A001_X1296_X8ef': 1,
                    'uid___A001_X1273_X2e3': 9,
                    'uid___A001_X1296_X7b1': 2,
                    'uid___A001_X12a3_X3be': 3,
                    'uid___A001_X1288_Xf93': 1,
                    'uid___A001_X1284_X95f': 1,
                    'uid___A001_X1296_X1d1': 1,
                    'uid___A001_X12fc_X108': 1,
                    # E2E6.
                    'uid___A002_Xcff05c_Xc2': 1,
                    'uid___A002_Xcff05c_X1fb': 1,
                    'uid___A001_X131f_X11': 1
}

def load_asdm_sizes(fname, verbose=False):
    import os
    import re

    ids_re = 'MOUS_(uid___.+_.+)/rawdata/(uid___[a-zA-Z0-9_]+)'
    sizes = {}
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(fpath, 'r') as ifile:
        if verbose:
            print('fpath: {}'.format(fpath))
        for line in ifile:
            if verbose:
                print('ASDM sizes line: {}'.format(line))
            size_str, path = line.split()
            # from KB to GB:
            eb_size = int(size_str) / float(1024.) / 1024
            ids_match = re.search(ids_re, path)
            if ids_match:
                mous = ids_match.group(1)
                eb_uid = ids_match.group(2)
                if verbose:
                    print('Found: MOUS {}, EB {}, size: {:.2f}'.
                          format(mous, eb_uid, eb_size))

                if mous in sizes:
                    sizes[mous].append(eb_size)
                else:
                    sizes[mous] = [eb_size]
            else:
                print('WARNING: failed to parse path: {}'.format(path))

    ebs_cnt = {}
    for key in sizes:
        ebs_cnt[key] = len(sizes[key])
        sizes[key] = sum(sizes[key])
    print('Loaded asdm sizes from {}. Final sizes: {}'.format(fpath, sizes))

    return sizes, ebs_cnt


# ASDM sizes calculated from the PLWG 2019 calibration benchmark
_plwg_cal_asdm_sizes = 'asdm_sizes_plwg_Calonly_r42517_c5.6.18.txt'
# ASDM sizes compiled after that
_more_asdm_sizes = 'asdm_sizes_cv_misc.txt'
_files_sizes = [_plwg_cal_asdm_sizes, _more_asdm_sizes]
for sfile in _files_sizes:
    __loaded_plwg_cal_mous_sizes, __loaded_ebs_cnt = load_asdm_sizes(sfile)
    mous_sizes.update(__loaded_plwg_cal_mous_sizes)
    ebs_cnt.update(__loaded_ebs_cnt)

def get_asdms_size(mous):
    if mous not in mous_sizes:
        print(' WARNING: didn\'t find ASDM size for this MOUS: {}'.format(mous))
    return mous_sizes.get(mous, 0)
