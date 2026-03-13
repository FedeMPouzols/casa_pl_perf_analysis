import os


sizes_fname = "asdm_sizes_both_plwg_files_and_du.csv"
sizes_fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), sizes_fname)


def load_asdm_sizes(sizes_fpath: str):
    asdm_sizes = {}

    with open(sizes_fpath, "r") as file:
        size_lines = file.readlines()
        # Very weak loading/parsing
        for line in size_lines:
            if line.strip().startswith("#"):
                continue
            parts = line.split(",")
            size, asdm_uid = parts[0].strip(), parts[1].strip()
            asdm_sizes[asdm_uid] = int(size) / 1024.0 / 1024.0

    return asdm_sizes


def get_mous_asdms_size(run_info: dict) -> int:
    mous_size = -1
    try:
        mous_size = sum(
            [_asdm_sizes[asdm_uid] for asdm_uid in run_info["_eb_uids_all"]]
        )
    except KeyError as exc:
        print(
            " WARNING: no size available for mous: {0}. Exception: {1}".format(
                run_info["_mous"], exc
            )
        )

    return mous_size


_asdm_sizes = load_asdm_sizes(sizes_fpath)
