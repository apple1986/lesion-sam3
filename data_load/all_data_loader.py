from data_load.brats_builder import load_data_brats_mets
from data_load.busi_builder import load_data_busi
from data_load.isic2018_builder import load_data_isic2018
from data_load.islse22_builder import load_data_islse22
from data_load.kvasir_builder import load_data_KVASIR
from data_load.msd_colon_builder import load_data_msd_colon
from data_load.msd_kidney_builder import load_data_msd_kidney
from data_load.msd_liver_builder import load_data_msd_liver
from data_load.msd_lung_builder import load_data_msd_lung
from data_load.msd_pancreas_builder import load_data_msd_pancreas
from data_load.sts51_builder import load_data_sts51
from data_load.utswdataset_builder import load_data

def all_datasets():

    DATASET_REGISTRY = [



        # -------- PET-------- # /CT 
        ("sts", load_data_sts51,
        {"image": ("curPT"), "mask": "curGT"}, "PETCT"), # "curCT",

        # -------- MRI/CT/PE/etc --------
        ("isles", load_data_islse22,
         {"image": ("dwi",), "mask": "curGT"}, "DWI"),

        ("brats", load_data_brats_mets,
         {"image": ("curMR",), "mask": "curGT"}, "v2"),

        # -------- CT --------
        ("colon", load_data_msd_colon,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("kidney", load_data_msd_kidney,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("liver", load_data_msd_liver,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("lung", load_data_msd_lung,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("pancreas", load_data_msd_pancreas,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),


        # -------- 2D datasets --------
        ("busi", load_data_busi, None, "Ultrasound"),
        ("isic", load_data_isic2018, None, "Dermoscopy"),
        ("kvasir", load_data_KVASIR, None, "Endoscopy"),
    ]

    return DATASET_REGISTRY


def all_testsets():

    DATASET_REGISTRY = [



        # -------- MRI/CT/PE/etc --------
        ("brats", load_data_brats_mets,
         {"image": ("curMR",), "mask": "curGT"}, "v2"),

        ("pancreas", load_data_msd_pancreas,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),


        ("isles", load_data_islse22,
         {"image": ("dwi",), "mask": "curGT"}, "DWI"),

        # -------- CT --------
        ("colon", load_data_msd_colon,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("kidney", load_data_msd_kidney,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("liver", load_data_msd_liver,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

        ("lung", load_data_msd_lung,
         {"image": ("curCT",), "mask": "curGT"}, "CT"),

            # -------- PET/CT --------
        ("sts", load_data_sts51,
        {"image": ("curCT", "curPT"), "mask": "curGT"}, "PETCT"),
        # -------- 2D datasets --------
        ("busi", load_data_busi, None, "Ultrasound"),
        ("kvasir", load_data_KVASIR, None, "Endoscopy"),
        ("isic", load_data_isic2018, None, "Dermoscopy"),
    ]

    return DATASET_REGISTRY


def all_utsw_set():

    DATASET_REGISTRY = [

        # -------- UTSW-in-house data --------
        ("utsw_abd", load_data,
         {"image": ("curMR",), "mask": "curGT"}, "abd_all", "itv",),

        ("utsw_brain", load_data,
         {"image": ("curMR",), "mask": "curGT"}, "brain_ctv", "brain_ctv",),         

        ("utsw_liver", load_data,
         {"image": ("curMR",), "mask": "curGT"}, "liver_all", "itv",),      



        # # -------- MRI/CT/PE/etc --------

        # ("brats", load_data_brats_mets,
        #  {"image": ("curMR",), "mask": "curGT"}, "v2"),

        # ("isles", load_data_islse22,
        #  {"image": ("curMR",), "mask": "curGT"}, "None"),

        # # -------- CT --------
        # ("colon", load_data_msd_colon,
        #  {"image": ("curCT",), "mask": "curGT"}, "CT"),

        # ("kidney", load_data_msd_kidney,
        #  {"image": ("curCT",), "mask": "curGT"}, "CT"),

        # ("liver", load_data_msd_liver,
        #  {"image": ("curCT",), "mask": "curGT"}, "CT"),

        # ("lung", load_data_msd_lung,
        #  {"image": ("curCT",), "mask": "curGT"}, "CT"),

        # ("pancreas", load_data_msd_pancreas,
        #  {"image": ("curCT",), "mask": "curGT"}, "CT"),




        # # -------- PET/CT --------
        # ("sts", load_data_sts51,
        # {"image": ("curCT", "curPT"), "mask": "curGT"}, "PETCT"),

        # # -------- 2D datasets --------
        # ("busi", load_data_busi, None, "Ultrasound"),
        # ("isic", load_data_isic2018, None, "Dermoscopy"),
        # ("kvasir", load_data_KVASIR, None, "Endoscopy"),
    ]

    return DATASET_REGISTRY


