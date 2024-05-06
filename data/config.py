working_path = "./"
location = "data/DATA/"
subpath = working_path + location

mapACT = "act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits"
dr5_clusters = "DR5_cluster-catalog_v1.1.fits"
archieve_dr5 = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/"

mapACT_url = archieve_dr5 + 'maps/' + mapACT
dr5_clusters_url = archieve_dr5 + dr5_clusters

mapACT_out = subpath + mapACT
dr5_clusters_out = subpath + dr5_clusters

segmentation_maps_pics = working_path + "data/example/"
samples_location = working_path + location
dr5_sample_location = samples_location + "test_dr5/"

clusters_out = segmentation_maps_pics + "cl5.csv"
randoms_out = segmentation_maps_pics + "r5.csv"
stars_out = segmentation_maps_pics + "gaia5.csv"
