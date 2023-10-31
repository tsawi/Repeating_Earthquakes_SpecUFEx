# Repeating_Earthquakes_SpecUFEx

This code accompanies the article "Detecting repeating earthquakes on the San Andreas Fault with unsupervised machine-learning of spectrograms" by Sawi et al., 2023 (in revision).

Below is the project file directory for generating and clustering fingerprints to find repeating earthquake sequences on a 10-km segment of the San Andreas Fault about 80 km northwest of Parkfield, California. Seismic waveforms can be accessed at https://ddrt.ldeo.columbia.edu (Waldhauser et al., 2009) and the Northern California Earthquake Data Center (NCEDC) at https://ncedc.org/.


1. **Create catalog for study area and time period:** Execute the code blocks in the Jupyter Notebook `notebooks/0_BuildInitialCatalog.ipynb` to trim the full catalog (`catalogs/NCA_REPQcat_20210919_noMeta_v2.csv`) to `NCADDiff_5km_2019_500mFault_WS21.csv`.

2. **Download waveforms from 7 stations:** Use the catalog to download waveforms from the following 7 stations: BAV, BVL, BSG, BRV, BPI, BEM, BBN.

3. **Edit .yaml file and create copies:** Edit the `.yaml` file `notebooks/SA_REQS_BSG.yaml` to reflect correct file paths. Then, copy that `.yaml` file six times, replacing the station names with those of the other six stations.

4. **Download waveform data for each station:** Using the `NCADDiff_5km_2019_500mFault_WS21.csv` catalog, download waveform data for each station from the NDEDC, saving to the file location specified in the `.yaml` files.

5. **Create spectrograms and generate fingerprints:** Execute the code blocks in the Jupyter Notebook `notebooks/1_SpecUFExWorkflow_BSG_clean.ipynb` for each station. Output includes a catalog, "SA_REQS_BSG_cat_keep_sgram.csv," and an HDF5 of the spectrograms and fingerprints indexed by event ID.

6. **Cluster the fingerprints:** Cluster the fingerprints for each station and filter by magnitude and location criteria (i.e., earthquakes in clusters must be closely located to be considered potential RES). Execute the code blocks in the Jupyter Notebook `notebooks/2_ClusterWorkflow_BSG_v2.ipynb` for each station. Output will be a catalog, e.g., "SA_REQS_v28_BSG_updatedCat_MaxExp_WSloc_2.csv".

7. **Merge all catalogs of potential RES:** Execute the code blocks in the Jupyter Notebook `notebooks/3_Combine_Cats_HypoDD.ipynb`. The outputs are folders for each potential RES that are formatted as input for HypoDD.

8. **Relocate the individual potential RES:** Relocate the individual potential RES using HypoDD. HypoDD software can be found at [HypoDD Software](https://www.ldeo.columbia.edu/~felixw/hypoDD.html).

9. **Manually check the clusters:** Manually check the clusters for overlapping rupture areas.



Repeating_Earthquakes_SpecUFEx/
|
├── data/
│   │
│   ├── catalogs/
│   │   ├── reloc_cat_final_Rx0_1Stations_20230325.csv
│   │   ├── SA_REQS_v28_BSG_updatedCat_MaxExp_WSloc_2.csv
│   │   ├── SA_REQS_BSG_cat_keep_sgram.csv
│   │   ├── NCA_REPQcat_20210919_noMeta_v2.csv
│   │   ├── NCADDiff_5km_2019_500mFault_WS21.csv
│   ├── creep/
│   │   ├── felix_CA_Qfaults.csv
│   ├── raw/
│   │   ├── NCAeqDD.v202112.1.txt
│
├── notebooks/
│   ├── 0_BuildInitialCatalog.ipynb
│   ├── 1_SpecUFExWorkflow_BSG_clean.ipynb
│   ├── 2_ClusterWorkflow_BSG_v2.ipynb
│   ├── 3_Combine_Cats_HypoDD.ipynb
│   ├── SA_REQS_BSG.yaml
│   ├── src/
│   │   ├── notebook_functions.py
│   │   ├── f3_merge_functions.py
│   │   ├── f2_cluster_functions.py
│   │   ├── f1_spectrogram_functions.py
│
├── README.md
├── LICENSE



### Works Cited:
* NCEDC (2014), Northern California Earthquake Data Center. UC Berkeley Seismological Laboratory. Dataset. doi:10.7932/NCEDC.
* Sawi T., Waldhauser F., Holtzman B. K., Groebner N; "Detecting repeating earthquakes on the San Andreas Fault with unsupervised machine-learning of spectrograms," The Seismic Record (2023, in revision)
* Waldhauser, F. (2009). Near-real-time double-difference event location using long-term seismic archives, with application to Northern California. Bulletin of the Seismological Society of America 99, 2736–2848.
