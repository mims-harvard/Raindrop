# PhysioNet 2012 Challenge dataset #

### Raw data###
* /rawdata/
  * Raw data: folders: set-a, set-b, set-c.  Files: Outcomes-a.txt, Outcomes-b.txt, Outcomes-c.txt
  * Data source: https://www.physionet.org/content/challenge-2012/1.0.0/

### Data parsing (process_scripts) ###
Run following scripts in turn:
* /process_scripts/
  * in the folder raw_data/set-{a,b,c} extract raw_data/set-{a,b,c}/set-{a,b,c}.zip to get text files (.txt), which are further processed with ParseData.py
  * ParseData.py : generate arr\_outcomes.npy, ts\_params.npy, static\_params.npy, and P\_list.npy
  * IrregularSampling.py: generate: extended\_static\_params.npy, PTdict\_list.npy
  * create a folder '../splits/', where next script will save files
  * Generate\_splitID.py: generate phy12\_splitX.npy where X range from 1 to 5. Only contains the IDs. Train/val/test =  8:1:1

Note: PTdict\_list.npy and arr\_outcomes.npy are the most important files.


### Processed data ###:
* /processed_data/
  * PTdict_list.npy
  * arr_outcomes.npy  
  * ts_params.npy
  * extended_static_params.npy
  
  * static_params.npy
  * P_list.npy
* /splits/
  * phy12\_splitsX.npy  where X range from 1 to 5. In splits folder: there are 5 npy files, each contains three array representing idx\_train,idx\_val,and idx\_test.
  
