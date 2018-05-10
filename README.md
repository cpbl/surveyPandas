# surveypandas

Tools for managing (survey) datasets in which variables have descriptions and their possible values may each have a description (label) as well. Data are in a Pandas DataFrame, and the labels are in a nested dict. Routines for reading in, writing out to Stata (other commercial formats?); managing missing value categories; etc.

The emphasis here is on managing the "codebook" information, including for instance changing variable names while keeping the codebook updated, etc.


Example session (to do): 

    sdf = read_stata('WV6_Stata_v_2016_01_01.dta.gz')
    sdf.assert_unique_columns()
    sdf.rename_variables_from_descriptions()
    sdf.assert_unique_columns()
    sdf.set_float_values_from_negative_integers() 
    sdf.assert_unique_columns()
    print sdf.grep('satis') # Search for all columns with this string in their name or documentation
    sdf[ sdf.grep('satis')  ].describe()

    sdf.to_pickle(paths['scratch']+'surveypandas-test-raw.spandas') # Fairly compact
    pd.DataFrame(sdf).to_pickle(paths['scratch']+'surveypandas-test-df.spandas') # Fairly compact
    with open(paths['scratch']+'surveypandas-test-cb.spandas', 'wb') as f:  pkl.dump(sdf.codebook, f)

    print (' Reading raw...')
    sdf2 = read_pickle(paths['scratch']+'surveypandas-test-raw.spandas') 
