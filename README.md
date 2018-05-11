# surveypandas

Tools for managing (survey) datasets in which variables have descriptions and their possible values may each have a description (label) as well. Data are in a Pandas DataFrame, and the labels are in a nested dict. Routines for reading in, writing out to Stata (other commercial formats?); managing missing value categories; etc.

The emphasis here is on managing the "codebook" information, including for instance changing variable names while keeping the codebook updated, etc.


Example session (to do): 

    from surveypandas import read_stata, surveyDataFrame, read_pickle
    sdf = read_stata('WV6_Stata_v_2016_01_01.dta.gz')
    sdf.assert_unique_columns()
    sdf.rename_variables_from_descriptions()
    sdf.assert_unique_columns()
    sdf.set_float_values_from_negative_integers() 
    sdf.assert_unique_columns()
    sdf.grep('satis') # Search for all columns with this string in their name or documentation
    sdf[  sdf.grep('satis')  ].describe()

    sdf.to_pickle('mydata.spandas') # Compressed 

    sdf2 = read_pickle('mydata.spandas') 
