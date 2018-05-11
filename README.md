# surveypandas

Tools for managing (survey) datasets in which variables have descriptions and their possible values may each have a description (label) as well. Data are in a Pandas DataFrame, and the labels are in a nested dict. Routines for reading in, writing out to Stata (other commercial formats?); managing missing value categories; etc.

The emphasis here is on managing the "codebook" information, including for instance changing variable names while keeping the codebook updated, etc.


Example session (to do): 

    from surveypandas import read_stata, surveyDataFrame, read_pickle
    sdf = read_stata('WV6_Stata_v_2016_01_01.dta.gz')     # Load both the data and codebook information from a Stata file
    sdf.rename_variables_from_descriptions()              # Rename remaining columns to something readable, based on their descriptions:
    sdf.set_float_values_from_negative_integers()         # Create a missing value lookup for integer columns based on the codebook
    cols =sdf.grep('satis')                                     # Search for all columns with this string in their name or documentation (case insensitive)
    sdf.dgrep('satis')                                    # Report stats and descriptions for those same columns
    sdf[  cols  ].describe()                              # Alternative syntax to above
    sdf.to_floats()[ cols ].describe()              # Show stats on non-missing values for the columns of interest
    sdf.to_pickle('mydata.spandas')                       # Save data in compressed python format

    sdf2 = read_pickle('mydata.spandas')                  # And read it back
