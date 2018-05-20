# surveypandas

Tools for managing (survey) datasets in which variables have descriptions and their possible values may each have a description (label) as well. Data are in a Pandas DataFrame, and the labels are in a nested dict. Routines for reading in, writing out to Stata (other commercial formats?); managing missing value categories; etc.

The emphasis here is on managing the "codebook" information, including for instance changing variable names while keeping the codebook updated, etc.


Example session (to do): 


    from surveypandas import read_stata, surveyDataFrame, read_pickle

    df = read_stata('WV6_Stata_v_2016_01_01.dta.gz')     # Load both the data and codebook information from a Stata file

    df.rename_columns(dict(
        A170 = 'SWL',
        X025R = 'educ3',
        ),
          inplace=True)                                  # Renames columns and corresponding codebook entries
    df.rename_columns_from_descriptions(inplace=True, skip_already_renamed = True)        # Rename remaining columns to something readable, based on their codebook descriptions

    df.set_float_values_from_negative_integers()         # Create a missing value lookup for integer columns based on the codebook
    df.set_NaN_strings(["Don't know", "Not asked", "Refused"],) # Do the same thing, but using value labels

    cols =df.grep('satis')                               # Search for all columns with this string in their name or documentation (case insensitive)
    df.dgrep('satis')                                    # Report stats and descriptions for those same columns
    df[  cols  ].describe()                              # Alternative syntax to above
    df.to_floats()[ cols ].describe()                    # Show stats on non-missing values for the columns of interest

    df.to_pickle('mydata.spandas')                       # Save data in compressed python format
    df2 = read_pickle('mydata.spandas')                  # And read it back

    df['countryName'] = df.as_labels('country')          # Use codebook to create str-value column from numeric codes

