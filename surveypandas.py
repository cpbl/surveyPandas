#!/usr/bin/python

"""
Here we define a "pandas + dict" dataformat, in which a DataFrame and a dict together specify the values as well as the variable descriptions and value descriptions for (survey) data.
The DataFrame is a Pandas DataFrame.
 The dict is nested, with format:
{'_dataset_description': string,
variablename: {'description': string,
               'values': { int1: string,
                           int2: string, 
                           ....
                          }
              }
}

This means that variables which have the same set of values descriptions list those descriptions redundantly.

We save the data and labels together in a pickled file with name ending in '.pysurvey'

"""

def load_text_data_using_SAS_syntax(sasfile='/home/cpbl/rdc/inputData/GSS27/Syntax_Syntaxe/GSS27SI_PUMF.sas',datafile='/home/cpbl/rdc/inputData/GSS27/Data/C27PUMF.txt',outfilePandas=None,outfileStata=None):
    """
    Statistics Canada gives "formatted" text data files along with SAS "cards" (code/syntax) to process it.
    This routine reads their SAS definitions of the variable columns, their labels, ad the value labels for each variable. 
    It saves the dataset in pandas+dict form.
    It also creates Stata code for loadig the labels info.
    """
    import re
    from codecs import open
    import pandas as pd
    sas= open(sasfile,'rt','utf8').read()
    sfileformat=re.findall('INPUT\n(.*?);\n *\n',sas,re.DOTALL)
    assert len(sfileformat)==1

    
    # Find descriptions of variables
    vardescs=re.findall('label\n(.*?);\n *\n',sas,re.DOTALL)
    assert len(vardescs)==1
    labels=dict([[a.strip(),{'description':b.strip(' "')}] for a,b, in [LL.split('=') for LL in vardescs[0].split('\n') if LL.strip()]])
    # Eventually, put both variable labels and value labels into this dict, above: labels
    

    # Find VALUELABEL names' association with variable names:
    svallabelnames=re.findall('format\n(.*?);\n *\n',sas,re.DOTALL)
    assert len(svallabelnames)==1
    listValuesNames=[re.split(' +',LL.strip(' .')) for LL in svallabelnames[0].split('\n') if LL.strip()]

    # Rather than grab the section for the value label definitions as I have above for other sections, just grab all definitions directly:
    """ They look like:
        VALUE     YESN03F
                    1 = "Yes - Specify"
                    2 = "No"
                    6 = "Valid skip"
                    7 = "Don't know"
                    8 = "Refusal"
                    9 = "Not stated"
                    ;
    """
    valuedefs=re.findall(' +VALUE +([^ ]*)\n(.*?) +;\n',sas,re.DOTALL)
    valuelabels,intvaluelabels={},{}
    for vdname,defs in valuedefs:
        onedef=[LL.split('=') for LL in defs.split('\n') if LL.strip()]
        vals=dict([[a[0].strip(),a[1].strip(' "')] for a in onedef])
        valuelabels[vdname]=vals
        #assert all([int(a[0])==float(a[0]) for a in onedef])
        intvals=dict([[float(a[0].strip()),a[1].strip(' "')] for a in onedef])
        intvaluelabels[vdname]=intvals
        
    # Now, although this means much duplication of the values information, that is still small compared with data size: so assign values to each variable, ie getting rid of the valuelabel-name:
    for thevar, thevln in listValuesNames:
        print thevar, thevln
        labels[thevar]['values']=intvaluelabels[thevln]

    # Find columns for each variable
    varfmts=sfileformat[0].split('\n')
    # one line looks like:
    # '    @         1     RECID                                  5.  /*      1 -      5 */'
    # '    @         6     MYFLOAT                                  5.2  /*      6 -      10 */'
    # format of the variable in the form w.d (where w indicates the total width of the variable,  including any signs and decimal points, and d indicates the number of places after the decimal).
    
    fmts=[re.findall(r'@ +(\d*) +(\w*) +(\d*)\.',oneline)[0] for oneline in varfmts if oneline.strip()]

    df=pd.read_fwf(datafile, colspecs='infer', widths=[int(rr[2]) for rr in fmts],names=[rr[1] for rr in fmts],header=None)

    if outfilePandas is not None:
        assert outfilePandas.endswith('.pysurvey')
        from cpblUtilities import shelfSave
        shelfSave(outfilePandas,[df,labels])

    if outfileStata is not None:
        import pystata as stata
        """ 
        Generate code to define all the labels in Stata
        """
        fn,ext=os.path.splitext(outfileStata)
        rawfile=fn+'_raw'+ext
        df.to_stata(rawfile)

        myCodebook=stata.stataCodebookClass(codebook=labels)
        foiu
        outstata=stata.stataLoad(rawfile)+"""
        """
        """ I have two equivalent functions still . :(
 createValueLabels(self,lookupDict,varname=None,labelname=None):
 assignLabelsInStata(self,autofindBooleans=True,missing=None,onlyVars=None,valuesOnly=False):
        """
        foooiiii



if __name__ == '__main__':
    #pass
    # parseSimultaneousQuantileRegression()
    from cpblDefaults import *
    load_text_data_using_SAS_syntax(sasfile='/home/cpbl/rdc/inputData/GSS27/Syntax_Syntaxe/GSS27SI_PUMF.sas',datafile='/home/cpbl/rdc/inputData/GSS27/Data/C27PUMF.txt',outfilePandas=WP+'test.pysurvey',outfileStata=WP+'GSS27')
