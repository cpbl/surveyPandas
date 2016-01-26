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

Optionally, there can also be 'labelsname' and 'prompt' fields for each variable. These contain respectively some name for the set of values; and the actual wording of the survey question, if any, behind the variable.


This means that variables which have the same set of values descriptions list those descriptions redundantly.

We save the data and labels together in a pickled file with name ending in '.pysurvey'

First function below 99% complete/tested... but then Statscan gave me Stata do files.
Still to do: 
add the infix command to read the data, as well:

 infix acc_rate 1-4 spdlimit 6-7 acc_pts 9-11 using highway.raw

or, create a .dct file which defines the format:

infix using highway
infix dictionary {
acc_rate 1-4
spdlimit 6-7
acc_pts 9-11
}

and then read it in with

infix using highway.dct


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
    def _tonumeric(ss):
        if '.' in ss: return(float(ss))
        return(int(ss))
    for vdname,defs in valuedefs:
        onedef=[LL.split('=') for LL in defs.split('\n') if LL.strip()]
        vals=dict([[a[0].strip(),a[1].strip(' "')] for a in onedef])
        valuelabels[vdname]=vals
        #assert all([int(a[0])==float(a[0]) for a in onedef])
        intvals=dict([[_tonumeric(a[0].strip()),a[1].strip(' "')] for a in onedef])
        intvaluelabels[vdname]=intvals
        
    # Now, although this means much duplication of the values information, that is still small compared with data size: so assign values to each variable, ie getting rid of the valuelabel-name:
    for thevar, thevln in listValuesNames:
        #print thevar, thevln
        labels[thevar]['values']=intvaluelabels[thevln]

    # Find columns for each variable
    varfmts=sfileformat[0].split('\n')
    # one line looks like:
    # '    @         1     RECID                                  5.  /*      1 -      5 */'
    # '    @         6     MYFLOAT                                  5.2  /*      6 -      10 */'
    # format of the variable in the form w.d (where w indicates the total width of the variable,  including any signs and decimal points, and d indicates the number of places after the decimal).
    
    fmts=[re.findall(r'@ +(\d*) +(\w*) +(\d*)\.',oneline)[0] for oneline in varfmts if oneline.strip()]

    #acc_rate 1-4 spdlimit 6-7 acc_pts 9-11
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

        codebook=surveycodebook(labels)
        survey=surveypandas(codebook=labels)
        import numpy as np
        indices=np.cumsum([1]+[int(aa[2]) for aa in fmts])
        thevars=[aa[1] for aa in fmts]
        stataout='  infix '+' '.join(['%s %d-%d'%(thevars[ii],indices[ii],indices[ii+1]-1) for ii in range(len(thevars))])+""" using """+datafile+'\n'

#        stataout+=stata.stataLoad(rawfile)
        stataout+=survey.cb.assignLabelsInStata(autofindBooleans=True,missing=None,onlyVars=None,valuesOnly=False)

        stataout+=stata.stataSave(outfileStata)
        stata.stataSystem(stataout)

###########################################################################################
###
class surveycodebook(dict):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    def __init__(self,source,fromDTA=False,fromTSV=False,fromPDF=False, loadName=None,):#  recreate=None, toLower=None,showVars=None,survey=None,version=None,stringsAreTeX=None):#*args,foo=None):  # Agh. June 2010: added "version" here this might confuse things, but there was a bug...
        """ Allow instantiation from a dict (codebook=dict) or from a Stata dataset itself (fromDTA=filename)

        myCodebook=stataCodebookClass()
        myCodebook=stataCodebookClass(fromDTA=myfilepath)
        myCodebook=stataCodebookClass(codebook=myDict)
        myCodebook=stataCodebookClass( a dict already in format of codebook)
            i.e.: {varname:    }

        """
        if source.__class__ == surveycodebook or source.__class__ == dict: 
            dict.__init__(self, source)
        return
    ################################################################
    ################################################################
    def assignLabelsInStata(self,autofindBooleans=True,missing=None,onlyVars=None,valuesOnly=False):
    ################################################################
    ################################################################
        """ 2014 June: Overwrite all variable labels and value labels based on the codebook.

        By default, also look for yes/no boolean variables, and recode them to be 1/0.

        onlyVars : Don't actually bother with output except for these variables. What's nice is that this (which must be a list), can contain '.*' to denote a wildcard (or other REs)

        Not yet implemented: 

        missing : set of values like "don't know" which should be considered missing
        valuesOnly: This will create labels for values, but it won't relabel the variables themselves

ohoh. is this sthe same as createValueLAbels?  Retire one of them!? The other doesn't do the value labels.
        """
        outs=''
        import re
        if isinstance(onlyVars,str):
            onlyVars=[onlyVars]
        for thisVar,vcb in self.items():
            if onlyVars: # Skip any definitions if not in desired list
                if thisVar not in onlyVars and re.match(onlyVars[0],thisVar) is None:
                    continue
            valueLs=vcb.get('labels',vcb.get('values',{}))
            if valueLs:
                #assert not any(['"' in alabel for aval,alabel in valueLs.items()])
                yes=[aa   for aa,bb in valueLs.items() if bb.lower()=='yes']
                no=[aa   for aa,bb in valueLs.items() if bb.lower()=='no']
                #if set(sorted([vv.lower() for vv in valueLs.values()]))==set(['yes','no']):
                if len(yes)==1 and len(no)==1 and len(valueLs)==2:
                    outs+='\ncapture noisily replace %s = %s == %d\n'%(thisVar,thisVar,yes[0])
                    self[thisVar]['labels']={1:'yes',0:'no'}
                    vcb=self[thisVar]
                #assert not any(["don't know" in aval for aval in valueLs.values()])
                valueLabelName=thisVar+'_LABEL' if 'labelsname' not in vcb else vcb['labelsname']
                outs+='\n label define '+valueLabelName+' '+' '.join(['%d "%s"'%(aval,alabel) for aval,alabel in valueLs.items()])+'\n'
                #assert 'other than ' not in '\n label define %s_LABEL'%thisVar+' '+' '.join(['%d "%s"'%(aval,alabel) for aval,alabel in valueLs.items()])+'\n'
                outs+='\n capture noisily label values %s %s\n'%(thisVar,valueLabelName)

            #assert not '"' in vcb['desc']
            desckey='description' if 'description' in vcb else 'desc'
            outs+='\n'+'*'*(not not valuesOnly)+'capture noisily label variable %s "%s"\n'%(thisVar,vcb[desckey])
        return(outs)

import pandas as pd
###########################################################################################
###
class surveypandas(pd.DataFrame):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,                   codebook=None):
        # Can I not use super() in the line below?
        pd.DataFrame.__init__(self, data=None, index=None, columns=None, dtype=None, copy=False)
        if codebook is not None:
            self.cb=surveycodebook(codebook)
                 

if __name__ == '__main__':
    #pass
    # parseSimultaneousQuantileRegression()
    from cpblDefaults import *
    load_text_data_using_SAS_syntax(sasfile='/home/cpbl/rdc/inputData/GSS27/Syntax_Syntaxe/GSS27SI_PUMF.sas',datafile='/home/cpbl/rdc/inputData/GSS27/Data/C27PUMF.txt',outfilePandas=WP+'test.pysurvey',outfileStata=WP+'GSS27')




