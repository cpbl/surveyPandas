#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
SurveyPandas adds survey-data features to the Pandas data structure.
Survey data typically has the following extra information:
 - "Variable labels": Description of each variable
 - "Value labels": If the variable encodes discrete response options,  a description of what each value means
 - "Float values": Possibility of mapping integer values to NaN
 - Possibly a questionnaire question related to the variable

Some variables may share value labels. In this case, the actual lookups may be shared across variables or may be duplicated/redundant (ie, implementation tbd). To simplify this, the object can also contain a list of names of sets/lookups of value labels.

Classes:

surveyDataFrame : a Pandas DataFrame object augmented with survey-related data and methods
surveySeries : corresponding Series object. 

surveycodebook: 

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

n.b.: statsmodels.iolib.foreign.StataReader  seems not to be maintained; only Stata versions 8-12 can be read. Thus we use Stata text output as a version-independent way to transfer.

"""

from surveypandas_config import defaults
paths= defaults['paths']
from collections import OrderedDict
import os
import pandas as pd
from cpblUtilities import doSystem
from copy import deepcopy


def some_unicode_quotes_to_latex(str):
    subs =[
        [u"\u0027", "\\textquotesingle "],
        [u'´', u"'"],
        [u'“', "``"],
        [u'”', "''"],
        ]
    for a,b in subs:
        str=str.replace(a,b)
    return str

def treat_OrderedDict_strings_recursively(dictlike, method):
    """ Recursively treat strings in a dict or OrderedDict by applying some method to each string in each key/value. E.g. to remove some unicode
    This is for OrderedDicts. normal dicts to not accept the argument to popitem, so we would need to loop through the keys explicitly (what's wrong with that?)
    """
    def treat_value(vv):
        if isinstance(v, basestring):
            return method(vv)
        if v.__class__ in [dict, OrderedDict]:
            return treat_OrderedDict_strings_recursively(vv, method)
        return vv
    for k in  list(dictlike.keys()):
        v=dictlike.pop(k)
        dictlike[k if not isinstance(k, basestring) else method(k)] = treat_value(v)
    if 0:
        for _ in range(len(dictlike)):
            k, v = dictlike.popitem(False)
            print k
            dictlike[k if not isinstance(k, basestring) else method(k)] = treat_value(v)
    return dictlike # Needed only for recursion, since the original dict is changed in place

def test_treat_odict_strings_recursively():
    x = {1:2, 3:4, 5:6}
    print x
    treat_OrderedDict_strings_recursively(x, some_unicode_quotes_to_latex)
    print x
    x = {'234': 'oiwe', 5:23, u'Don´t know':{u'Don´t know':u'Don\xb4t know'}}
    print x
    treat_OrderedDict_strings_recursively(x, some_unicode_quotes_to_latex)
    print x
    assert "Don't know" in x
    assert "Don't know" in x["Don't know"]
    assert 5 in x
    

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
class surveycodebook(OrderedDict):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    """
        The primary dict/data of this object is an OrderedDict of information about survey data variables.

    Value labels are lookups (OrderedDicts) from integer values to strings  which can initially be shared between variables (keys of surveycodebook). However, they must be deepcopied and made independent if ever the value labels are changed for some variable.
    The same is true of Float Values, which is a lookup that can be used to allow some integers to have NaN values

    In general, where text may be used for formatted printing, LaTeX markup is used.

        self._variable_labels = {}
        self._value_labels = {}  # Maps column names to named lookups
        self._named_labels = {}  # Stores named lookups, each of which maps integer values to labels
        self._named_float_values = {} # Stores named lookups, each of which maps integer values to float values (including NaN)
        self._questions = {} # Maps column names to relevant questionnaire question / further info
    """
    
    def __init__(self,source= None):#, fromDTA=False,fromTSV=False,fromPDF=False, loadName=None,):#  recreate=None, toLower=None,showVars=None,survey=None,version=None,stringsAreTeX=None):#*args,foo=None):  # Agh. June 2010: added "version" here this might confuse things, but there was a bug...

        """ Allow instantiation from a dict (codebook=dict) or from a Stata dataset itself (fromDTA=filename)

        myCodebook=stataCodebookClass()
        myCodebook=stataCodebookClass(fromDTA=myfilepath)
        myCodebook=stataCodebookClass(codebook=myDict)
        myCodebook=stataCodebookClass( a dict already in format of codebook)
            i.e.: {varname:    }

        """


        if source.__class__ in [surveycodebook,  OrderedDict]:
            super(surveycodebook, self).__init__(source)
        elif source.__class__ == dict: # Got an unordered dict for some reason. Should do some checking on its contents: TO DO
            super(surveycodebook, self).__init__(OrderedDict(source))
            
        elif isinstance(source, basestring):
            foo =self._from_stata(source)
            #assert isinstance(foo, OrderedDict)
            print foo
            super(surveycodebook, self).__init__(foo)
        else:
            super(surveycodebook, self).__init__(source)
        print 'after init', self
            
    ################################################################
    def _from_stata(self,datafilepath,recreate=None,toLower=None, subset = None):
    ################################################################
        """
        Initialise a codebook object from a Stata file's own information. Read this via text output, rather than from Stata digital file (See pyDTA project for an attempt at that).
        If Stata's codebooks have not already been logged, or if they are older than the dta, Stata will be called to generate the text output.
        If it has, the resulting log files will simply be parsed.

        Takes a .dta stata file and generates text files containing the codebook and label list. Can be slow for big datasets.
        See the parsing function, next, for parsing said text files.

        Note: the do file and log file goes in the source dir (do file has to go there since the statasystem call puts it in the same place). The tsv goes in workingPath.

        Note: this is run even on raw versions of datasets, which is kind of useless in the case of summary statistics, since they may contain all manner of non-response numeric values still...

        subset lists a set of variables to include
        """
        from pystata import stripdtagz # Import pystata locally, since it's only one interface for surveyPandas
        from codecs import open # Overwrite open so as to always use utf8 for text files and reading Stata output

        datafilepath=stripdtagz(datafilepath)
        sourceDir,sourceName=os.path.split(datafilepath)
        assert '.' not in sourceName         # because Stata screws up filenames for choosing log name/location
        if not os.path.exists(datafilepath+'.dta.gz'): # Assumes use of gzuse rather than use. (TO DO: allow for uncompressed .dta files)
            raise Error('   ********* There is no data file '+datafilepath+' from which to make a codebook... so no DTA codebook for you! (If this is not solved by running through any Stata execution, something is wrong!)')
            return {}


        ###fxdnm=datafilepath#sourceDir+'/'+sourceName#.replace('.','_')
        CdoFileName=datafilepath+'_StataCodebook'+'.do' ## paths['working']+sourceName
        ClogFileName=datafilepath+'_StataCodebook'+'.log'

        if recreate==False:
            print ' WARNING! RECREATE SET TO "FALSE", WHICH ALLOWS CODEBOOK TO BECOME OUTDATED. SET IT BACK TO DEFAULT "NONE"'

        # Logic below: if recreate is None, then check to see if log file is outdated or not. If recreate is True or False, then that overrides.
        forceC=recreate
        if recreate in [None,True] and not os.path.exists(ClogFileName):
            print('%s does not exist. Recreating from %s.'%(ClogFileName,datafilepath+'.dta.gz'))
            forceC=True
        elif recreate in [None,True] and os.path.getmtime(ClogFileName)<os.path.getmtime(datafilepath+'.dta.gz'):
            print('%s is older than %s.  Recreating it.'%(ClogFileName,datafilepath+'.dta.gz'))
            forceC=True
        subsetString='codebook \n'
        if subset:
            subsetString="""
foreach var in  """+' '.join(subset)+""" {
capture confirm variable `var',exact
if _rc==0 {
codebook `var'
}
}
"""
        if forceC:

            print '    To create '+CdoFileName+':  '
            from pystata import stataSystem,stataLoad
            rlogfn=stataSystem("""
              clear
            set more off
            """+stataLoad(datafilepath)+"""
             """+subsetString+"""
            * DONE SUCCESSFULLY GOT TO END
            """)
            if 'DONE SUCCESSFULLY GOT TO END' in open(rlogfn,'rt', encoding='UTF8').read():
                doSystem('cp %s %s'%(rlogfn,ClogFileName))
                print " Overwrote "+ClogFileName
            else:
                print " Failed to update "+ClogFileName

        LdoFileName=datafilepath+'_StataLabelbook'+'.do' ## paths['working']+sourceName
        LlogFileName=datafilepath+'_StataLabelbook'+'.log'
        # Logic below: if recreate is None, then check to see if log file is outdated or not. If recreate is True or False, then that overrides.
        forceL=recreate
        if recreate in [None,True] and not os.path.exists(LlogFileName):
            print('%s does not exist! Recreating from %s.'%(LlogFileName,datafilepath+'.dta.gz'))
            forceL=True
        elif recreate in [None,True] and os.path.getmtime(LlogFileName)<os.path.getmtime(datafilepath+'.dta.gz'):
            print('%s is older than %s!  Recreating it.'%(LlogFileName,datafilepath+'.dta.gz'))
            forceL=True
        if forceL:

            print '    To create '+LdoFileName+':  '
            rlogfn=stataSystem("""
              clear
            set more off
            """+stataLoad(datafilepath)+"""
            labelbook
            * DONE SUCCESSFULLY GOT TO END
            """)#,filename='tmp_doMakeLabelbook')
            if 'DONE SUCCESSFULLY GOT TO END' in open(rlogfn,'rt').read():
                #import shutil
                #shutil.move(rlogfn,LdoFileName)
                doSystem('cp %s %s'%(rlogfn,LlogFileName))
                print " Overwrote "+LlogFileName
            else:
                print " Failed to update "+LlogFileName


        # Check work (and make a tsv version) by calling the following:
        # This sets self to the codebook.
        #return OrderedDict({1:2, '3':'f'})
        return self.parseStataCodebook(ClogFileName,LlogFileName,toLower=toLower)
        assert self.keys()
        return self

            
        
    ################################################################
    ################################################################
    def parseStataCodebook(self,codebookFile,labelbookFile,toLower=None):
    ################################################################
    ################################################################
        """

        This function is now meant to be used only internally.
        Just call fromStataCodebook to initialise a codebook object from a Stata file's own information.
        If it has not already been done, Stata will be called. If it has, the resulting log files will simply be parsed.


        If your dataset is well internally documented, use Stata to create a text log file of the codebook command and another of the labelbook command. Feed those files to this function to get a dict of the available variables.

    When Stata prints the variable description on multiple lines, this captures it properly.

    One issue is that the resulting codebook structure does not preserve the order of variables (which is prserverd in the codebook command in Stata. So return a separate list of variable names?

        """
        from codecs import open # Overwrite open so as to always use utf8 for text files and reading Stata output
        
        print '  Parsing stata codebook file '+codebookFile

        cbook= []
        try:
            ff=open(codebookFile,encoding='utf-8').read()
        except (UnicodeDecodeError):
            print('    ---> (Legacy?) utf-8 method Failed on codebook reading. Trying non-utf-8')
            ff=open(codebookFile).read()
        import re

        #variableNames=re.findall(r'-------------------------------------------------------------------------------\n([^\n\s]*)\s+([^\n\s]*)\n-------------------------------------------------------------------------------',ff,re.MULTILINE)#
        #print variableNames

        grs=ff.split('-------------------------------------------------------------------------------')
        #variableNs=grs[1::2]
        listOfNames=[]
        #variableDescs=grs[2::2]
        for hl, details in zip(grs[1::2],    grs[2::2]):
            vv,desc=re.findall(r'([^\s]*)\s+(.*)',hl.strip(),re.DOTALL)[0]
            if toLower:
                vv=vv.lower()
            listOfNames+=[vv]
            cbook+= [[vv,  dict(desc = re.sub(r'\s+',' ',desc),
                            stataCodebook = details # For reference, store everything
                            )]]
        cbook = OrderedDict(cbook) # Preserves variable order from Stata
        
        """ NOW READ LABELBOOK
        This may fail if labels are reused for multi variables, but can be fixed...
        IT now works for value labels which span more than one line, though it's frageile / kludged.

        """

        # Dec 2011: desperate. can't deal with utf-8 sheis. so using errors='replace'. :(
        ff=open(labelbookFile,encoding='utf-8',errors='replace').read()
        lrs=ff.split('-------------------------------------------------------------------------------')
        variableLabels=lrs[2::2]
        for vL in variableLabels:
            labelListAndVars=re.findall(r'\n\s+definition\n(.*?)\n\s*variables:(.*?)\n',vL,re.MULTILINE+re.DOTALL)[0]

            for  avar in  labelListAndVars[1].strip().split(','):
                var=avar.strip()
                couldBeMultipleVars=var.split(' ')
                var=couldBeMultipleVars[0]
                if toLower:
                    var=var.lower()
                if var in cbook: # 2010 Jan: N.B. THis was not necessary until I started allowing "subset" restriction for the codebook: I now may get codebook for a subset of variables, but labelbook for all of them.
                    cbook[var]['labels']={}
                for otherVar in couldBeMultipleVars[1:]:
                    if var in cbook: # See comment just above Jan 2010
                        cbook[otherVar]['labels']= cbook[var]['labels']

                """Horrid kludge to join multi-line descriptions: (ie assuming a fairly fixed format by STata)
                It has a problem for cases when there are values with no description?
                """
                revisedTable=labelListAndVars[0].strip().replace('\n               ',' ').split('\n')
                for LL in revisedTable:#labelListAndVars[0].strip().split('\n'):
                    assert not '               ' in LL
                    val_name_=re.findall(r'([^\s]*)\s+(.*)',LL.strip())
                    if val_name_:
                        val_name=val_name_[0]
                    else: # Maybe there's a value without a label here?
                        val_name=[LL.strip(),'']
                    if var in cbook: # See comment above, Jan 2010
                        # The value could be a ".a" or etc, ie not a number!
                        if val_name[0].startswith('.'):
                            cbook[var]['labels'][val_name[0]]=val_name[1]
                        else:
                            cbook[var]['labels'][int(val_name[0])]=val_name[1]
                            assert not '.' in val_name[0] #fishing.. does my code for ".a" work?
                        cbook[var]['labelbook']=deepcopy(vL)

            #       print cbook[var]['desc']+':'+var+ str(cbook[var]['labels'])

        # Let's also make a convenient summary tsv file of the variables. Use original order of variables

        import os
        cbfDir,cbfName=os.path.split(codebookFile)

        fnn=paths['working']+cbfName+'_varlist.tsv'
        fout= open(fnn,'wt', encoding='utf8')
        for vv in listOfNames:#cbook:
            fout.write('\t%s\t%s\n'%(vv,cbook[vv]['desc']))
        fout.close()
        assert listOfNames
        print "   Parsed codebook file to find %d variables; Wrote %s."%(len(listOfNames),fnn)

        assert self==None or self=={}
        return cbook
        self.update(cbook)
        self.__orderedVarNames=listOfNames
        #self._stataCodebookClass__orderedVarNames =listOfNames
        #self.variableOrder.update(listOfNames)
        return

    def clean_up_strings(self):
        ohoh
        some_unicode_quotes_to_latex


        
###########################################################################################
###
class Bsurveycodebook(OrderedDict):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    """
        The primary dict/data of this object is an OrderedDict of information about survey data variables.

    Value labels are lookups (OrderedDicts) from integer values to strings  which can initially be shared between variables (keys of surveycodebook). However, they must be deepcopied and made independent if ever the value labels are changed for some variable.
    The same is true of Float Values, which is a lookup that can be used to allow some integers to have NaN values

    In general, where text may be used for formatted printing, LaTeX markup is used.

        self._variable_labels = {}
        self._value_labels = {}  # Maps column names to named lookups
        self._named_labels = {}  # Stores named lookups, each of which maps integer values to labels
        self._named_float_values = {} # Stores named lookups, each of which maps integer values to float values (including NaN)
        self._questions = {} # Maps column names to relevant questionnaire question / further info
    """
    
    def __init__(self,source= None, fromDTA=False,fromTSV=False,fromPDF=False, loadName=None,):#  recreate=None, toLower=None,showVars=None,survey=None,version=None,stringsAreTeX=None):#*args,foo=None):  # Agh. June 2010: added "version" here this might confuse things, but there was a bug...

        """ Allow instantiation from a dict (codebook=dict) or from a Stata dataset itself (fromDTA=filename)

        myCodebook=stataCodebookClass()
        myCodebook=stataCodebookClass(fromDTA=myfilepath)
        myCodebook=stataCodebookClass(codebook=myDict)
        myCodebook=stataCodebookClass( a dict already in format of codebook)
            i.e.: {varname:    }

        """
        if source.__class__ in [surveycodebook,  OrderedDict]:
            super(surveycodebook, self).__init__(source)
        if source.__class__ == dict: # Got an unordered dict for some reason. Should do some checking on its contents: TO DO
            super(surveycodebook, self).__init__(OrderedDict(source))
        if isinstance(source, basestring) and source.endswith('.dta.gz'):
            foo =self._from_stata(source)
            assert isinstance(foo, OrderedDict)
            super(surveycodebook, self).__init__(foo)
            if 0: 
                for a,b in foo.items():
                    self[a]=b
            #super(surveycodebook, self).__init__(self, {1:2, '3':'f'})


    
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


def test_surveycodebook():
    stata_filename = paths['working']+'WV6_Stata_v_2016_01_01'
    cb = surveycodebook(stata_filename)
    # Make the strings easier for LaTeX and other printing:
    treat_OrderedDict_strings_recursively(cb, some_unicode_quotes_to_latex)
    

###########################################################################################
###
class surveypandas(pd.DataFrame):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,                   codebook=None):
        # Can I not use super() in the line below?
        pd.DataFrame.__init__(self, data=None, index=None, columns=None, dtype=None, copy=False)
        self.codebook = None
        
        if codebook is not None:
            self.codebook=surveycodebook(codebook)
                 

     

# Module interfaces to surveyPandas:
def read_stata(stata_filename):
    """ 
    Load data from a Stata file into a Pandas DataFrame derivative, but also load all the label and value label information into a codebook structure.
    """
    from pystata import dta2df
    df = dta2df(stata_filename)
    cb = surveycodebook(stata_filename)
    df.codebook = cb
    return df
    
    #Test codebook loading:
    # Load 

    
if __name__ == '__main__':
    test_treat_odict_strings_recursively()
    test_surveycodebook()

    df =read_stata(paths['working']+'WV6_Stata_v_2016_01_01.dta.gz')

#cb =surveycodebook({1:2, '3':'f'})
#print cb.keys()



