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
surveySeries : corresponding Series object.  Not started yet.
surveyCodebook: A nested dict providing information about variables in the dataset


Optionally, there can also be 'labelsname' and 'prompt' fields for each variable. These contain respectively some name for the set of values; and the actual wording of the survey question, if any, behind the variable.


This means that variables which have the same set of values descriptions list those descriptions redundantly.

We save the data and labels together in a pickled file with name ending in '.spandas'



n.b.: statsmodels.iolib.foreign.StataReader  seems not to be maintained; only Stata versions 8-12 can be read. Thus we use Stata text output as a version-independent way to transfer.

"""

from surveypandas_config import defaults
paths= defaults['paths']
from collections import OrderedDict
import os
import pandas as pd
import numpy as np
import cPickle as pkl

try:
    from cpblUtilities import doSystem, dgetget
except ImportError:
    print(' Some functions will not work. Install cpblutilities ')
    
from copy import deepcopy
VERBOSE=True

def some_unicode_quotes_to_latex(str):
    subs =[
        [u"\u0027", ";"], #\\textquotesingle "],
        [u'´', "'"],
        [u'\u2019', "'"],
        [u'“', "``"],
        [u'”', "''"],
        ]
    for a,b in subs:
        str=str.replace(a,b)
    return str

def treat_OrderedDict_strings_recursively(dictlike, method, inplace=True):
    """ Recursively treat strings in a dict or OrderedDict by applying some method to each string in each key/value. E.g. to remove some unicode
    This is for OrderedDicts. normal dicts to not accept the argument to popitem, so we would need to loop through the keys explicitly (what's wrong with that?)
    """
    assert inplace # False not written yet
    def treat_value(vv):
        if isinstance(v, basestring):
            return method(vv)
        if v.__class__ in [dict, OrderedDict]:
            return treat_OrderedDict_strings_recursively(vv, method)
        return vv
    for k in  list(dictlike.keys()):
        v=dictlike.pop(k)
        dictlike[k if not isinstance(k, basestring) else method(k)] = treat_value(v)
    return dictlike # Needed only for recursion, since the original dict is changed in place

def test_treat_odict_strings_recursively():
    x = {1:2, 3:4, 5:6}
    #print x
    treat_OrderedDict_strings_recursively(x, some_unicode_quotes_to_latex)
    #print x
    x = {'234': 'oiwe', 5:23, u'Don´t know':{u'Don´t know':u'Don\xb4t know'}}
    #print x
    treat_OrderedDict_strings_recursively(x, some_unicode_quotes_to_latex)
    #print x
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

        codebook=surveyCodebook(labels)
        survey=surveyDataFrame(codebook=labels)
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
class surveyCodebook(OrderedDict):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    """
        The primary dict/data of this object is an OrderedDict of information about survey data variables.

    Value labels are lookups (OrderedDicts) from integer values to strings  which can initially be shared between variables (keys of surveyCodebook). However, they must be deepcopied and made independent if ever the value labels are changed for some variable.
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


        if source.__class__ in [surveyCodebook,  OrderedDict]:
            super(surveyCodebook, self).__init__(source)
        elif source.__class__ == dict: # Got an unordered dict for some reason. Should do some checking on its contents: TO DO
            super(surveyCodebook, self).__init__(OrderedDict(source))
            
        elif isinstance(source, basestring):
            foo =self._from_stata(source)
            super(surveyCodebook, self).__init__(foo)
        else:
            super(surveyCodebook, self).__init__(source)
            
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
        from pystata import stripdtagz # Import pystata locally, since it's only one interface for surveyDataFrame
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


    def get_value_label(self, col, val):
        return dgetget(self, [col, 'labels', val], val)
        

    
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
    
    def rename_keys(self, substitution_pairs, inplace=False):
        """  To preserve order, this cycles through every key, just to change any. So the more that are passed at once in subsituion_pairs, the better
        """
        dsubstitution_pairs = dict( substitution_pairs)
        if inplace:
            obj = self
        else:
            obj = deepcopy(self)
        for k in  list(obj.keys()):
            v=obj.pop(k, False)
            if k in dsubstitution_pairs:
                obj[dsubstitution_pairs[k]] = v
                dsubstitution_pairs.pop(k) # For efficiency in "if", above
            else:
                obj[k] = v
        if not inplace:
            return obj


def test_surveyCodebook():
    stata_filename = paths['working']+'WV6_Stata_v_2016_01_01'
    cb = surveyCodebook(stata_filename)
    # Make the strings easier for LaTeX and other printing:
    treat_OrderedDict_strings_recursively(cb, some_unicode_quotes_to_latex)
    

###########################################################################################
###
class surveyDataFrame(pd.DataFrame):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    def __init__(self, data=None, codebook=None,  # Note: order is different from pandas.DataFrame
                 index=None, columns=None, dtype=None, copy=False,                   
                 drop = True, # Drop codebook entries which don't have associated columns
                 ):
        """ Basic construction is surveyDataFrame( DataFrame, dict or surveyCodebook )
        """
        # Can I not use super() in the line below?
        super(surveyDataFrame, self).__init__(data=data, index =index, columns= columns, dtype= dtype, copy=copy)
        
        #pd.DataFrame.__init__(self, data=data, index =index, columns= columns, dtype= dtype, copy=copy)
        self.codebook = None
        
        if codebook is not None:
            self.codebook=surveyCodebook(codebook)
            if drop:
                self.drop_unused_codebooks()
    #def _from_dataframe_and_codebook(self, df, cb):
    #    return( surveyDataFrame(data = df, codebook=cb) )
    def drop_unused_codebooks(self):
        for k in self.codebook:
            if k not in self.columns:
                self.codebook.pop(k)

    def remap_values(self, mapping):
        pass
    def as_labels(self, subset=None, inplace=False):
        """ Remap a column (oops-  That should be for surveySeries) to its value labels.
        Until this is written for surveySeries, do it for all columns in subset. Must be length one. :(
        """
        assert subset is not None and (len(subset)==1 or isinstance(subset, basestring))
        assert inplace is False
        if not  isinstance(subset, basestring): subset= subset[0]
        assert subset in self.codebook
        v = self[subset]
        return v.map(lambda vv: self.codebook[subset]['labels'].get(vv, vv))
                     
    def copy(self, deep=True):
        """
        Make a copy of this objects data.
        Parameters
        ----------
        deep : boolean or string, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices or the data are copied.
            Note that when ``deep=True`` data is copied, actual python objects
            will not be copied recursively, only the reference to the object.
            This is in contrast to ``copy.deepcopy`` in the Standard Library,
            which recursively copies object data.
        Returns
        -------
        copy : type of caller
        """
        # Use DataFrame's copy for the DF part. And deepcopy 
        return surveyDataFrame(data = pd.DataFrame(self).copy(deep=True),
                            codebook = deepcopy(self.codebook) )

    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        return self.copy(deep=True)
    def to_pickle(self, path, compression='infer',
                  protocol= pkl.HIGHEST_PROTOCOL):
        """ See pandas.io.pickle and pandas.DataFrame.to_pickle. Clearly, the latter is not very specific to the structure of DataFrames, since I can pass in a dict instead. But the approach below preserves the calling format of Pandas' DataFrame, at least. """
        #from pandas.io.common import _get_handle, _infer_compression, _stringify_path
        from pandas.io.pickle import to_pickle as pandas_to_pickle
        return pandas_to_pickle( {'o': self, 'c':self.codebook}, path, compression=compression, protocol=protocol)


    def describe(self, **argv):
        print( pd.DataFrame.describe(self,**argv))
        print('\n')
        # And also show descriptions, at least:
        for cc in self.columns:
            print('{}: {}'.format(cc, dgetget(self.codebook, [cc,'desc'], '')))
     
    def rename_columns(self, subs_dict, inplace=False):
        """  """
        # Now, rename in codebook
        cb = self.codebook.rename_keys(subs_dict, inplace=inplace)
        # And rename columns:
        df = self.rename(columns = dict(subs_dict), inplace=inplace)
        if not inplace:
            return surveyDataFrame(df, codebook = cb)
        else:
            self.assert_unique_columns() # To do: this should be checked for inplace=False too!

        
    def rename_columns_from_descriptions(self, subset = None, skip_already_renamed = True, inplace = False):
        """ Use 'desc' field of codebook to provide crude variable names:
        """
        assert len(self.columns.unique()) == len(self.columns)
        subs_list=[]
        r_subs_dict ={}
        for vv in self.columns:
            if subset is not None and vv in subset:
                continue
            impn = dgetget(self.codebook, [vv, 'imported_name'], '') 
            if impn and impn != vv and skip_already_renamed:
                continue
            if vv not in self.codebook:
                raise("sf")
            formattedDesc = ''.join([c if c.isalnum() else '_' for c in self.codebook[vv]['desc'] ])
            while formattedDesc in r_subs_dict:
                formattedDesc += '_'
            r_subs_dict[formattedDesc] = vv
            #subs_dict += [[vv, formattedDesc]]

        subs_dict = dict([(b,a) for a,b in r_subs_dict.items()])
        return self.rename_columns(subs_dict, inplace = inplace)

    def grep(self, search_string):
        """ Look for a string in variable names, descriptions, questionnaire questions, etc
        """
        found = []
        ss = search_string.lower()
        for k,v in self.items():
            if ss in (k + dgetget(self.codebook,[k, 'desc'],'') + dgetget(self.codebook,[k, 'question'],'') ).lower():
                found += [k]
        return found
        print '\n'.join(found)

    def dgrep(self, search_string,
              width=None, # Max Display width for descriptions
              to_floats = True, # By default, do any NaN conversion of integer-valued columns so as to give useful counts in describe()
            ):
        """ Also give the descriptions for columns found by grep.  What would be a better name/etc for this?
        TO DO: Also show questionnaire questions if available.
        """
        cc = self.grep(search_string)
        if cc and to_floats:
            if VERBOSE: print(' Converting {} columns to floats prior to describe()...'.format(len(cc)))
            surveyDataFrame(self[ cc ], codebook= self.codebook).to_floats().describe()
        elif cc:
            surveyDataFrame(self[ cc ], codebook= self.codebook).describe()
        else:
            print(' Nothing found matching "{}"'.format(search_string))

    def set_NaN_strings(self, list_of_strings, subset=None, exclude=None):
        """ list of strings is something like ["Don't know", "Not asked"].
        Integer values with these labels will be set to NaN in the float_values lookup.
        """
        pass
        
    def set_float_values_from_negative_integers(self, subset=None, exclude=None):
        """ Create new "float_values" lookup in the codebook.
        For integer values, assume negative values.
        This method may not be named well?"""
        cb = self.codebook
        counter=0
        for k,cbk in cb.items():
            if 'float_values' not in cbk and 'labels' in cbk:
                assert all(([isinstance(kk,int) for kk in cbk['labels'].keys()]))
                cbk['float_values'] = dict([(k, np.nan if k<0 else k) for k,v in cbk['labels'].items()])
                counter += 1
        if VERBOSE:
            print(' Created float_values lookups for {} columns.'.format(counter))
                
    def assert_unique_columns(self):
        assert len(self.columns.unique()) == len(self.columns)

    def to_floats(self, inplace=False):
        """
        Use the float_values element in codebook entries to recast columns as floats, with non-response values changed to NaN.
        For this to work, there needs to exist "float_values" lookups in codebook entries, for instance as created by set_float_values_from_negative_integers
        """
        newself = self.copy(deep=True) if not inplace else self
        for k,cbk in newself.codebook.items():
            newself[k] = newself[k].map(lambda vv,k=k: dgetget(newself.codebook, [k, 'float_values', vv], vv))
        if not inplace:
            return newself

    def append_indicators(self, col, reference_value = None, dropna = True):
        """ Append a complete set of indicator variables (columns) for the values of discrete variable discretecol.
        If convenient, these will be named according to the value labels.
        The set of new columns is returned.

        If reference_value is provided, an indicator for this value will be excluded.
        """
        assert dropna is True # Alternative not written yet
        newcols =[]
        uvals = self[col].dropna().unique()
        if col in self.codebook:
            nv = len(uvals)
            labels = self.as_labels(col)         
            nl = len(labels.unique())
            assert nv == nl # Other cases not written yet
            for vv in uvals: #labels.unique():
                LL = self.codebook.get_value_label(col, vv)
                if LL == reference_value:
                    continue
                newcol = 'i'+col+'.'+''.join([c for c in LL if c.isalnum()])
                assert newcol not in self
                assert newcol not in self.codebook
                self[newcol] =  (labels == LL).astype(int) # What about NaNs?
                self.codebook[newcol] = {'desc': 'Indicator for {} == {} ("{}")'.format(col, vv, LL )  }
                newcols += [newcol]
            return newcols
        # There are no labels for this col
        for vv in uvals:
            newcol = 'i'+col+'.'+''.join([c for c in vv if c.isalnum()])
            assert newcol not in self
            assert newcol not in self.codebook
            self[newcol] =  (self[col] == vv).astype(int) # What about NaNs?
            self.codebook[newcol] = {'desc': 'Indicator for {} == {}'.format(col, vv )  }
            newcols += [newcol]
        return newcols
    def to_stata(self, filename, compress=None,  **args):
        """ This *should* attempt to make column names Stata-friendly, ...
              and export codebook info to Stata (I may already have routines for some of this?)
        What to do with index?

        compress=True: by default, convert .dta files to .dta.gz files (See Stata's gzuse page). Otherwise, go by the filename extension
        """
        cols = self.columns
        assert len(cols) == len(cols.unique())
        newcols = [cc.replace(' ','_') for cc in cols]
        # To do: check for duplicates
        # To do: truncate to Stata length
        outdf = pd.DataFrame(self).copy()
        outdf.columns = newcols
        # To do: check/correct filename extension, better than one line below
        if filename.endswith('.dta'):
            outdf.to_stata(filename, **args)
        elif filename.endswith('.dta.gz'):
            outdf.to_stata(filename[:-3], **args)
            os.system('gzip {}'.format(filename[:-3]))
        else:
            ToDO_foooo
        return
    # Here, reproduce various DataFrame calls so as to reconstruct the surveypandas by re-adding the codebook information afterwards:
    def query(self, qs, **args):
        return  surveyDataFrame(pd.DataFrame.query(self, qs, **args), codebook=self.codebook)
        
# Module interfaces to surveyDataFrame:
def read_pickle(path, compression='infer'):
    with open(path) as f:
        loaded_obj = pkl.load(f)
    if isinstance(loaded_obj , dict) and 'o' in loaded_obj and 'c' in loaded_obj:
        return  surveyDataFrame(loaded_obj['o'], codebook = loaded_obj['c'])
    if isinstance(loaded_obj , pd.DataFrame):
        return  surveyDataFrame(loaded_obj)
    
def read_stata(stata_filename, unicode_to_latex=True):
    """ 
    Load data from a Stata file into a Pandas DataFrame derivative, but also load all the label and value label information into a codebook structure.
    """
    from pystata import dta2df
    df = dta2df(stata_filename)
    assert not df.empty
    sdf = surveyDataFrame( df )
    assert not sdf.empty
    cb = surveyCodebook(stata_filename)
    for k,v in cb.items():
        cb[k]['imported_name'] = k # Keep record of original variable name

    if unicode_to_latex:        
        # Make the strings easier for LaTeX and other printing:
        treat_OrderedDict_strings_recursively(cb, some_unicode_quotes_to_latex)
    sdf.codebook = cb
    return sdf
    
    #Test codebook loading:
    # Load 





# Testing here
def test_surveypandas():
    test_treat_odict_strings_recursively()
    test_surveyCodebook()
    sdf = read_stata(paths['working']+'WV6_Stata_v_2016_01_01.dta.gz')
    sdf.assert_unique_columns()
    sdf.rename_columns_from_descriptions()
    sdf.assert_unique_columns()
    sdf.set_float_values_from_negative_integers() 
    sdf.assert_unique_columns()
    print sdf.grep('satis')
    sdf['Satisfaction_with_your_life'].describe()

    sdf.to_pickle(paths['scratch']+'surveypandas-test-raw.spandas') # Fairly compact
    pd.DataFrame(sdf).to_pickle(paths['scratch']+'surveypandas-test-df.spandas') # Fairly compact
    with open(paths['scratch']+'surveypandas-test-cb.spandas', 'wb') as f:        pkl.dump(sdf.codebook, f)

    print (' Reading raw...')
    sdf2 = read_pickle(paths['scratch']+'surveypandas-test-raw.spandas') 
    assert len(sdf2)==len(sdf)
    assert sdf2.codebook

    if 1: 
        print(' Rest will be slow...')
        print (' Creating float...')
        fdf= sdf.to_floats() # Very slow, still.
        fdf['Satisfaction_with_your_life'].describe()
        print (' Saving float...')
        fdf.to_pickle(paths['scratch']+'surveypandas-test-float.spandas') # Enormous. Don't do this if you can avoid it.
        print (' Reading float...')
        fdf2 = read_pickle(paths['scratch']+'surveypandas-test-raw.spandas') 
        assert len(fdf2)==len(fdf)
        assert fdf2.codebook


    
if __name__ == '__main__':
    # Debug:
    sdf = read_stata(paths['working']+'WV6_Stata_v_2016_01_01.dta.gz')
    sdf.append_indicators('V2')
    
    pass
    
