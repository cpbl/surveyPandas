#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

This is to provide a link between statistical analysis done using surveypandas or pystata, and the cpblUtilities.textables methods, which produce flexTeXtable format for LaTeX.

See the plan.org for plans.

This starts by providing interfaces from statistical routines, or their output, to regression model objects in the tradition of my pystata package.

to keep it flexible, a regmodel object is a dict with various elements. They include a DataFrame
of coefficients or marginal effects.


"""


import pandas as pd
import numpy as np
import statsmodels
from copy import deepcopy
from pystata import  composeLaTeXregressionTable # Copied here!
from collections import OrderedDict
import itertools
from cpblUtilities import orderListByRule
import cpblUtilities.textables as textables
from cpblUtilities.textables.stats_formatting import df_format_estimate_column_with_latex_significance,  chooseSFormat

                                                  
from pystata import  substitutedNames # Temporary! Replace this


def from_statsmodels(res, **args): #flags=None, name=None, ):
    """ Take result object from a statsmodels estimate, and put in into a latexRegressions model dict format

    """
    estcoefs = pd.DataFrame(OrderedDict([['b', res.params],
                                    ['p', res.pvalues],
                                   ['t', res.tvalues],
                                    ['se', res.bse],
                                  ]))
    estcoefs.index.name = 'xvar'
    rm = {'engine':'statsmodels',
          'method': {
              statsmodels.regression.linear_model.OLS: 'ols'
          }.get(res.model.__class__, 'unknown'),
          #'resultobject':
          'estcoefs': estcoefs.reset_index(),
          'depvar': res.model.endog_names,
          'eststats': OrderedDict(([['r2', res.rsquared],
                        ['r2a', res.rsquared_adj],
                       ['N',  res.nobs],
                       ])),
          'summary': res.summary2(),
          'rawLogfileOutput': None, # rename this to rawStataLogFileOutput? And remove it when non-stata?
          'res': res, # For debugging only. :)
          'flags':  OrderedDict([]),
          'textralines': OrderedDict([]), # Let's deprecate this? And just call them flags?
          'warnings': [],
          }
    rm.update(args)
    return rm




from pystata.latexRegressions import latexRegressionFile as pystlr
# Build this up gradually by replacing methods from pystata's version
###########################################################################################
###
class latexRegressionFile(pystlr):  #  # # # # #    MAJOR CLASS    # # # # #  #
    ###
    #######################################################################################
    def appendRegressionTable_from_models(
            self,
            models,
            tableFilePath=None,
            suppressSE=False,
            substitutions=None,
            transposed=None,
            ):
        """
        Instantiate a textables.latexTable object, create the appropriate LaTeX files, and then include the resulting external LaTeX table into the LaTeX file (self).
        """

        if transposed is None: transposed = 'both'
        assert transposed in ['both', True, False]
        out1, out2 = self.prepare_normal_and_transposed_regression_table_DataFrames_from_model_list(
            models,
            suppressSE=suppressSE,
            substitutions=substitutions,
            tableFormat= {'title':'test',
                          'comments':'none test',
                          'caption': ' no yet',},
            transposed=transposed,
        )  #,hideRows=hideRows),modelTeXformat=modelTeXformat,
        # {'comments':tableComments,'caption':tableCaption,}


        texTable = textables.latexTable(dataframe = out1.reset_index())
        callerTex = texTable.toCPBLtable(tableFilePath)
        self.append(callerTex)

        

    def concat_model_list(self,models):
        """ Take a list of model dicts and return three DataFrames:
         - estimated coefficients by model
         - estimation stats by model
         - extra flags ("textralines") by model

        The index of each df will be the common model-identifying columns.

        This is analogous to / replacement for pystata's modelResultsByVar 
        
        """
        #models = deepcopy(models)
        coefs, stats, extras = [],[],[]
        allkeys = []
        for ii, mm in enumerate(models):
            df = mm['estcoefs']
            stat_dict = mm['eststats']
            flag_dict = mm['flags'] if mm['flags'] else  OrderedDict([])
            mm['modelNum'] = ii
            indices = [kk for kk in ['modelNum', 'modelName', 'depvar',
                                     'groupname', #deprecated in surveypandas
                                     'modelName1', 'modelName2', 'modelName3', 'modelName4', 'modelName5']  if kk in mm]
            allkeys += indices
            for II,vv in itertools.product(indices, [df, stat_dict, flag_dict]):
                vv[II] = mm[II]
            coefs += [df]
            stats += [stat_dict]
            extras += [flag_dict]
        allkeys = list(np.unique(allkeys))
        return ( pd.concat(coefs).set_index(allkeys),
                 pd.DataFrame(stats).set_index(allkeys),
                 pd.DataFrame(extras).set_index(allkeys) )



    ###########################################################################################
    ###
    def prepare_normal_and_transposed_regression_table_DataFrames_from_model_list(self, models,
                                    tableFormat=None,
                                    suppressSE=False,
                                    showFlags=None,
                                    showStats=None,
                                    substitutions=None,
                                    modelTeXformat=None,
                                    transposed=None,
                                    multirowLabels=True,
                                    showOnlyVars=None,
                                    hideVars=None):
        #######################################################################################
        """ See pystata's composeLaTeXregressionTable for history/comments
        """

        assert not tableFormat == None
        if tableFormat == None:
            tableFormat = {}

        if 'title' not in tableFormat and 'caption' in tableFormat:
            tableFormat['title'] = tableFormat['caption']

        if 'hideVars' in tableFormat and hideVars == None:
            hideVars = tableFormat['hideVars']
        if hideVars is None: hideVars = []
        
        if substitutions is None:
            substitutions = self.substitutions
            
        if 0: 
            # Make .csv output copy for the same data:
            tsvTableFormat = deepcopy(tableFormat)
            tsvTableFormat.update({'csvMode': 'all'})
            composeTSVregressionTable(
                models,
                substitutions=substitutions,
                tableTitle=tableFormat['title'],
                caption=tableFormat['caption'],
                comments=tableFormat.get('comments', ''),
                tableFormat=tsvTableFormat)

        # May 2011: try this:
        if tableFormat.get('hideModelNames', False):
            for mm in models:
                if 'texModelName' in mm:
                    mm.pop('texModelName')
                mm['modelName'] = ''

        # Use the code from modelsToPairedRows to order the variables... and start by using modelResultsByVar to get the right lists of vars (in three categories)

        byVar, byStat, byTextraline = self.concat_model_list(models) 

        variableOrder = tableFormat.get('variableOrder', self.variableOrder)
        if variableOrder is None:
            variableOrder = []
        if isinstance(variableOrder, basestring):
            variableOrder = [vv for vv in variableOrder.split(' ') if vv]

        # In order to ensure the constant term comes last... let's append all variables known from substitutions to the end of variable order:
        # Following line fails, since const substition is part of substitutions, and could be early...
        variableOrder += [vvv[0] for vvv in substitutions]

        # look both for e(stat) and stat (case of Stata)
        hideStats = [
            sv for sv in byStat.keys()
            if 'e(%s)' % sv in hideVars or sv in hideVars
        ]  ###'r2','r2_a','r2_p','N','p','N_clust'

        coefVars = orderListByRule(byVar.xvar.values, variableOrder, dropIfKey=hideVars)
        statsVars = orderListByRule(
            orderListByRule(byStat.keys(),
                            ['r2', 'r2_a', 'r2_p', 'N', 'p', 'N_clust']),
            variableOrder,
            dropIfKey=hideStats)
        flagsVars = orderListByRule(
            byTextraline.keys(), variableOrder, dropIfKey=hideVars)

        if showOnlyVars:  # In which case variableOrder, variableOrder will have no effect:
            coefVars = [
                vv for vv in showOnlyVars if vv in coefVars
            ]  #orderListByRule(vars,showOnlyVars) if vv in showOnlyVars]
            statsVars = [
                vv for vv in showOnlyVars if vv in statsVars
            ]  #orderListByRule(vars,showOnlyVars) if vv in showOnlyVars]
            flagsVars = [
                vv for vv in showOnlyVars if vv in flagsVars
            ]  #orderListByRule(vars,showOnlyVars) if vv in showOnlyVars]

            # Choose the format for the table, where some choice is left
            # Oct 2009: it seems "none" is not getting this far. It's already being converted to True somewhere. So use 'auto' or fix it.
            # June 2011: This should just be used for choosing which one to display, since both should be built in to tex file ...
            
        if transposed == None or (isinstance(transposed, basestring) and
                                  transposed == 'auto'):
            transposed = True  #False

            # 30 across long dimension (11") by 20 across short dimension (8.5") is pretty packed. So decide here whether to do non-transposed.
            # Begin here various heuristics....
            if len(coefVars) + len(statsVars) + len(flagsVars) > 30 and len(
                    models) <= 20:
                transposed = False
            elif len(coefVars) + len(statsVars) + len(flagsVars) > 20 and len(
                    models) <= 10:
                transposed = False

        subs = substitutions

        modelsAsRows = transposed == True
        varsAsRows = transposed == False

        r2names = [
            'e(r2-a)', 'e(r2)', 'e(r2-p)', 'r2', 'r2_a', 'r2-a', 'r2_p', 'r2-p'
        ]

        def formatEstStat(model, estat):
            """
            pre-format these so that we can do 3 sig digs for r2:
            """
            if estat in r2names:
                return (chooseSFormat(
                    dgetget(model, 'eststats', estat, fNaN),
                    lowCutoff=1.0e-3,
                    threeSigDigs=True))  #,convertStrings=True
            else:
                return (chooseSFormat(dgetget(model, 'eststats', estat, fNaN))
                        )  #lowCutoff=1.0e-3,convertStrings=True,threeSigDigs=True)

        # Some strings can be set regardless of transposed or conventional layout:
        tableLabel = r'tab:%s' % (''.join(
            [s for s in tableFormat.get('caption', '') if s not in ' ,.~()-']))

        landscape = False  # This maybe used to be more automated, depending on ntexcols, ntexcols. I've set it to False because I tend to have one continuous landscape environment for the whole tex file now.

        # huh? this section used be after the big transposed if! Weird.
        if 0:
            ntexrows, ntexcols = 1 + (
                1 + int(suppressSE)
            ) * len(models), 2 + len(coefVars + statsVars + flagsVars)

            formats = 'lc*{%d}{r}' % (ntexcols - 2)  # or: 'l'+'c'*nvars
            if multirowLabels:
                formats = 'lp{3cm}*{%d}{r}' % (ntexcols - 2)  # or: 'l'+'c'*nvars

            ###assert not '&' in [cellsvmmodel[0] for cellsvmmodel in cellsvm] # This would be a mistake in regTable caller?
            headersLine = '\t&'.join(['', ''] + [
                r'\begin{sideways}\sltcheadername{%s}\end{sideways}' %
                substitutedNames(vv, substitutions)
                for vv in coefVars + flagsVars + statsVars
            ]) + '\\\\ \n' + r'\hline'  ####cellsvmmodel[0] for cellsvmmodel in cellsvm])+'\\\\ \n'+r'\hline'#\cline{1-\ctNtabCols}'
            headersLine1 = headersLine + '\\hline\n'  #r'\cline{1-\ctNtabCols}'+'\n'
            headersLine2 = headersLine + '\n'

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # First, do preparaation as though vars as rows:
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        df_format_estimate_column_with_latex_significance(byVar,
                                                      'b',
                                                          p_col='p',
                                                          se_col='se',
                                                      replace_estimate_column=True,
                                                          drop_p_column=True,
                                                          )
        # Use multiIndex columns for vars-as-rows mode:
        vars_as_rows = byVar.reset_index()[['modelName','xvar','b','se']].set_index(['xvar', 'modelName']).unstack(level=1).swaplevel(axis=1).sort_index(axis=1).fillna('')

        vars_as_rows_paired = textables.interleave_se_columns_as_rows(vars_as_rows, wrap_se_for_LaTeX=True, duplicate_index=False)
        # And get rid of the "b" column level:
        vars_as_rows_paired.columns= vars_as_rows_paired.columns.get_level_values(0)

        index_vars = ['modelName'] # if modelName1, etc don't exist

        # Final vars-as-rows table:
        bs = byStat.reset_index()[statsVars+index_vars].set_index(index_vars)
        bs['N'] = bs['N'].astype(int).astype(str)
        bs = bs.T
        textables.formatDFforLaTeX(bs)
        df_var = pd.concat([vars_as_rows_paired,   bs  ])

        return df_var, None # Normal and transposed versions


        includeTeX, callerTeX = cpblTableStyC(
            tableElements=varsAsRowsElements,
            tableElementsTrans=modelsAsRowsElements,
            showTransposed=transposed)
        if multirowLabels:
            callerTeX = r"""\renewcommand{\sltrheadername}[1]{\multirow{2}{3cm}{\hspace{0pt}#1\vfill}}
    \renewcommand{\sltrbheadername}[1]{\multirow{-2}{3cm}{\hspace{0pt}#1\vfill}}
    """ + callerTeX

        return (includeTeX, callerTeX, transposed)
        
        okay

        
        #varsAsRows=True
        body = ''
        for vv in coefVars:
            pValues = None
            # Caution! I'm introducing here May 2011: if there is any suestTest in the table, then all p-values given by Stata will be used, even though they are often "0.0". But hopefully 0.0 corresponds to smaller than 10^3 or whatever my most stringent level is. (since otherwise, I've been using the t-stat, which as more precision, to calculate the p-value category myself).
            # N.B. for varsAsRows, using "byVar" has already selected "p" as the display coefficient for any suestTest columns.
            if any([mm.get('special', '') in ['suestTests'] for mm in models]):
                pValues = [None] + byVar[vv]['p']

            ##        if model.get('special','') in ['suestTests']:
            ##         displayEst='p' # For OLS etc
            ##         displayErr='nothing!'
            ##         estValues=[r'\sltheadernum{'+model.get('texModelNum','(%d)'%(model.get('modelNum',0)))+'}',
            ##     			  r'\sltrheadername{'+ model['tmpTableRowName']+'}']+[dgetget(model,'estcoefs',vv,displayEst,fNaN) for vv in coefVars]+[dgetget(model,'textralines',vv,fNaN) for vv in flagsVars]+[formatEstStat(model,vv) for vv in statsVars]
            ##         errValues=['','']+[dgetget(model,'estcoefs',vv,displayErr,fNaN) for vv in coefVars]+['' for vv in flagsVars]+['' for vv in statsVars]
            ##         pValues=['','']+[dgetget(model,'estcoefs',vv,'p',fNaN) for vv in coefVars]+['' for vv in flagsVars]+['' for vv in statsVars]

            ##         tworows=formatPairedRow([estValues,errValues],
            ##                                 greycells='tableshading' in model and model['tableshading'] in ['grey'],
            ##                                 pValues=pValues)
            assert '_' not in substitutedNames(vv, substitutions)
            tworows = formatPairedRow(
                [[r'\sltrheadername{%s}' % substitutedNames(vv, substitutions)] +
                 byVar[vv]['coefs'], [''] + byVar[vv]['ses']],
                pValues=pValues)
            #    ['']+[dgetget(model,'estcoefs',vv,displayErr,fNaN) for vv in coefVars]+['' for vv in flagsVars]+['' for vv in statsVars]],greycells='tableshading' in model and model['tableshading'] in ['grey'])#,modelsAsRows=True)
            #tworows=tworows[0:(2-int(suppressSE))]  # Include standard errors?
            body+= '\t& '.join([cc for cc in tworows[0]])+'\\\\ \n'+r'\showSEs{'+\
                            '\t& '.join([cc for cc in tworows[1]]) +' \\\\ }{}\n'
        body += r'\hline ' + '\n'  # Separate the coefs from extralines..
        for vv in flagsVars:
            body += '\t& '.join([substitutedNames(vv, substitutions)] +
                                byTextraline[vv]) + '\\\\ \n'
        for estat in statsVars:
            lowCutoff, threeSigDigs = (1.0e-3, True) if estat in r2names else (
                1.0e-5, False) if estat in ['widstat', 'jp', 'idp'] else (None,
                                                                          False)
            body += '\t& '.join([substitutedNames(estat, substitutions)] + [
                chooseSFormat(
                    cc, lowCutoff=lowCutoff, threeSigDigs=threeSigDigs)
                for cc in byStat[estat]
            ]) + '\\\\ \n'
            #assert not 'idp' == estat

        ntexrows, ntexcols = 1 + len(coefVars + statsVars + flagsVars), 1 + (
            1 + int(suppressSE)) * len(models)  # ?????NOT CHECKED
        formats = 'l*{%d}{r}' % (ntexcols - 1)  # or: 'l'+'c'*nvars
        if any(['|' in mm['format'] for mm in models]):
            formats = 'l' + ''.join(
                [mm['format'] for mm in models
                 ])  # N.b. this is rewritten below for use in multicolum headers.

        def smartColumnHeader(colgroups, colheads, colnums, colformats=None):
            """
            See description for main function, above.
            returns headersline1,headersline2
    See also 201709 single_to_multicolumn_fixer() in cpblUtilities/textables
            """
            if not any(colheads):
                assert not any(
                    colgroups
                )  # It would be silly to have group names but no names: If you just want one row, use names. (?)
                return ('\t&'.join([''] + [
                    r'\sltcheadername{%s}' % (model.get('texModelNum', '(%d)' % (
                        model.get('modelNum', 0)))) for model in models
                ]) + '\\\\ \\hline \n', r'\ctFirstHeader')
            #  Now, loop through and find consecutive groups...
            if colformats is None:
                colformats = ['c' for xx in colheads]

            def findAdjacentRepeats(
                    colnames, cformats
            ):  # Build list of possibly-multicolumn headers for one row.
                hgroups = []
                for ih, hh in enumerate(colnames):
                    if ih > 0 and hh == hgroups[-1][0]:
                        hgroups[-1][1] += 1
                        hgroups[-1][2] = 'c' + '|' * (
                            cformats[ih].endswith('|')
                        )  # Multicolumn headings should all be centered.
                    else:
                        hgroups += [[hh, 1, cformats[ih]]]
                return (hgroups)

            # I'm calling the top header hgroup1, the lower one hgroup0.  So in the future, we could simply accept an array of names, or call them name, name1, etc.
            hgroups1 = None if not any(colgroups) else findAdjacentRepeats(
                colgroups, colformats)
            hgroups0 = findAdjacentRepeats(colheads, colformats)
            rotateNames = not any(colgroups) and not any(
                [hh[1] > 1 for hh in hgroups0])
            if rotateNames:
                headersLine = '\\cpbltoprule\n' + '\t&'.join([''] + [
                    r'\begin{sideways}\sltcheadername{%s}\end{sideways}' %
                    substitutedNames(
                        model.get('texModelName', model.get('name', '')),
                        substitutions) for model in models
                ]) + '\\\\ \n'
                # IF there are numbers, too, then show them as a second row!
                if any([
                        'modelNum' in model or 'texModelNum' in model
                        for model in models
                ]):
                    headersLine += '\t&'.join([''] + [
                        r'\begin{sideways}\sltcheadername{%s}\end{sideways}' % (
                            model.get('texModelNum', '(%d)' % (model.get(
                                'modelNum', 0)))) for model in models
                    ]) + '\\\\ \\hline \n'
                return (r'\ctSubsequentHeaders \hline ', headersLine)

            headersLine = '\\cpbltoprule\n'
            if any(colgroups):
                headersLine += '\t&'.join([''] + [
                    r'\multicolumn{%d}{%s}{\sltcheadername{%s}}' % (
                        hh[1], hh[2], hh[0]) for hh in hgroups1
                ]) + '\\\\ \n'
            headersLine += '\t&'.join([''] + [
                r'\multicolumn{%d}{%s}{\sltcheadername{%s}}' % (
                    hh[1], hh[2], hh[0]) for hh in hgroups0
            ]) + '\\\\ \n'
            # IF there are numbers, too, then show them as a second row
            if any(
                    colnums
            ):  #['modelNum' in model or 'texModeulNum' in model for model in models]):
                headersLine += '\t&'.join([''] + [
                    r'\sltcheadername{%s}' % nns for nns in colnums
                ]) + '\\\\ \n'
            return (r'\ctSubsequentHeaders \hline ', headersLine)
            #            for ih,hh in enumerate(colheads):
            #               if ih>0 and hh==hgroups[-1][0]:
            #                    hgroups[-1][1]+=1
            #                    hgroups[-1][2]='c'+'|'*(colformats[ih].endswith('|'))# Multicolumn headings should all be centered.
            #               else:
            #                    hgroups+=[[hh,1,colformats[ih]]]
            #
            #            if any([hh[1]>1 for hh in hgroups]):
            #                """ Do not rotate any numbers or headings. Use multicolumn: since there are repeated headers."""
            #                headersLine='\\cpbltoprule\n'+ '\t&'.join(['']+[r'\multicolumn{%d}{%s}{\sltcheadername{%s}}'%(hh[1],hh[2],hh[0]) for hh in hgroups])+'\\\\ \n'
            #                # IF there are numbers, too, then show them as a second row
            #                if any(colnums):#['modelNum' in model or 'texModeulNum' in model for model in models]):
            #                    headersLine+='\t&'.join(['']+[r'\sltcheadername{%s}'%nns for nns in colnums])+'\\\\ \n'
            #            else:
            #                 headersLine='\\cpbltoprule\n'+ '\t&'.join(['']+[r'\begin{sideways}\sltcheadername{%s}\end{sideways}'%substitutedNames(model.get('texModelName',model.get('name','')),substitutions) for model in models])+'\\\\ \n'
            #                 # IF there are numbers, too, then show them as a second row!
            #                 if any(['modelNum' in model or 'texModelNum' in model for model in models]):
            #                    headersLine+='\t&'.join(['']+[r'\begin{sideways}\sltcheadername{%s}\end{sideways}'%(model.get('texModelNum','(%d)'%(model.get('modelNum',0)))) for model in models])+'\\\\ \\hline \n'
            #
            #return(r'\ctSubsequentHeaders \hline ',headersLine)
            # So we have colgroups AND colheads defined

        if 0:  # March 2015: I think the following lines are all obseleted by the smartColumnHeader call:
            headersLine = '\t&'.join([''] + [
                r'\begin{sideways}\sltcheadername{%s}\end{sideways}' %
                substitutedNames(
                    model.get('texModelName', model.get('name', '')),
                    substitutions) for model in models
            ]) + '\\\\ \n'
            # IF there are numbers, too, then show them as a second row
            if any([
                    'modelNum' in model or 'texModelNum' in model
                    for model in models
            ]):
                headersLine += '\t&'.join([''] + [
                    r'\begin{sideways}\sltcheadername{%s}\end{sideways}' %
                    (model.get('texModelNum', '(%d)' % (model.get('modelNum', 0))))
                    for model in models
                ]) + '\\\\ \n'
            headersLine1 = headersLine + r'\hline' + '\\hline\n'  #r'\cline{1-\ctNtabCols}'+'\n'
            headersLine2 = headersLine + r'\hline' + '\n'
        headersLine1, headersLine2 = smartColumnHeader(
            [
                substitutedNames(model.get('modelGroupName', ''), substitutions)
                for model in models
            ], [
                substitutedNames(
                    model.get('texModelName', model.get('name', '')),
                    substitutions) for model in models
            ], [
                model.get('texModelNum', '(%d)' % (model.get('modelNum', 0)))
                for model in models
            ],
            colformats=[mm['format'] for mm in models])
        varsAsRowsElements = deepcopy(
            cpblTableElements(
                body=body,
                cformat=formats,
                firstPageHeader=headersLine1,
                otherPageHeader=headersLine2,
                tableTitle=tableFormat.get('title', None),
                caption=tableFormat.get('caption', None),
                label=tableLabel,
                ncols=ntexcols,
                nrows=ntexrows,
                footer=colourLegend() + ' ' + tableFormat.get('comments', None),
                tableName=tableFormat.get('title', None),
                landscape=landscape))

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Second, do preparaation as though models as rows:
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #modelsAsRows:
        """
            Main loop over models, adding appropriately to output array of LaTeX entries.
            Add a row or paired row
            """
        body = ''
        # Decide whether to show just model numbers for rows, or model numbers and names:
        for model in models:
            model['tmpTableRowName'] = model.get(
                'texModelName',
                substitutedNames(str(model.get('name', '')), substitutions))
        if all([
                model['tmpTableRowName'] == models[0]['tmpTableRowName']
                for model in models
        ]):
            # IF all the row(model) names are the same, let's not show them. Rather, just put a comment in the comments.
            for model in models:
                model['tmpTableRowName'] = ''
            tableFormat['comments'] = tableFormat.get(
                'comments', '') + ' N.B.: All models/rows were named %s. ' % (
                    models[0].get('texModelName', ''))
            multirowLabels = False

        # Decide whether to separate models by showing their group names as mostly-blank rows:
        mgns = [mm.get('modelGroupName', '') for mm in models]  # unused???
        latestModelGroupName = ''

        # Loop over models, creating a pair of rows for each (coefficients and standard errors)
        for model in models:
            assert 'estcoefs' in model or 'separator' in model  # means not yet programmed
            if 'flags' in model:  # flags must have been turned into a dict or a list of pairs
                assert 'textralines' in model
                # Above replaces lines below, since now the reformatting of flags has been done in textralines:
                #model['flags']=dict(model['flags'])
            if 'estcoefs' in model:  # This is an estimate, not a mean, not a spacer
                mgn = model.get('modelGroupName', ''
                                if latestModelGroupName == '' else '------------')
                if not latestModelGroupName == mgn:
                    latestModelGroupName = mgn
                    body += r'\multicolumn{2}{l}{' + substitutedNames(
                        mgn, substitutions) + ':}' + '\t& '.join(
                            ['' for vv in coefVars + statsVars]) + ' \\\\ \n'
                if model.get('special', '') in ['suestTests']:
                    displayEst = 'p'  # For OLS etc
                    displayErr = 'nothing!'
                    estValues = [
                        r'\sltheadernum{' + model.get('texModelNum', '(%d)' %
                                                      (model.get('modelNum', 0))) +
                        '}', r'\sltrheadername{' + model['tmpTableRowName'] + '}'
                    ] + [
                        dgetget(model, 'estcoefs', vv, displayEst, fNaN)
                        for vv in coefVars
                    ] + [
                        dgetget(model, 'textralines', vv, fNaN) for vv in flagsVars
                    ] + [formatEstStat(model, vv) for vv in statsVars]
                    errValues = ['', ''] + [
                        dgetget(model, 'estcoefs', vv, displayErr, fNaN)
                        for vv in coefVars
                    ] + ['' for vv in flagsVars] + ['' for vv in statsVars]
                    pValues = ['', ''] + [
                        dgetget(model, 'estcoefs', vv, 'p', fNaN)
                        for vv in coefVars
                    ] + ['' for vv in flagsVars] + ['' for vv in statsVars]

                    tworows = formatPairedRow(
                        [estValues, errValues],
                        greycells='tableshading' in model and
                        model['tableshading'] in ['grey'],
                        pValues=pValues)

                else:
                    displayEst = 'b'  # For OLS etc
                    displayErr = 'se'

                    tworows = formatPairedRow(
                        [[
                            r'\sltheadernum{' + model.get('texModelNum', '(%d)' % (
                                model.get('modelNum', 0))) + '}',
                            r'\sltrheadername{' + model['tmpTableRowName'] + '}'
                        ] + [
                            dgetget(model, 'estcoefs', vv, displayEst, fNaN)
                            for vv in coefVars
                        ] + [
                            dgetget(model, 'textralines', vv, fNaN)
                            for vv in flagsVars
                        ] + [formatEstStat(model, vv) for vv in statsVars],
                         ['', ''] + [
                             dgetget(model, 'estcoefs', vv, displayErr, fNaN)
                             for vv in coefVars
                         ] + ['' for vv in flagsVars] + ['' for vv in statsVars]],
                        greycells='tableshading' in model and
                        model['tableshading'] in ['grey'])  #,modelsAsRows=True)

                #multiRow=r'\multirow{2}{*}{\hspace{0}'
                #multiRowEnd='}'

                #print [[multiRow,r'\sltheadernum{',model.get('texModelNum','(%d)'%(model.get('modelNum',0))),'}',multiRowEnd,multiRow, model.get('texModelName',str(model.get('name',''))),multiRowEnd],[dgetget(model,'estcoefs',vv,displayEst,fNaN) for vv in coefVars],[dgetget(model,'textralines',vv,fNaN) for vv in flagsVars],[formatEstStat(model,vv) for vv in statsVars],  ['',''],[dgetget(model,'estcoefs',vv,displayErr,fNaN) for vv in coefVars],['' for vv in flagsVars],['' for vv in statsVars]]
                # BIG BUG IS HERE/BELOW. UNFIXED OCT 2009.

                #tworows=tworows[0:(2-int(suppressSE))]  # Include standard errors?
                if 'special' in model:  #any(['special' in mm for mm in models]):
                    pass

                    #for icol in enumerate(tworows[0]):
                    #    tworows[0][icol]=r'\rowcolor{caggNormal} '+ tworows[0][icol].replace(r'\aggc','')
                    #    tworows[1][icol]=r'\rowcolor{caggNormal} '+ tworows[1][icol].replace(r'\aggc','')

                if r'\aggc' in tworows[0][0]:
                    # APRIL 2010 KLUUUUUUUUUUDGE to get colortbl working with multirow: switch partly to rowcolor! models as cols not done yet...
                    # Use awful kludge so as not to lose the text of second line...  Can just do this for all rows, even not shaded??
                    assert 'sltrheadername' in tworows[0][1]
                    tworows[1][1] = tworows[0][1].replace('sltrheadername',
                                                          'sltrbheadername')
                    tworows[0][1] = ''


                    body+= r'\rowcolor{caggNormal} ' +   ('\t& '.join([cc for cc in tworows[0]])).replace(r'\aggc','')+'\\\\ \n'+  r'\rowcolor{caggNormal} '  + r'\showSEs{'+\
                        '\t& '.join([cc for cc in tworows[1]]) +' \\\\ }{}\n'
                else:
                    body+= ('\t& '.join([cc for cc in tworows[0]]))+'\\\\ \n'+   r'\showSEs{'+\
                        '\t& '.join([cc for cc in tworows[1]]) +' \\\\ }{}\n'

                if model['format'].endswith('|'):
                    body += '\\bottomrule\n'  # or should it be: (r'\cline{1-\ctNtabCols}'+' \n')
            else:
                assert 0

        ntexrows, ntexcols = 1 + (
            1 + int(suppressSE)
        ) * len(models), 2 + len(coefVars + statsVars + flagsVars)

        formats = 'lc*{%d}{r}' % (ntexcols - 2)  # or: 'l'+'c'*nvars
        if multirowLabels:
            formats = 'lp{3cm}*{%d}{r}' % (ntexcols - 2)  # or: 'l'+'c'*nvars

        ###assert not '&' in [cellsvmmodel[0] for cellsvmmodel in cellsvm] # This would be a mistake in regTable caller?
        # Unlike for non-transposed, we aren't using smartHeaders function here...
        headersLine = '\t&'.join(['', ''] + [
            r'\begin{sideways}\sltcheadername{%s}\end{sideways}' %
            substitutedNames(vv, substitutions)
            for vv in coefVars + flagsVars + statsVars
        ]) + '\\\\ \n'  ####cellsvmmodel[0] for cellsvmmodel in cellsvm])+'\\\\ \n'+r'\hline'#\cline{1-\ctNtabCols}'
        headersLine1 = '\\cpbltoprule \n' + headersLine + '\\hline\n'  #r'\cline{1-\ctNtabCols}'+'\n'
        headersLine2 = '\\hline \n ' + headersLine + r'\hline' + '\n'

        modelsAsRowsElements = deepcopy(
            cpblTableElements(
                body=body,
                cformat=formats,
                firstPageHeader=headersLine1,
                otherPageHeader=headersLine2,
                tableTitle=tableFormat.get('title', None),
                caption=tableFormat.get('caption', None),
                label=tableLabel,
                ncols=ntexcols,
                nrows=ntexrows,
                footer=colourLegend() + ' ' + tableFormat.get('comments', None),
                tableName=tableFormat.get('title', None),
                landscape=landscape))

        # Now, let's always put the non-transposed as the default orientation inthe .tex file
        #includeTeX,callerTeX=cpblTableStyC(cpblTableElements(body=body,cformat=formats,firstPageHeader=headersLine1,otherPageHeader=headersLine2,tableTitle=tableFormat.get('title',None),caption=tableFormat.get('caption',None),label=tableLabel, ncols=ntexcols,nrows=ntexrows,footer=colourLegend()+' '+tableFormat.get('comments',None),tableName=tableFormat.get('title',None),landscape=landscape))
        includeTeX, callerTeX = cpblTableStyC(
            tableElements=varsAsRowsElements,
            tableElementsTrans=modelsAsRowsElements,
            showTransposed=transposed)
        if multirowLabels:
            callerTeX = r"""\renewcommand{\sltrheadername}[1]{\multirow{2}{3cm}{\hspace{0pt}#1\vfill}}
    \renewcommand{\sltrbheadername}[1]{\multirow{-2}{3cm}{\hspace{0pt}#1\vfill}}
    """ + callerTeX

        return (includeTeX, callerTeX, transposed)


from cpblUtilities import tonumeric
def modelResultsByVar_df(modelResults, tableFilename=None):
    ################################################################################################
    ################################################################################################
    """
Unlike the old pystata version, this just deals with a DataFrame and chooses three ordered subsets of its columns!
    """
    nullValues2 = ['.', '', '0', tonumeric('')]

    haoooooopy
    # Get list of all
    import operator
    allvarsM = uniqueInOrder(
        reduce(operator.add, [mm['estcoefs'].keys()
                              for mm in modelResults], []))
    allstatsM = uniqueInOrder(
        reduce(operator.add, [mm['eststats'].keys()
                              for mm in modelResults], []))
    allTextralinesM = uniqueInOrder(
        reduce(operator.add, [
            mm['textralines'].keys() for mm in modelResults
            if 'textralines' in mm
        ], []))

    #create dict with rows:
    byVar = {}
    for vv in allvarsM:
        vvname = vv
        #if vvname.startswith('z_'):
        #    vvname=vvname[2:]
        displayCoef = [
            ['b', 'p'][int(mmm.get('special', '') in ['suestTests'])]
            for mmm in modelResults
        ]
        byVar[vvname] = {
            'coefs': [
                mmm['estcoefs'].get(vv, {}).get(displayCoef[imm], fNaN)
                for imm, mmm in enumerate(modelResults)
            ],
            'ses': [
                mmm['estcoefs'].get(vv, {}).get('se', fNaN)
                for mmm in modelResults
            ],
            'p': [
                mmm['estcoefs'].get(vv, {}).get('p', fNaN)
                for mmm in modelResults
            ]
        }
    byStat = {}
    for vv in allstatsM:
        byStat[vv] = [mmm['eststats'].get(vv, fNaN) for mmm in modelResults]
    """ What if something exists as a stat in one model but as a textraline in a different model? In this case, the textraline should be moved to stats for all models.
    (  This may only be for N_clust)
    """
    for vv in allTextralinesM:  # Anything that is a textraline in ANY model
        if vv in byStat:  # Anything that is a stat in ANY model
            for mmm in modelResults:
                # First, we should fail if there are conflicts for this model:
                assert not dgetget(
                    mmm, ['textralines', vv], '') or not dgetget(
                        mmm, ['eststats', vv], '') or dgetget(
                            mmm, ['textralines', vv], '') == dgetget(
                                mmm, ['eststats', vv], '')
                # Otherwise, let's move this from textralines to stats in this model:
                if dgetget(mmm, ['textralines', vv], ''):
                    mmm['eststats'][vv] = dgetget(mmm, ['textralines', vv], '')
            mmm['textralines'][
                vv] = ''  # Rather than deleting it, set it to blank? It'll get dropped anyway.
            print '    textralines: Moved ' + vv + ' to stats '
            assert not any([
                dgetget(mmm, ['textralines', vv], '') for mmm in modelResults
            ])
            # Now, remove this item from the textralines that will be displayed!
            allTextralinesM = [tlm for tlm in allTextralinesM if not tlm == vv]
            # Note that in general I do not clean up empty textralines, except in this case, because they might be specified on purpose to put an empty space in a table? hmm, no: I could use "~" for that.

        # Clean up empty rows (unused regressors):
    kk = byVar.keys()
    droppedNames = []
    for vv in kk:
        isVal = [btp not in nullValues2 for btp in byVar[vv]['coefs']]
        if not any(isVal):
            del byVar[vv]
            droppedNames += [vv]
    if droppedNames:
        if tableFilename:
            print " modelResultsByVar: Dropping variables with no table entries from %s!: " % os.path.split(
                tableFilename)[1], droppedNames

    # Construct the "extra lines", ie the attributes that are not simple regressors:
    # First, get a list of attribute pairs specified for each model; ie parse the various ways they can be listed (done, above)
    byTextraline = {}
    for vv in allTextralinesM:
        byTextraline[vv] = [
            dgetget(mmm, 'textralines', vv, '') for mmm in modelResults
        ]

    # Should this be here? Sept 2009. It's other places too, right now..
    # Drop r2 if we have r2_a:
    if 'r2_a' in byStat and all(byStat['r2_a']) and 'r2' in byStat:
        del byStat['r2']
        debugprint('Dropping r2 in favour of r2_a.')

    from pylab import isnan
    assert 'N_clust' not in byStat or byStat['N_clust'][0] > 1 or isnan(
        byStat['N_clust'][0])  #Bug check. ? against what?
    #    assert not any([kk in byTextraline for kk in byStat])

    return (byVar, byStat, byTextraline)



def test_all():
    foo












