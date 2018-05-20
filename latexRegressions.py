#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

See the plan.org for plans.

This starts by providing interfaces from statistical routines, or their output, to regression model objects in the tradition of my pystata package.

to keep it flexible, a regmodel object is a dict with various elements. They include a DataFrame
of coefficients or marginal effects.

"""


import pandas as pd
import statsmodels
def from_statsmodels(res, **args): #flags=None, name=None, ):
    """ Take result object from a statsmodels estimate, and put in into a latexRegressions model dict format
    """
    rm = {'engine':'statsmodels',
          'method': {
              statsmodels.regression.linear_model.OLS: 'ols'
          }.get(res.model.__class__, 'unknown'),
          #'resultobject':
          'estcoefs': pd.DataFrame({'b':res.params,
                                    'p':res.pvalues,
                                   't':res.tvalues,
                                   'se':res.bse,
                                  }),
          'estparams': {'r2':res.rsquared,
                       'r2a':res.rsquared_adj,
                       'N': res.nobs,
                       },
          'summary': res.summary2(),
          'rawLogfileOutput': None, # rename this to rawStataLogFileOutput? And remove it when non-stata?
          
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
    def appendRegressionTable(
            self,
            models,
            suppressSE=False,
            substitutions=None,
            transposed=None,
            tableFilePath=None,
            ):
        sosoie
