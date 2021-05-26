# -*- coding: utf-8 -*-
'''
Abrir .tex em python como string, e tentar colocar figuras
e textos de forma automatica nos locais corretos

pintar celula de vermelho
\cellcolor{red!25}

'''

import numpy as np
import os
import jinja2
from jinja2 import Template
import datetime as dt
import codecs
from datetime import datetime
import pandas as pd

#caminho do template .tex (boletim_TU.tex)
pathname = os.environ['HOME'] + '/Dropbox/Previsao/pnboia/resultados/'
pathname_template = os.environ['HOME'] + '/Dropbox/pnboia/boletim/'

#lista diretorio das previsoes
prevs = np.sort(os.listdir(pathname))

#previsao dia -7
rg = pd.read_table(pathname + prevs[-1] + '/PNBOIA_riogrande.txt',sep='\s*',
                     names=['ano','mes','dia','hora','minu','hs','tp','dp','spr'])

fl = pd.read_table(pathname + prevs[-1] + '/PNBOIA_floripa.txt',sep='\s*',
                     names=['ano','mes','dia','hora','minu','hs','tp','dp','spr'])

sa = pd.read_table(pathname + prevs[-1] + '/PNBOIA_santos.txt',sep='\s*',
                     names=['ano','mes','dia','hora','minu','hs','tp','dp','spr'])


#cria data com datetime
rg['data'] = pd.to_datetime(rg.ano.astype(str) + rg.mes.astype(str) + rg.dia.astype(str) + rg.hora.astype(str) ,format="%Y%m%d%H")
fl['data'] = pd.to_datetime(fl.ano.astype(str) + fl.mes.astype(str) + fl.dia.astype(str) + fl.hora.astype(str) ,format="%Y%m%d%H")
sa['data'] = pd.to_datetime(sa.ano.astype(str) + sa.mes.astype(str) + sa.dia.astype(str) + sa.hora.astype(str) ,format="%Y%m%d%H")


#deixa a data como indice
rg = rg.set_index('data')
fl = fl.set_index('data')
sa = sa.set_index('data')

###############################################################################
###############################################################################
###############################################################################
# Rio Grande


tab1_rg = {'hs' : ' & ' + '%.2f' %rg.hs[0] + ' & ' + '%.2f' %rg.hs[3] + ' & ' + '%.2f' %rg.hs[6] + ' & ' + '%.2f' %rg.hs[9] + ' & ' + \
                  '%.2f' %rg.hs[12] + ' & ' + '%.2f' %rg.hs[15] + ' & ' + '%.2f' %rg.hs[18] + ' & ' + '%.2f' %rg.hs[21],
           'tp' : ' & ' + '%.1f' %rg.tp[0] + ' & ' + '%.1f' %rg.tp[3] + ' & ' + '%.1f' %rg.tp[6] + ' & ' + '%.1f' %rg.tp[9] + ' & ' + \
                  '%.1f' %rg.tp[12] + ' & ' + '%.1f' %rg.tp[15] + ' & ' + '%.1f' %rg.tp[18] + ' & ' + '%.1f' %rg.tp[21],
           'dp' : ' & ' + '%.i' %rg.dp[0] + ' & ' + '%.i' %rg.dp[3] + ' & ' + '%.i' %rg.dp[6] + ' & ' + '%.i' %rg.dp[9] + ' & ' + \
                  '%.i' %rg.dp[12] + ' & ' + '%.i' %rg.dp[15] + ' & ' + '%.i' %rg.dp[18] + ' & ' + '%.i' %rg.dp[21],
           'spr' : ' & ' + '%.i' %rg.spr[0] + ' & ' + '%.i' %rg.spr[3] + ' & ' + '%.i' %rg.spr[6] + ' & ' + '%.i' %rg.spr[9] + ' & ' + \
                  '%i' %rg.spr[12] + ' & ' + '%i' %rg.spr[15] + ' & ' + '%.i' %rg.spr[18] + ' & ' + '%.i' %rg.spr[21]
          }

tab2_rg = {'hs' : ' & ' + '%.2f' %rg.hs[24] + ' & ' + '%.2f' %rg.hs[27] + ' & ' + '%.2f' %rg.hs[30] + ' & ' + '%.2f' %rg.hs[33] + ' & ' + \
                  '%.2f' %rg.hs[36] + ' & ' + '%.2f' %rg.hs[39] + ' & ' + '%.2f' %rg.hs[42] + ' & ' + '%.2f' %rg.hs[45],
           'tp' : ' & ' + '%.1f' %rg.tp[24] + ' & ' + '%.1f' %rg.tp[27] + ' & ' + '%.1f' %rg.tp[30] + ' & ' + '%.1f' %rg.tp[33] + ' & ' + \
                  '%.1f' %rg.tp[36] + ' & ' + '%.1f' %rg.tp[39] + ' & ' + '%.1f' %rg.tp[42] + ' & ' + '%.1f' %rg.tp[45],
           'dp' : ' & ' + '%.2f' %rg.dp[24] + ' & ' + '%.2f' %rg.dp[27] + ' & ' + '%.2f' %rg.dp[30] + ' & ' + '%.2f' %rg.dp[33] + ' & ' + \
                  '%i' %rg.dp[36] + ' & ' + '%i' %rg.dp[39] + ' & ' + '%i' %rg.dp[42] + ' & ' + '%i' %rg.dp[45],
           'spr' : ' & ' + '%i' %rg.spr[24] + ' & ' + '%i' %rg.spr[27] + ' & ' + '%i' %rg.spr[30] + ' & ' + '%i' %rg.spr[33] + ' & ' + \
                  '%i' %rg.spr[36] + ' & ' + '%i' %rg.spr[39] + ' & ' + '%i' %rg.spr[42] + ' & ' + '%i' %rg.spr[45]
          }

tab3_rg = {'hs' : ' & ' + '%.2f' %rg.hs[48] + ' & ' + '%.2f' %rg.hs[51] + ' & ' + '%.2f' %rg.hs[54] + ' & ' + '%.2f' %rg.hs[57] + ' & ' + \
                  '%.2f' %rg.hs[60] + ' & ' + '%.2f' %rg.hs[63] + ' & ' + '%.2f' %rg.hs[66] + ' & ' + '%.2f' %rg.hs[69],
           'tp' : ' & ' + '%.1f' %rg.tp[48] + ' & ' + '%.1f' %rg.tp[51] + ' & ' + '%.1f' %rg.tp[54] + ' & ' + '%.1f' %rg.tp[57] + ' & ' + \
                  '%.1f' %rg.tp[60] + ' & ' + '%.1f' %rg.tp[63] + ' & ' + '%.1f' %rg.tp[66] + ' & ' + '%.1f' %rg.tp[69],
           'dp' : ' & ' + '%i' %rg.dp[48] + ' & ' + '%i' %rg.dp[51] + ' & ' + '%i' %rg.dp[54] + ' & ' + '%i' %rg.dp[57] + ' & ' + \
                  '%i' %rg.dp[60] + ' & ' + '%i' %rg.dp[63] + ' & ' + '%.i' %rg.dp[66] + ' & ' + '%.i' %rg.dp[69],
           'spr' : ' & ' + '%.i' %rg.spr[48] + ' & ' + '%i' %rg.spr[51] + ' & ' + '%i' %rg.spr[54] + ' & ' + '%i' %rg.spr[57] + ' & ' + \
                  '%.i' %rg.spr[60] + ' & ' + '%.i' %rg.spr[63] + ' & ' + '%.i' %rg.spr[66] + ' & ' + '%.i' %rg.spr[69]
          }

tab4_rg = {'hs' : ' & ' + '%.2f' %rg.hs[72] + ' & ' + '%.2f' %rg.hs[75] + ' & ' + '%.2f' %rg.hs[78] + ' & ' + '%.2f' %rg.hs[81] + ' & ' + \
                  '%.2f' %rg.hs[84] + ' & ' + '%.2f' %rg.hs[87] + ' & ' + '%.2f' %rg.hs[90] + ' & ' + '%.2f' %rg.hs[93],
           'tp' : ' & ' + '%.1f' %rg.tp[72] + ' & ' + '%.1f' %rg.tp[75] + ' & ' + '%.1f' %rg.tp[78] + ' & ' + '%.1f' %rg.tp[81] + ' & ' + \
                  '%.1f' %rg.tp[84] + ' & ' + '%.1f' %rg.tp[87] + ' & ' + '%.1f' %rg.tp[90] + ' & ' + '%.1f' %rg.tp[93],
           'dp' : ' & ' + '%i' %rg.dp[72] + ' & ' + '%i' %rg.dp[75] + ' & ' + '%i' %rg.dp[78] + ' & ' + '%i' %rg.dp[81] + ' & ' + \
                  '%i' %rg.dp[84] + ' & ' + '%i' %rg.dp[87] + ' & ' + '%.i' %rg.dp[90] + ' & ' + '%.i' %rg.dp[93],
           'spr' : ' & ' + '%.i' %rg.spr[72] + ' & ' + '%i' %rg.spr[75] + ' & ' + '%i' %rg.spr[78] + ' & ' + '%i' %rg.spr[81] + ' & ' + \
                  '%.i' %rg.spr[84] + ' & ' + '%.i' %rg.spr[87] + ' & ' + '%.i' %rg.spr[90] + ' & ' + '%.i' %rg.spr[93]
          }

tab5_rg = {'hs' : ' & ' + '%.2f' %rg.hs[96] + ' & ' + '%.2f' %rg.hs[99] + ' & ' + '%.2f' %rg.hs[102] + ' & ' + '%.2f' %rg.hs[105] + ' & ' + \
                  '%.2f' %rg.hs[108] + ' & ' + '%.2f' %rg.hs[111] + ' & ' + '%.2f' %rg.hs[114] + ' & ' + '%.2f' %rg.hs[117],
           'tp' : ' & ' + '%.1f' %rg.tp[96] + ' & ' + '%.1f' %rg.tp[99] + ' & ' + '%.1f' %rg.tp[102] + ' & ' + '%.1f' %rg.tp[104] + ' & ' + \
                  '%.1f' %rg.tp[108] + ' & ' + '%.1f' %rg.tp[111] + ' & ' + '%.1f' %rg.tp[114] + ' & ' + '%.1f' %rg.tp[117],
           'dp' : ' & ' + '%i' %rg.dp[96] + ' & ' + '%i' %rg.dp[99] + ' & ' + '%i' %rg.dp[102] + ' & ' + '%i' %rg.dp[105] + ' & ' + \
                  '%i' %rg.dp[108] + ' & ' + '%i' %rg.dp[111] + ' & ' + '%.i' %rg.dp[113] + ' & ' + '%.i' %rg.dp[117],
           'spr' : ' & ' + '%.i' %rg.spr[96] + ' & ' + '%i' %rg.spr[99] + ' & ' + '%i' %rg.spr[102] + ' & ' + '%i' %rg.spr[105] + ' & ' + \
                  '%.i' %rg.spr[108] + ' & ' + '%.i' %rg.spr[111] + ' & ' + '%.i' %rg.spr[114] + ' & ' + '%.i' %rg.spr[117]
          }

tab6_rg = {'hs' : ' & ' + '%.2f' %rg.hs[120] + ' & ' + '%.2f' %rg.hs[123] + ' & ' + '%.2f' %rg.hs[126] + ' & ' + '%.2f' %rg.hs[129] + ' & ' + \
                  '%.2f' %rg.hs[132] + ' & ' + '%.2f' %rg.hs[135] + ' & ' + '%.2f' %rg.hs[138] + ' & ' + '%.2f' %rg.hs[141],
           'tp' : ' & ' + '%.1f' %rg.tp[120] + ' & ' + '%.1f' %rg.tp[123] + ' & ' + '%.1f' %rg.tp[126] + ' & ' + '%.1f' %rg.tp[129] + ' & ' + \
                  '%.1f' %rg.tp[132] + ' & ' + '%.1f' %rg.tp[135] + ' & ' + '%.1f' %rg.tp[138] + ' & ' + '%.1f' %rg.tp[141],
           'dp' : ' & ' + '%i' %rg.dp[120] + ' & ' + '%i' %rg.dp[123] + ' & ' + '%i' %rg.dp[126] + ' & ' + '%i' %rg.dp[129] + ' & ' + \
                  '%i' %rg.dp[132] + ' & ' + '%i' %rg.dp[135] + ' & ' + '%.i' %rg.dp[138] + ' & ' + '%.i' %rg.dp[141],
           'spr' : ' & ' + '%.i' %rg.spr[120] + ' & ' + '%i' %rg.spr[123] + ' & ' + '%i' %rg.spr[126] + ' & ' + '%i' %rg.spr[128] + ' & ' + \
                  '%.i' %rg.spr[132] + ' & ' + '%.i' %rg.spr[135] + ' & ' + '%.i' %rg.spr[138] + ' & ' + '%.i' %rg.spr[141]
          }

tab7_rg = {'hs' : ' & ' + '%.2f' %rg.hs[144] + ' & ' + '%.2f' %rg.hs[147] + ' & ' + '%.2f' %rg.hs[150] + ' & ' + '%.2f' %rg.hs[153] + ' & ' + \
                  '%.2f' %rg.hs[156] + ' & ' + '%.2f' %rg.hs[159] + ' & ' + '%.2f' %rg.hs[162] + ' & ' + '%.2f' %rg.hs[165],
           'tp' : ' & ' + '%.1f' %rg.tp[144] + ' & ' + '%.1f' %rg.tp[147] + ' & ' + '%.1f' %rg.tp[150] + ' & ' + '%.1f' %rg.tp[153] + ' & ' + \
                  '%.1f' %rg.tp[156] + ' & ' + '%.1f' %rg.tp[159] + ' & ' + '%.1f' %rg.tp[162] + ' & ' + '%.1f' %rg.tp[165],
           'dp' : ' & ' + '%i' %rg.dp[144] + ' & ' + '%i' %rg.dp[147] + ' & ' + '%i' %rg.dp[150] + ' & ' + '%i' %rg.dp[153] + ' & ' + \
                  '%i' %rg.dp[156] + ' & ' + '%i' %rg.dp[159] + ' & ' + '%.i' %rg.dp[162] + ' & ' + '%.i' %rg.dp[165],
           'spr' : ' & ' + '%.i' %rg.spr[144] + ' & ' + '%i' %rg.spr[147] + ' & ' + '%i' %rg.spr[150] + ' & ' + '%i' %rg.spr[153] + ' & ' + \
                  '%.i' %rg.spr[156] + ' & ' + '%.i' %rg.spr[159] + ' & ' + '%.i' %rg.spr[162] + ' & ' + '%.i' %rg.spr[165]
          }


###############################################################################
###############################################################################
###############################################################################
# Rio Grande


tab1_fl = {'hs' : ' & ' + '%.2f' %fl.hs[0] + ' & ' + '%.2f' %fl.hs[3] + ' & ' + '%.2f' %fl.hs[6] + ' & ' + '%.2f' %fl.hs[9] + ' & ' + \
                  '%.2f' %fl.hs[12] + ' & ' + '%.2f' %fl.hs[15] + ' & ' + '%.2f' %fl.hs[18] + ' & ' + '%.2f' %fl.hs[21],
           'tp' : ' & ' + '%.1f' %fl.tp[0] + ' & ' + '%.1f' %fl.tp[3] + ' & ' + '%.1f' %fl.tp[6] + ' & ' + '%.1f' %fl.tp[9] + ' & ' + \
                  '%.1f' %fl.tp[12] + ' & ' + '%.1f' %fl.tp[15] + ' & ' + '%.1f' %fl.tp[18] + ' & ' + '%.1f' %fl.tp[21],
           'dp' : ' & ' + '%.i' %fl.dp[0] + ' & ' + '%.i' %fl.dp[3] + ' & ' + '%.i' %fl.dp[6] + ' & ' + '%.i' %fl.dp[9] + ' & ' + \
                  '%.i' %fl.dp[12] + ' & ' + '%.i' %fl.dp[15] + ' & ' + '%.i' %fl.dp[18] + ' & ' + '%.i' %fl.dp[21],
           'spr' : ' & ' + '%.i' %fl.spr[0] + ' & ' + '%.i' %fl.spr[3] + ' & ' + '%.i' %fl.spr[6] + ' & ' + '%.i' %fl.spr[9] + ' & ' + \
                  '%i' %fl.spr[12] + ' & ' + '%i' %fl.spr[15] + ' & ' + '%.i' %fl.spr[18] + ' & ' + '%.i' %fl.spr[21]
          }

tab2_fl = {'hs' : ' & ' + '%.2f' %fl.hs[24] + ' & ' + '%.2f' %fl.hs[27] + ' & ' + '%.2f' %fl.hs[30] + ' & ' + '%.2f' %fl.hs[33] + ' & ' + \
                  '%.2f' %fl.hs[36] + ' & ' + '%.2f' %fl.hs[39] + ' & ' + '%.2f' %fl.hs[42] + ' & ' + '%.2f' %fl.hs[45],
           'tp' : ' & ' + '%.1f' %fl.tp[24] + ' & ' + '%.1f' %fl.tp[27] + ' & ' + '%.1f' %fl.tp[30] + ' & ' + '%.1f' %fl.tp[33] + ' & ' + \
                  '%.1f' %fl.tp[36] + ' & ' + '%.1f' %fl.tp[39] + ' & ' + '%.1f' %fl.tp[42] + ' & ' + '%.1f' %fl.tp[45],
           'dp' : ' & ' + '%.2f' %fl.dp[24] + ' & ' + '%.2f' %fl.dp[27] + ' & ' + '%.2f' %fl.dp[30] + ' & ' + '%.2f' %fl.dp[33] + ' & ' + \
                  '%i' %fl.dp[36] + ' & ' + '%i' %fl.dp[39] + ' & ' + '%i' %fl.dp[42] + ' & ' + '%i' %fl.dp[45],
           'spr' : ' & ' + '%i' %fl.spr[24] + ' & ' + '%i' %fl.spr[27] + ' & ' + '%i' %fl.spr[30] + ' & ' + '%i' %fl.spr[33] + ' & ' + \
                  '%i' %fl.spr[36] + ' & ' + '%i' %fl.spr[39] + ' & ' + '%i' %fl.spr[42] + ' & ' + '%i' %fl.spr[45]
          }

tab3_fl = {'hs' : ' & ' + '%.2f' %fl.hs[48] + ' & ' + '%.2f' %fl.hs[51] + ' & ' + '%.2f' %fl.hs[54] + ' & ' + '%.2f' %fl.hs[57] + ' & ' + \
                  '%.2f' %fl.hs[60] + ' & ' + '%.2f' %fl.hs[63] + ' & ' + '%.2f' %fl.hs[66] + ' & ' + '%.2f' %fl.hs[69],
           'tp' : ' & ' + '%.1f' %fl.tp[48] + ' & ' + '%.1f' %fl.tp[51] + ' & ' + '%.1f' %fl.tp[54] + ' & ' + '%.1f' %fl.tp[57] + ' & ' + \
                  '%.1f' %fl.tp[60] + ' & ' + '%.1f' %fl.tp[63] + ' & ' + '%.1f' %fl.tp[66] + ' & ' + '%.1f' %fl.tp[69],
           'dp' : ' & ' + '%i' %fl.dp[48] + ' & ' + '%i' %fl.dp[51] + ' & ' + '%i' %fl.dp[54] + ' & ' + '%i' %fl.dp[57] + ' & ' + \
                  '%i' %fl.dp[60] + ' & ' + '%i' %fl.dp[63] + ' & ' + '%.i' %fl.dp[66] + ' & ' + '%.i' %fl.dp[69],
           'spr' : ' & ' + '%.i' %fl.spr[48] + ' & ' + '%i' %fl.spr[51] + ' & ' + '%i' %fl.spr[54] + ' & ' + '%i' %fl.spr[57] + ' & ' + \
                  '%.i' %fl.spr[60] + ' & ' + '%.i' %fl.spr[63] + ' & ' + '%.i' %fl.spr[66] + ' & ' + '%.i' %fl.spr[69]
          }

tab4_fl = {'hs' : ' & ' + '%.2f' %fl.hs[72] + ' & ' + '%.2f' %fl.hs[75] + ' & ' + '%.2f' %fl.hs[78] + ' & ' + '%.2f' %fl.hs[81] + ' & ' + \
                  '%.2f' %fl.hs[84] + ' & ' + '%.2f' %fl.hs[87] + ' & ' + '%.2f' %fl.hs[90] + ' & ' + '%.2f' %fl.hs[93],
           'tp' : ' & ' + '%.1f' %fl.tp[72] + ' & ' + '%.1f' %fl.tp[75] + ' & ' + '%.1f' %fl.tp[78] + ' & ' + '%.1f' %fl.tp[81] + ' & ' + \
                  '%.1f' %fl.tp[84] + ' & ' + '%.1f' %fl.tp[87] + ' & ' + '%.1f' %fl.tp[90] + ' & ' + '%.1f' %fl.tp[93],
           'dp' : ' & ' + '%i' %fl.dp[72] + ' & ' + '%i' %fl.dp[75] + ' & ' + '%i' %fl.dp[78] + ' & ' + '%i' %fl.dp[81] + ' & ' + \
                  '%i' %fl.dp[84] + ' & ' + '%i' %fl.dp[87] + ' & ' + '%.i' %fl.dp[90] + ' & ' + '%.i' %fl.dp[93],
           'spr' : ' & ' + '%.i' %fl.spr[72] + ' & ' + '%i' %fl.spr[75] + ' & ' + '%i' %fl.spr[78] + ' & ' + '%i' %fl.spr[81] + ' & ' + \
                  '%.i' %fl.spr[84] + ' & ' + '%.i' %fl.spr[87] + ' & ' + '%.i' %fl.spr[90] + ' & ' + '%.i' %fl.spr[93]
          }

tab5_fl = {'hs' : ' & ' + '%.2f' %fl.hs[96] + ' & ' + '%.2f' %fl.hs[99] + ' & ' + '%.2f' %fl.hs[102] + ' & ' + '%.2f' %fl.hs[105] + ' & ' + \
                  '%.2f' %fl.hs[108] + ' & ' + '%.2f' %fl.hs[111] + ' & ' + '%.2f' %fl.hs[114] + ' & ' + '%.2f' %fl.hs[117],
           'tp' : ' & ' + '%.1f' %fl.tp[96] + ' & ' + '%.1f' %fl.tp[99] + ' & ' + '%.1f' %fl.tp[102] + ' & ' + '%.1f' %fl.tp[104] + ' & ' + \
                  '%.1f' %fl.tp[108] + ' & ' + '%.1f' %fl.tp[111] + ' & ' + '%.1f' %fl.tp[114] + ' & ' + '%.1f' %fl.tp[117],
           'dp' : ' & ' + '%i' %fl.dp[96] + ' & ' + '%i' %fl.dp[99] + ' & ' + '%i' %fl.dp[102] + ' & ' + '%i' %fl.dp[105] + ' & ' + \
                  '%i' %fl.dp[108] + ' & ' + '%i' %fl.dp[111] + ' & ' + '%.i' %fl.dp[113] + ' & ' + '%.i' %fl.dp[117],
           'spr' : ' & ' + '%.i' %fl.spr[96] + ' & ' + '%i' %fl.spr[99] + ' & ' + '%i' %fl.spr[102] + ' & ' + '%i' %fl.spr[105] + ' & ' + \
                  '%.i' %fl.spr[108] + ' & ' + '%.i' %fl.spr[111] + ' & ' + '%.i' %fl.spr[114] + ' & ' + '%.i' %fl.spr[117]
          }

tab6_fl = {'hs' : ' & ' + '%.2f' %fl.hs[120] + ' & ' + '%.2f' %fl.hs[123] + ' & ' + '%.2f' %fl.hs[126] + ' & ' + '%.2f' %fl.hs[129] + ' & ' + \
                  '%.2f' %fl.hs[132] + ' & ' + '%.2f' %fl.hs[135] + ' & ' + '%.2f' %fl.hs[138] + ' & ' + '%.2f' %fl.hs[141],
           'tp' : ' & ' + '%.1f' %fl.tp[120] + ' & ' + '%.1f' %fl.tp[123] + ' & ' + '%.1f' %fl.tp[126] + ' & ' + '%.1f' %fl.tp[129] + ' & ' + \
                  '%.1f' %fl.tp[132] + ' & ' + '%.1f' %fl.tp[135] + ' & ' + '%.1f' %fl.tp[138] + ' & ' + '%.1f' %fl.tp[141],
           'dp' : ' & ' + '%i' %fl.dp[120] + ' & ' + '%i' %fl.dp[123] + ' & ' + '%i' %fl.dp[126] + ' & ' + '%i' %fl.dp[129] + ' & ' + \
                  '%i' %fl.dp[132] + ' & ' + '%i' %fl.dp[135] + ' & ' + '%.i' %fl.dp[138] + ' & ' + '%.i' %fl.dp[141],
           'spr' : ' & ' + '%.i' %fl.spr[120] + ' & ' + '%i' %fl.spr[123] + ' & ' + '%i' %fl.spr[126] + ' & ' + '%i' %fl.spr[128] + ' & ' + \
                  '%.i' %fl.spr[132] + ' & ' + '%.i' %fl.spr[135] + ' & ' + '%.i' %fl.spr[138] + ' & ' + '%.i' %fl.spr[141]
          }

tab7_fl = {'hs' : ' & ' + '%.2f' %fl.hs[144] + ' & ' + '%.2f' %fl.hs[147] + ' & ' + '%.2f' %fl.hs[150] + ' & ' + '%.2f' %fl.hs[153] + ' & ' + \
                  '%.2f' %fl.hs[156] + ' & ' + '%.2f' %fl.hs[159] + ' & ' + '%.2f' %fl.hs[162] + ' & ' + '%.2f' %fl.hs[165],
           'tp' : ' & ' + '%.1f' %fl.tp[144] + ' & ' + '%.1f' %fl.tp[147] + ' & ' + '%.1f' %fl.tp[150] + ' & ' + '%.1f' %fl.tp[153] + ' & ' + \
                  '%.1f' %fl.tp[156] + ' & ' + '%.1f' %fl.tp[159] + ' & ' + '%.1f' %fl.tp[162] + ' & ' + '%.1f' %fl.tp[165],
           'dp' : ' & ' + '%i' %fl.dp[144] + ' & ' + '%i' %fl.dp[147] + ' & ' + '%i' %fl.dp[150] + ' & ' + '%i' %fl.dp[153] + ' & ' + \
                  '%i' %fl.dp[156] + ' & ' + '%i' %fl.dp[159] + ' & ' + '%.i' %fl.dp[162] + ' & ' + '%.i' %fl.dp[165],
           'spr' : ' & ' + '%.i' %fl.spr[144] + ' & ' + '%i' %fl.spr[147] + ' & ' + '%i' %fl.spr[150] + ' & ' + '%i' %fl.spr[153] + ' & ' + \
                  '%.i' %fl.spr[156] + ' & ' + '%.i' %fl.spr[159] + ' & ' + '%.i' %fl.spr[162] + ' & ' + '%.i' %fl.spr[165]
          }



###############################################################################
###############################################################################
###############################################################################
# Rio Grande

tab1_sa = {'hs' : ' & ' + '%.2f' %sa.hs[0] + ' & ' + '%.2f' %sa.hs[3] + ' & ' + '%.2f' %sa.hs[6] + ' & ' + '%.2f' %sa.hs[9] + ' & ' + \
                  '%.2f' %sa.hs[12] + ' & ' + '%.2f' %sa.hs[15] + ' & ' + '%.2f' %sa.hs[18] + ' & ' + '%.2f' %sa.hs[21],
           'tp' : ' & ' + '%.1f' %sa.tp[0] + ' & ' + '%.1f' %sa.tp[3] + ' & ' + '%.1f' %sa.tp[6] + ' & ' + '%.1f' %sa.tp[9] + ' & ' + \
                  '%.1f' %sa.tp[12] + ' & ' + '%.1f' %sa.tp[15] + ' & ' + '%.1f' %sa.tp[18] + ' & ' + '%.1f' %sa.tp[21],
           'dp' : ' & ' + '%.i' %sa.dp[0] + ' & ' + '%.i' %sa.dp[3] + ' & ' + '%.i' %sa.dp[6] + ' & ' + '%.i' %sa.dp[9] + ' & ' + \
                  '%.i' %sa.dp[12] + ' & ' + '%.i' %sa.dp[15] + ' & ' + '%.i' %sa.dp[18] + ' & ' + '%.i' %sa.dp[21],
           'spr' : ' & ' + '%.i' %sa.spr[0] + ' & ' + '%.i' %sa.spr[3] + ' & ' + '%.i' %sa.spr[6] + ' & ' + '%.i' %sa.spr[9] + ' & ' + \
                  '%i' %sa.spr[12] + ' & ' + '%i' %sa.spr[15] + ' & ' + '%.i' %sa.spr[18] + ' & ' + '%.i' %sa.spr[21]
          }

tab2_sa = {'hs' : ' & ' + '%.2f' %sa.hs[24] + ' & ' + '%.2f' %sa.hs[27] + ' & ' + '%.2f' %sa.hs[30] + ' & ' + '%.2f' %sa.hs[33] + ' & ' + \
                  '%.2f' %sa.hs[36] + ' & ' + '%.2f' %sa.hs[39] + ' & ' + '%.2f' %sa.hs[42] + ' & ' + '%.2f' %sa.hs[45],
           'tp' : ' & ' + '%.1f' %sa.tp[24] + ' & ' + '%.1f' %sa.tp[27] + ' & ' + '%.1f' %sa.tp[30] + ' & ' + '%.1f' %sa.tp[33] + ' & ' + \
                  '%.1f' %sa.tp[36] + ' & ' + '%.1f' %sa.tp[39] + ' & ' + '%.1f' %sa.tp[42] + ' & ' + '%.1f' %sa.tp[45],
           'dp' : ' & ' + '%.2f' %sa.dp[24] + ' & ' + '%.2f' %sa.dp[27] + ' & ' + '%.2f' %sa.dp[30] + ' & ' + '%.2f' %sa.dp[33] + ' & ' + \
                  '%i' %sa.dp[36] + ' & ' + '%i' %sa.dp[39] + ' & ' + '%i' %sa.dp[42] + ' & ' + '%i' %sa.dp[45],
           'spr' : ' & ' + '%i' %sa.spr[24] + ' & ' + '%i' %sa.spr[27] + ' & ' + '%i' %sa.spr[30] + ' & ' + '%i' %sa.spr[33] + ' & ' + \
                  '%i' %sa.spr[36] + ' & ' + '%i' %sa.spr[39] + ' & ' + '%i' %sa.spr[42] + ' & ' + '%i' %sa.spr[45]
          }

tab3_sa = {'hs' : ' & ' + '%.2f' %sa.hs[48] + ' & ' + '%.2f' %sa.hs[51] + ' & ' + '%.2f' %sa.hs[54] + ' & ' + '%.2f' %sa.hs[57] + ' & ' + \
                  '%.2f' %sa.hs[60] + ' & ' + '%.2f' %sa.hs[63] + ' & ' + '%.2f' %sa.hs[66] + ' & ' + '%.2f' %sa.hs[69],
           'tp' : ' & ' + '%.1f' %sa.tp[48] + ' & ' + '%.1f' %sa.tp[51] + ' & ' + '%.1f' %sa.tp[54] + ' & ' + '%.1f' %sa.tp[57] + ' & ' + \
                  '%.1f' %sa.tp[60] + ' & ' + '%.1f' %sa.tp[63] + ' & ' + '%.1f' %sa.tp[66] + ' & ' + '%.1f' %sa.tp[69],
           'dp' : ' & ' + '%i' %sa.dp[48] + ' & ' + '%i' %sa.dp[51] + ' & ' + '%i' %sa.dp[54] + ' & ' + '%i' %sa.dp[57] + ' & ' + \
                  '%i' %sa.dp[60] + ' & ' + '%i' %sa.dp[63] + ' & ' + '%.i' %sa.dp[66] + ' & ' + '%.i' %sa.dp[69],
           'spr' : ' & ' + '%.i' %sa.spr[48] + ' & ' + '%i' %sa.spr[51] + ' & ' + '%i' %sa.spr[54] + ' & ' + '%i' %sa.spr[57] + ' & ' + \
                  '%.i' %sa.spr[60] + ' & ' + '%.i' %sa.spr[63] + ' & ' + '%.i' %sa.spr[66] + ' & ' + '%.i' %sa.spr[69]
          }

tab4_sa = {'hs' : ' & ' + '%.2f' %sa.hs[72] + ' & ' + '%.2f' %sa.hs[75] + ' & ' + '%.2f' %sa.hs[78] + ' & ' + '%.2f' %sa.hs[81] + ' & ' + \
                  '%.2f' %sa.hs[84] + ' & ' + '%.2f' %sa.hs[87] + ' & ' + '%.2f' %sa.hs[90] + ' & ' + '%.2f' %sa.hs[93],
           'tp' : ' & ' + '%.1f' %sa.tp[72] + ' & ' + '%.1f' %sa.tp[75] + ' & ' + '%.1f' %sa.tp[78] + ' & ' + '%.1f' %sa.tp[81] + ' & ' + \
                  '%.1f' %sa.tp[84] + ' & ' + '%.1f' %sa.tp[87] + ' & ' + '%.1f' %sa.tp[90] + ' & ' + '%.1f' %sa.tp[93],
           'dp' : ' & ' + '%i' %sa.dp[72] + ' & ' + '%i' %sa.dp[75] + ' & ' + '%i' %sa.dp[78] + ' & ' + '%i' %sa.dp[81] + ' & ' + \
                  '%i' %sa.dp[84] + ' & ' + '%i' %sa.dp[87] + ' & ' + '%.i' %sa.dp[90] + ' & ' + '%.i' %sa.dp[93],
           'spr' : ' & ' + '%.i' %sa.spr[72] + ' & ' + '%i' %sa.spr[75] + ' & ' + '%i' %sa.spr[78] + ' & ' + '%i' %sa.spr[81] + ' & ' + \
                  '%.i' %sa.spr[84] + ' & ' + '%.i' %sa.spr[87] + ' & ' + '%.i' %sa.spr[90] + ' & ' + '%.i' %sa.spr[93]
          }

tab5_sa = {'hs' : ' & ' + '%.2f' %sa.hs[96] + ' & ' + '%.2f' %sa.hs[99] + ' & ' + '%.2f' %sa.hs[102] + ' & ' + '%.2f' %sa.hs[105] + ' & ' + \
                  '%.2f' %sa.hs[108] + ' & ' + '%.2f' %sa.hs[111] + ' & ' + '%.2f' %sa.hs[114] + ' & ' + '%.2f' %sa.hs[117],
           'tp' : ' & ' + '%.1f' %sa.tp[96] + ' & ' + '%.1f' %sa.tp[99] + ' & ' + '%.1f' %sa.tp[102] + ' & ' + '%.1f' %sa.tp[104] + ' & ' + \
                  '%.1f' %sa.tp[108] + ' & ' + '%.1f' %sa.tp[111] + ' & ' + '%.1f' %sa.tp[114] + ' & ' + '%.1f' %sa.tp[117],
           'dp' : ' & ' + '%i' %sa.dp[96] + ' & ' + '%i' %sa.dp[99] + ' & ' + '%i' %sa.dp[102] + ' & ' + '%i' %sa.dp[105] + ' & ' + \
                  '%i' %sa.dp[108] + ' & ' + '%i' %sa.dp[111] + ' & ' + '%.i' %sa.dp[113] + ' & ' + '%.i' %sa.dp[117],
           'spr' : ' & ' + '%.i' %sa.spr[96] + ' & ' + '%i' %sa.spr[99] + ' & ' + '%i' %sa.spr[102] + ' & ' + '%i' %sa.spr[105] + ' & ' + \
                  '%.i' %sa.spr[108] + ' & ' + '%.i' %sa.spr[111] + ' & ' + '%.i' %sa.spr[114] + ' & ' + '%.i' %sa.spr[117]
          }

tab6_sa = {'hs' : ' & ' + '%.2f' %sa.hs[120] + ' & ' + '%.2f' %sa.hs[123] + ' & ' + '%.2f' %sa.hs[126] + ' & ' + '%.2f' %sa.hs[129] + ' & ' + \
                  '%.2f' %sa.hs[132] + ' & ' + '%.2f' %sa.hs[135] + ' & ' + '%.2f' %sa.hs[138] + ' & ' + '%.2f' %sa.hs[141],
           'tp' : ' & ' + '%.1f' %sa.tp[120] + ' & ' + '%.1f' %sa.tp[123] + ' & ' + '%.1f' %sa.tp[126] + ' & ' + '%.1f' %sa.tp[129] + ' & ' + \
                  '%.1f' %sa.tp[132] + ' & ' + '%.1f' %sa.tp[135] + ' & ' + '%.1f' %sa.tp[138] + ' & ' + '%.1f' %sa.tp[141],
           'dp' : ' & ' + '%i' %sa.dp[120] + ' & ' + '%i' %sa.dp[123] + ' & ' + '%i' %sa.dp[126] + ' & ' + '%i' %sa.dp[129] + ' & ' + \
                  '%i' %sa.dp[132] + ' & ' + '%i' %sa.dp[135] + ' & ' + '%.i' %sa.dp[138] + ' & ' + '%.i' %sa.dp[141],
           'spr' : ' & ' + '%.i' %sa.spr[120] + ' & ' + '%i' %sa.spr[123] + ' & ' + '%i' %sa.spr[126] + ' & ' + '%i' %sa.spr[128] + ' & ' + \
                  '%.i' %sa.spr[132] + ' & ' + '%.i' %sa.spr[135] + ' & ' + '%.i' %sa.spr[138] + ' & ' + '%.i' %sa.spr[141]
          }

tab7_sa = {'hs' : ' & ' + '%.2f' %sa.hs[144] + ' & ' + '%.2f' %sa.hs[147] + ' & ' + '%.2f' %sa.hs[150] + ' & ' + '%.2f' %sa.hs[153] + ' & ' + \
                  '%.2f' %sa.hs[156] + ' & ' + '%.2f' %sa.hs[159] + ' & ' + '%.2f' %sa.hs[162] + ' & ' + '%.2f' %sa.hs[165],
           'tp' : ' & ' + '%.1f' %sa.tp[144] + ' & ' + '%.1f' %sa.tp[147] + ' & ' + '%.1f' %sa.tp[150] + ' & ' + '%.1f' %sa.tp[153] + ' & ' + \
                  '%.1f' %sa.tp[156] + ' & ' + '%.1f' %sa.tp[159] + ' & ' + '%.1f' %sa.tp[162] + ' & ' + '%.1f' %sa.tp[165],
           'dp' : ' & ' + '%i' %sa.dp[144] + ' & ' + '%i' %sa.dp[147] + ' & ' + '%i' %sa.dp[150] + ' & ' + '%i' %sa.dp[153] + ' & ' + \
                  '%i' %sa.dp[156] + ' & ' + '%i' %sa.dp[159] + ' & ' + '%.i' %sa.dp[162] + ' & ' + '%.i' %sa.dp[165],
           'spr' : ' & ' + '%.i' %sa.spr[144] + ' & ' + '%i' %sa.spr[147] + ' & ' + '%i' %sa.spr[150] + ' & ' + '%i' %sa.spr[153] + ' & ' + \
                  '%.i' %sa.spr[156] + ' & ' + '%.i' %sa.spr[159] + ' & ' + '%.i' %sa.spr[162] + ' & ' + '%.i' %sa.spr[165]
          }




###############################################################################
#This tells Jinja how to handle LaTeX syntax

latex_jinja_env = jinja2.Environment(
    block_start_string = '\BLOCK{',
    block_end_string = '}',
    variable_start_string = '\VAR{',
    variable_end_string = '}',
    comment_start_string = '\#{',
    comment_end_string = '}',
    line_statement_prefix = '%-',
    line_comment_prefix = '%#',
    trim_blocks = True,
    autoescape = False,
    #loader = jinja2.FileSystemLoader(os.path.abspath('.'))
    loader = jinja2.FileSystemLoader(pathname_template)
)

# Modify to specify the template (colocar o template que vai ser mudado -  modelo com os \VAR)
template = latex_jinja_env.get_template('boletim_PNBOIA.tex')

#cria um arquivo para ser sobrescrito
#nome do arquivo a ser salvo (com a data de hoje)
filename = 'boletim_PNBOIA_' + dt.datetime.strftime(dt.datetime.now(),'%Y%m%d') + '.tex'

#outfile = open(pathname + filename,'w')
outfile = codecs.open(pathname_template + filename,'w','utf-8')

#cria arquivo com os valores substituidos (colocar todos em um so comando)
outfile.write(template.render(tab1_rg_hs=tab1_rg['hs'],
                              tab1_rg_tp=tab1_rg['tp'],
                              tab1_rg_dp=tab1_rg['dp'],
                              tab1_rg_spr=tab1_rg['spr'],
                              tab2_rg_hs=tab2_rg['hs'],
                              tab2_rg_tp=tab2_rg['tp'],
                              tab2_rg_dp=tab2_rg['dp'],
                              tab2_rg_spr=tab2_rg['spr'],
                              tab3_rg_hs=tab3_rg['hs'],
                              tab3_rg_tp=tab3_rg['tp'],
                              tab3_rg_dp=tab3_rg['dp'],
                              tab3_rg_spr=tab3_rg['spr'],
                              tab4_rg_hs=tab4_rg['hs'],
                              tab4_rg_tp=tab4_rg['tp'],
                              tab4_rg_dp=tab4_rg['dp'],
                              tab4_rg_spr=tab4_rg['spr'],
                              tab5_rg_hs=tab5_rg['hs'],
                              tab5_rg_tp=tab5_rg['tp'],
                              tab5_rg_dp=tab5_rg['dp'],
                              tab5_rg_spr=tab5_rg['spr'],
                              tab6_rg_hs=tab6_rg['hs'],
                              tab6_rg_tp=tab6_rg['tp'],
                              tab6_rg_dp=tab6_rg['dp'],
                              tab6_rg_spr=tab6_rg['spr'],
                              tab7_rg_hs=tab7_rg['hs'],
                              tab7_rg_tp=tab7_rg['tp'],
                              tab7_rg_dp=tab7_rg['dp'],
                              tab7_rg_spr=tab7_rg['spr'],
                              tab1_fl_hs=tab1_fl['hs'],
                              tab1_fl_tp=tab1_fl['tp'],
                              tab1_fl_dp=tab1_fl['dp'],
                              tab1_fl_spr=tab1_fl['spr'],
                              tab2_fl_hs=tab2_fl['hs'],
                              tab2_fl_tp=tab2_fl['tp'],
                              tab2_fl_dp=tab2_fl['dp'],
                              tab2_fl_spr=tab2_fl['spr'],
                              tab3_fl_hs=tab3_fl['hs'],
                              tab3_fl_tp=tab3_fl['tp'],
                              tab3_fl_dp=tab3_fl['dp'],
                              tab3_fl_spr=tab3_fl['spr'],
                              tab4_fl_hs=tab4_fl['hs'],
                              tab4_fl_tp=tab4_fl['tp'],
                              tab4_fl_dp=tab4_fl['dp'],
                              tab4_fl_spr=tab4_fl['spr'],
                              tab5_fl_hs=tab5_fl['hs'],
                              tab5_fl_tp=tab5_fl['tp'],
                              tab5_fl_dp=tab5_fl['dp'],
                              tab5_fl_spr=tab5_fl['spr'],
                              tab6_fl_hs=tab6_fl['hs'],
                              tab6_fl_tp=tab6_fl['tp'],
                              tab6_fl_dp=tab6_fl['dp'],
                              tab6_fl_spr=tab6_fl['spr'],
                              tab7_fl_hs=tab7_fl['hs'],
                              tab7_fl_tp=tab7_fl['tp'],
                              tab7_fl_dp=tab7_fl['dp'],
                              tab7_fl_spr=tab7_fl['spr'], 
                              tab1_sa_hs=tab1_sa['hs'],
                              tab1_sa_tp=tab1_sa['tp'],
                              tab1_sa_dp=tab1_sa['dp'],
                              tab1_sa_spr=tab1_sa['spr'],
                              tab2_sa_hs=tab2_sa['hs'],
                              tab2_sa_tp=tab2_sa['tp'],
                              tab2_sa_dp=tab2_sa['dp'],
                              tab2_sa_spr=tab2_sa['spr'],
                              tab3_sa_hs=tab3_sa['hs'],
                              tab3_sa_tp=tab3_sa['tp'],
                              tab3_sa_dp=tab3_sa['dp'],
                              tab3_sa_spr=tab3_sa['spr'],
                              tab4_sa_hs=tab4_sa['hs'],
                              tab4_sa_tp=tab4_sa['tp'],
                              tab4_sa_dp=tab4_sa['dp'],
                              tab4_sa_spr=tab4_sa['spr'],
                              tab5_sa_hs=tab5_sa['hs'],
                              tab5_sa_tp=tab5_sa['tp'],
                              tab5_sa_dp=tab5_sa['dp'],
                              tab5_sa_spr=tab5_sa['spr'],
                              tab6_sa_hs=tab6_sa['hs'],
                              tab6_sa_tp=tab6_sa['tp'],
                              tab6_sa_dp=tab6_sa['dp'],
                              tab6_sa_spr=tab6_sa['spr'],
                              tab7_sa_hs=tab7_sa['hs'],
                              tab7_sa_tp=tab7_sa['tp'],
                              tab7_sa_dp=tab7_sa['dp'],
                              tab7_sa_spr=tab7_sa['spr'] 
                            ))

#salva arquivo .tex modificado
outfile.close()

#gera o pdf
os.system("pdflatex " + pathname_template + filename)

#copia para o boletim em /Previsao
#os.system('cp ' + pathname_template + filename[:-4] + '.pdf' + ' ' + os.environ['HOME'] + '/Dropbox/Previsao/boletim/')


arq = "'Boletim PNBOIA - %s'" %dt.datetime.strftime(dt.datetime.now(),'%d/%m/%Y')

#manda por email
print 'Enviando por e-mail...'

#os.system("/usr/bin/mutt -s " + arq + " -a " + filename[:-4] + '.pdf' + " -- pereira.henriquep@gmail.com izabelcm.nogueira@gmail.com isabelacabral@oceanica.ufrj.br \
#                                                                             talitha.sl@gmail.com tamiris.alfama@poli.ufrj.br ufrjlioc@gmail.com < /dev/null")

print 'E-mail enviado.'