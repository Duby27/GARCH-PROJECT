#%%
import os
import pandas as pd
import numpy as np

os.chdir('D:/Quant Projects/GARCH Project')
df = pd.read_csv('results/stage1/reliability_results.csv')
df.head()

# %%
sse = df[df['index'] == 'sse'].sort_values('NLL').reset_index(drop=True)
eustoxx = df[df['index'] == 'eustoxx'].sort_values('NLL').reset_index(drop=True)
ftse = df[df['index'] == 'ftse'].sort_values('NLL').reset_index(drop=True)
nikkei = df[df['index'] == 'nikkei'].sort_values('NLL').reset_index(drop=True)
sp500 = df[df['index'] == 'sp500'].sort_values('NLL').reset_index(drop=True)
sp500
# %%
sse_nll = float(sse['NLL'].iloc[0])
sse_grad_mod = np.sqrt(sse['grad_alpha'].iloc[0]**2 + sse['grad_beta'].iloc[0]**2 +sse['grad_omega'].iloc[0]**2)
sse_n = len(sse[sse['NLL'] <= sse_nll +1e-4])
print(sse_nll)
print(sse_grad_mod)
print(sse_n)
# %%
eustoxx_nll = float(eustoxx['NLL'].iloc[0])
eustoxx_grad_mod = np.sqrt(
    eustoxx['grad_alpha'].iloc[0]**2
    + eustoxx['grad_beta'].iloc[0]**2
    + eustoxx['grad_omega'].iloc[0]**2
)
eustoxx_n = len(eustoxx[eustoxx['NLL'] <= eustoxx_nll + 1e-4])

print(eustoxx_nll)
print(eustoxx_grad_mod)
print(eustoxx_n)

# %%
ftse_nll = float(ftse['NLL'].iloc[0])
ftse_grad_mod = np.sqrt(
    ftse['grad_alpha'].iloc[0]**2
    + ftse['grad_beta'].iloc[0]**2
    + ftse['grad_omega'].iloc[0]**2
)
ftse_n = len(ftse[ftse['NLL'] <= ftse_nll + 1e-4])

print(ftse_nll)
print(ftse_grad_mod)
print(ftse_n)

# %%
nikkei_nll = float(nikkei['NLL'].iloc[0])
nikkei_grad_mod = np.sqrt(
    nikkei['grad_alpha'].iloc[0]**2
    + nikkei['grad_beta'].iloc[0]**2
    + nikkei['grad_omega'].iloc[0]**2
)
nikkei_n = len(nikkei[nikkei['NLL'] <= nikkei_nll + 1e-4])

print(nikkei_nll)
print(nikkei_grad_mod)
print(nikkei_n)

# %%
sp500_nll = float(sp500['NLL'].iloc[0])
sp500_grad_mod = np.sqrt(
    sp500['grad_alpha'].iloc[0]**2
    + sp500['grad_beta'].iloc[0]**2
    + sp500['grad_omega'].iloc[0]**2
)
sp500_n = len(sp500[sp500['NLL'] <= sp500_nll + 1e-4])

print(sp500_nll)
print(sp500_grad_mod)
print(sp500_n)
# %%
# it looks like a good fit for me. so:
final_params = pd.concat([
    sse.head(1),
    eustoxx.head(1),
    ftse.head(1),
    nikkei.head(1),
    sp500.head(1)
], ignore_index=True)

final_params = final_params[['index', 'alpha', 'beta', 'omega', 'alpha+beta', 'NLL']]

final_params.to_csv('results/stage1/final_params.csv', index=False)
# %%
