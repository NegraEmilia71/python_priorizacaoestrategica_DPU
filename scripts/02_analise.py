import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency

# --- 1. CARGA DOS DADOS (Onde o 'df' nasce) ---
# Certifique-se de que o caminho do arquivo está correto
df = pd.read_csv('base_consolidada.csv', sep=';', encoding='utf-8-sig')

# Padronização de nomes de colunas
df.columns = [c.lower().replace('.', '').replace(' ', '_').strip() for c in df.columns]

print(f">>> Registros carregados: {len(df)}") # Aqui deve aparecer 5570

# --- 2. TRATAMENTO PARA MANTER AS 5570 LINHAS (Imputação) ---
# Preenchemos nulos para que o ranking e a exportação não percam dados
cols_num = ['idhm', 'expectativa_vida', 'pib', 'ivcad', 'familias', 'pessoas']
for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        # Preenche com a média do estado, depois nacional, depois zero
        df[col] = df.groupby('estado')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean()).fillna(0)

for col in ['pib', 'pessoas', 'familias']:
    df.loc[df[col] <= 0, col] = 1

# --- 3. CRIAÇÃO DAS COLUNAS MÉTRICAS ---
df['log_pessoas'] = np.log1p(df['pessoas'])
df['log_pib'] = np.log1p(df['pib'])
df['z_ivcad'] = (df['ivcad'] - df['ivcad'].mean()) / (df['ivcad'].std() if df['ivcad'].std() != 0 else 1)

scaler = MinMaxScaler()
df[['ivcad_norm', 'fam_norm']] = scaler.fit_transform(df[['ivcad', 'familias']])
df['score_prioridade'] = (df['ivcad_norm'] * 0.7) + (df['fam_norm'] * 0.3)

# Identificação da coluna DPU 
col_dpu = 'dpu_presencial' if 'dpu_presencial' in df.columns else 'tem_dpu'

# Criação de métricas de eficiência socioeconômica
df['pib_per_capita'] = df['pib'] / df['pessoas']
# Inversão para que cidades pobres ganhem mais pontos no score
df['pib_inverso'] = 1 / (df['pib_per_capita'] + 0.001) 

scaler = MinMaxScaler(feature_range=(0, 100))
# Peso: 70% Vulnerabilidade Social | 30% Carência Econômica
df['score_bruto'] = (df['ivcad'] * 0.7) + (df['pib_inverso'] * 0.3)
df['score_prioridade'] = scaler.fit_transform(df[['score_bruto']])

# --- 4. MODELAGEM ---
print(">>> Executando modelagens estatísticas...")
cols_reg = ['z_ivcad', 'log_pessoas', 'idhm', 'expectativa_vida', 'log_pib', 'ivcad']
df_reg = df.dropna(subset=cols_reg).copy()

# Remove valores infinitos que podem ter surgido no log
df_reg = df_reg.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_reg)

# 4.2 Qui-Quadrado (Distribuição Regional da DPU)
try:
    if df[col_dpu].nunique() > 1:
        tabela_con = pd.crosstab(df['regiao'], df[col_dpu])
        chi2, p_valor, _, _ = chi2_contingency(tabela_con)
        sig_text = "SIGNIFICATIVA" if p_valor < 0.05 else "NÃO SIGNIFICATIVA"
    else:
        p_valor, sig_text = 1.0, "SEM VARIAÇÃO NA DPU"
except Exception as e:
    p_valor, sig_text = 1.0, f"ERRO: {e}"

# 4.3 Regressão Logística para explica a presença x ausencia da DPU
try:
    X_log = sm.add_constant(df_reg[['z_ivcad', 'log_pessoas']])
    y_log = df_reg[col_dpu].astype(str).str.upper().str.strip()
    y_log = y_log.map({'SIM': 1, 'TRUE': 1, '1': 1, '1.0': 1}).fillna(0).astype(int)
    
    modelo_logit = sm.Logit(y_log, X_log).fit(disp=0)
    resumo_logit = modelo_logit.summary().as_text()
except Exception as e:
    resumo_logit = f"Erro na Logística: {e}"

# 4.4 OLS para explica a Vulnerabilidade Social - IVCAD
try:
    X_ols = sm.add_constant(df_reg[['idhm', 'expectativa_vida', 'log_pib']])
    y_ols = df_reg['ivcad']
    modelo_ols = sm.OLS(y_ols, X_ols).fit()
    resumo_ols = modelo_ols.summary().as_text()
except Exception as e:
    resumo_ols = f"Erro no OLS: {e}"

print(f">>> Qui-Quadrado: {sig_text} (p={p_valor:.4f})")

# --- 5. RANKING DE EXPANSÃO INSTITUCIONAL ---

# 1. Definição da Máscara (Quem NÃO tem DPU)
# Usamos .astype(str) para garantir que funcione com Booleano, Int ou String
mask_expansao = df[col_dpu].astype(str).str.upper().str.strip().isin(['NAO', 'FALSE', '0', '0.0'])

# 2. CRIAÇÃO DA COLUNA DE DESEMPATE (O que estava faltando!)
# Somamos um valor minúsculo baseado na população para que cidades maiores 
# ganhem a posição em caso de empate no score_prioridade.
df['score_desempate'] = df['score_prioridade'] + (df['pessoas'] / 1e12)

# 3. Cálculo do Ranking Nacional
df['ranking_prioridade'] = 0
df.loc[mask_expansao, 'ranking_prioridade'] = (
    df.loc[mask_expansao, 'score_desempate']
    .rank(ascending=False, method='min')
    .astype(int)
)

# 4. Cálculo do Ranking Estadual (Onde ocorria o KeyError)
df['ranking_estadual'] = 0
df.loc[mask_expansao, 'ranking_estadual'] = (
    df[mask_expansao]
    .groupby('estado')['score_desempate']
    .rank(ascending=False, method='min')
    .astype(int)
)

# 5. Top 5 Prioridades para o Console
# Filtramos apenas onde o ranking é maior que 0 (ou seja, onde a máscara foi aplicada)
top_5 = df[df['ranking_prioridade'] > 0].nsmallest(5, 'ranking_prioridade')

print("\n" + "="*50)
print("TOP 5 MUNICÍPIOS PARA EXPANSÃO DPU (NACIONAL)")
print("="*50)
if not top_5.empty:
    print(top_5[['ranking_prioridade', 'municipio', 'estado', 'score_prioridade', 'ivcad']])
else:
    print("Aviso: Nenhum município encontrado para expansão. Verifique a coluna DPU.")

# --- CORREÇÃO DE ESCALA: IDHM e EXPECTATIVA DE VIDA ---
# 1. Corrigindo o IDHM (Alvo: 0.xxx)
def ajustar_idhm(valor):
    if pd.isna(valor): return valor
    v = float(str(valor).replace(',', '.'))
    if v > 10:    return v / 1000  # Ex: 750 -> 0.750
    if v > 1:     return v / 10    # Ex: 7.5 -> 0.750
    return v                       # Já está em 0.xxx

# 2. Corrigindo EXPECTATIVA DE VIDA (Alvo: xx.x)
def ajustar_expectativa(valor):
    if pd.isna(valor): return valor
    v = float(str(valor).replace(',', '.'))
    if v > 1000:  return v / 100   # Ex: 7250 -> 72.50
    if v > 100:   return v / 10    # Ex: 725 -> 72.5
    return v                       # Já está em xx.x

# Aplicando as funções
df['idhm'] = df['idhm'].apply(ajustar_idhm)
df['expectativa_vida'] = df['expectativa_vida'].apply(ajustar_expectativa)

print(f">>> Escalas saneadas! IDHM médio: {df['idhm'].mean():.3f}")

# --- 6. EXPORTAÇÃO DO RELATÓRIO TÉCNICO ---
# Definimos o nome do arquivo
nome_relatorio = 'RELATORIO_TECNICO_DPU_FINAL.csv'

with open(nome_relatorio, 'w', encoding='utf-8') as f:
    f.write("RELATÓRIO TÉCNICO DE INTELIGÊNCIA GEOGRÁFICA - DPU/PNUD\n")
    f.write("Foco: Identificação de Gaps Assistenciais e Priorização de Expansão\n")
    f.write("="*85 + "\n\n")
    f.write(f"Data de Geração: 2026-03-17 | Municípios Processados: {len(df)}\n")
    f.write("="*85 + "\n\n")
    
    # 1. Diagnóstico Regional (Qui-Quadrado)
    f.write("1. DIAGNÓSTICO DE DISTRIBUIÇÃO REGIONAL\n")
    f.write(f"   - P-valor (Qui-Quadrado): {p_valor:.4f}\n")
    f.write(f"   - Análise: A disparidade de atendimento entre regiões é {sig_text}.\n\n")

    # 2. Modelagem Logística
    f.write("2. MODELAGEM PREDITIVA (PREVALÊNCIA DA DPU)\n")
    f.write("   - Técnica: Regressão Logística com Padronização Z-Score.\n")
    f.write(f"{resumo_logit}\n\n")

    # 3. Modelagem OLS
    f.write("3. DETERMINANTES DE VULNERABILIDADE (OLS)\n")
    f.write("   - Variável Dependente: IVCAD\n")
    f.write(f"{resumo_ols}\n\n")

    # 4. Métricas de Impacto
    f.write("4. MÉTRICAS DE IMPACTO E VOLUMETRIA MÉDIA\n")
    if mask_expansao.any():# Calculando médias diretamente no relatório para precisão
        media_sem = df[mask_expansao]['familias'].mean()
        f.write(f"   - Média de Famílias em Municípios SEM DPU: {media_sem:,.0f}\n")
    
    if (~mask_expansao).any():
        media_com = df[~mask_expansao]['familias'].mean()
        f.write(f"   - Média de Famílias em Municípios COM DPU: {media_com:,.0f}\n\n")

    # 5. Top 5 Prioridades
    f.write("5. CLASSIFICAÇÃO TOP 5: PRIORIDADE PARA EXPANSÃO INSTITUCIONAL\n")
    f.write("-" * 85 + "\n")
    f.write(f"{'Ranking':<8} | {'Município':<25} | {'UF':<10} | {'IVCAD':<12} | {'Score':<10}\n")
    f.write("-" * 85 + "\n")
    
    # Filtramos o Top 5 
    df_top5 = df[mask_expansao].sort_values('ranking_prioridade').head(5)
    for _, row in df_top5.iterrows():
        f.write(f"{int(row['ranking_prioridade']):<8} | {str(row['municipio'])[:25]:<25} | {row['estado']:<10} | "
                f"{row['ivcad']:<12.3f} | {row['score_prioridade']:<10.4f}\n")
    
    f.write("-" * 85 + "\n")
    f.write("\n>>> FIM DO RELATÓRIO TÉCNICO <<<")
     
print(f">>> Relatório técnico gerado: {nome_relatorio}")

# --- 8. EXPORTAÇÃO FINAL ---
# Salvamos o arquivo completo que passou por todo o processo
df.to_excel('Base_DPU.xlsx', index=False)
print(f">>> SUCESSO! Arquivo exportado com {len(df)} linhas.")
