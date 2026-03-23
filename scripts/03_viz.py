import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Configurações de alta fidelidade
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# Definição da Paleta de Cores Padronizada
cor_sem = '#66b2a2' # Verde Água
cor_com = '#fc8d62' # Salmão/Laranja

print(">>> Iniciando Visualização Final Estratégica...")
try:# 1. CARGA DOS DADOS
    df = pd.read_excel('Base_DPU.xlsx')
    df.columns = [c.lower().strip() for c in df.columns]

    # Convertendo 'sim'/'nao' para Booleano para o Python entender a paleta
    mapeamento = {'sim': True, 'nao': False, 'SIM': True, 'NAO': False}
    df['tem_dpu_bool'] = df['tem_dpu'].map(mapeamento)
except Exception as e:
    print(f"Erro ao gerar graficos: {e}")

# --- 1. BOXPLOT: VULNERABILIDADE ---
# Criamos o boxplot com foco na comparação de médias
ax1 = sns.boxplot(
        data=df, 
        x='tem_dpu_bool', 
        y='ivcad', 
        palette={False: cor_sem, True: cor_com},
        hue='tem_dpu_bool',
        legend=False,
        order=[False, True],
        showmeans=True, # Adiciona o marcador de média
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
        fliersize=2, # Diminui o tamanho dos pontos de outlier para não poluir
        width=0.6
    )
# Melhorando a legibilidade dos eixos
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Cidades sem Unidade\n(Gap Assistencial)', 'Cidades com Unidade\n(Cobertura Atual)'], fontsize=11)

# Títulos Estratégicos
plt.title('DISTRIBUIÇÃO DA VULNERABILIDADE SOCIAL (IVCAD)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Índice de Vulnerabilidade (IVCAD)', fontsize=12)
plt.xlabel('', labelpad=10) # Remove label óbvio para limpar o visual

# Adicionando uma nota de rodapé automática
plt.figtext(0.99, 0.01, f'Fonte: DPU/PNUD (N={len(df)})', horizontalalignment='right', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig('01_Boxplot_IVCAD.png', bbox_inches='tight')
print(">>> Gráfico 01: Boxplot de Vulnerabilidade gerado.")

# --- 2. RANKING TOP 5 ---
plt.figure(figsize=(12, 8))
# 2.1 Filtragem e Formatação de Rótulos
df_prioridade = df[df['tem_dpu_bool'] == False].sort_values('ranking_prioridade').head(5).copy()
df_prioridade['label_mun'] = df_prioridade['municipio'].str.title() + " (" + df_prioridade['estado'] + ")"

# 2.2 Plotagem com Paleta Profissional
ax2 = sns.barplot(
    data=df_prioridade,
    x='score_prioridade',
    y='label_mun',
    palette='mako',
    hue='label_mun', # Define matiz para evitar aviso do Seaborn
    legend=False
)
# 2.3 Adição de Rótulos de Dados (Data Labels) nas Barras
for p in ax2.patches:
    width = p.get_width()
    ax2.annotate(f'{width:.1f}',
                    (width, p.get_y() + p.get_height() / 2),
                    ha = 'left', va = 'center',
                    xytext = (5, 0),
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')

# 2.4 Ajustes Estéticos Finais
plt.title('TOP 5 MUNICÍPIOS-ALVO PARA EXPANSÃO DA DPU', fontsize=15, fontweight='bold', pad=25)
plt.xlabel('Score de Prioridade Estratégica (0-100)', fontsize=11, labelpad=12)
plt.ylabel('', labelpad=5) # Limpa o eixo Y pois o rótulo já é autoexplicativo

# Adicionando uma linha vertical com a média nacional para comparação
media_nacional = df['score_prioridade'].mean()
plt.axvline(media_nacional, color='red', linestyle='--', alpha=0.6, label=f'Média Nac.: {media_nacional:.1f}')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('02_ranking_prioridade_DPU.png', bbox_inches='tight')
print(">>> Gráfico 02: Ranking Top 5 gerado.")

# --- 3. MATRIZ DE VÁCUO ESTRATÉGICO: IDHM vs IVCAD ----
# 1. Saneamento Rápido antes da Plotagem
def limpar_escala_idhm(x):
        if pd.isna(x): return x
        s = str(x).replace('.', '').replace(',', '')
        s = s.ljust(4, '0') if len(s) < 4 else s # Garante 4 dígitos (Ex: 805 -> 8050)
        val = float(s)
        if val > 1000: return val / 10000 # Ajusta 8050 para 0.8050
        if val > 1: return val / 1000 # Ajusta 805 para 0.805
        return val

df['idhm_plot'] = df['idhm'].apply(limpar_escala_idhm)

plt.figure(figsize=(12, 8))

# 2. Plotagem com os dados saneados
ax3 = sns.scatterplot(
        data=df, 
        x='idhm_plot', # Usando a coluna corrigida
        y='ivcad', 
        hue='tem_dpu_bool', 
        palette={False: cor_sem, True: cor_com},
        alpha=0.6,
        s=80,
        edgecolor='w',
        linewidth=0.5
    )

# 3. Linhas de Referência baseadas na nova escala (0 a 1)
media_ivcad = df['ivcad'].mean()
media_idhm = df['idhm_plot'].mean()

plt.axhline(media_ivcad, color='darkred', linestyle=':', alpha=0.5)
plt.axvline(media_idhm, color='darkblue', linestyle=':', alpha=0.5)

# 4. Títulos e Labels
plt.title('MATRIZ DE VÁCUO SOCIOECONÔMICO E ASSISTENCIAL', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Desenvolvimento Humano (IDHM Corrigido)', fontsize=12)
plt.ylabel('Índice de Vulnerabilidade (IVCAD)', fontsize=12)
plt.xlim(0.3, 0.9) # Limita o eixo X para a realidade do Brasil

# 5. Anotações de Quadrantes
plt.text(0.32, df['ivcad'].max(), 'QUADRANTE CRÍTICO\n(Alta Vulnerabilidade / Baixo IDH)', 
            fontsize=10, fontweight='bold', color='darkred', va='top')

# 6. Destaque dos Top 3 Alvos
top_alvos = df[df['tem_dpu_bool'] == False].nlargest(3, 'score_prioridade')
for i, row in top_alvos.iterrows():
        plt.annotate(row['municipio'], (row['idhm_plot'], row['ivcad']), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig('03_Matriz_Vácuo_DPU.png', bbox_inches='tight')
print(">>> Gráfico 03 gerado.")