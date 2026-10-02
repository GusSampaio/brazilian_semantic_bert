import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Configurações Iniciais
# ==========================================
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "srl-portuguese"

# Configuração visual para a apresentação (estilo mais acadêmico/clean)
sns.set_theme(style="whitegrid", context="talk")
PALETTE = "Blues_d"

# ==========================================
# 2. Extração dos Dados do MLflow
# ==========================================
print("Buscando dados do MLflow...")
df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

# Verifica se a coluna da tag existe. O MLflow geralmente salva as tags como 'tags.<nome_da_tag>'
TAG_COLUMN = "tags.Experimento"
if TAG_COLUMN not in df.columns:
    raise ValueError(f"A tag '{TAG_COLUMN}' não foi encontrada nos dados do MLflow. Verifique o nome exato da tag.")

# Limpa execuções que falharam e foca apenas nas métricas principais
# Assumindo que o metric_key_prefix="test" gerou a métrica "test_f1"
df_valid = df[df["status"] == "FINISHED"].copy()

# ==========================================
# 3. Geração dos Gráficos
# ==========================================

# --- Experimento 1: Comparação de Modelos com Desvio Padrão ---
# Filtrando pelas tags (Ajuste "1", "2", "3" para o valor exato que você usou no seu código)
print(df_valid.head())
df_exp1 = df_valid[df_valid[TAG_COLUMN] == 1]

if not df_exp1.empty:
    plt.figure(figsize=(10, 6))
    # O parâmetro errorbar="sd" faz o seaborn calcular o desvio padrão automaticamente baseado nas seeds
    ax = sns.barplot(
        data=df_exp1, 
        x="params.model_name", 
        y="metrics.test_f1", 
        errorbar="sd", 
        capsize=0.1, 
        palette=PALETTE
    )
    plt.title("Experimento 1: Desempenho F1 por Modelo", pad=20)
    plt.xlabel("Modelo")
    plt.ylabel("F1-Score (Teste)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("grafico_experimento_1.png", dpi=300)
    print("Gráfico do Experimento 1 salvo como 'grafico_experimento_1.png'")

# --- Experimento 2: Comparação de Estratégias ---
df_exp2 = df_valid[df_valid[TAG_COLUMN] == 2]

if not df_exp2.empty:
    plt.figure(figsize=(12, 6))
    # Aqui usamos 'hue' para separar os resultados pela estratégia de loss que você implementou
    ax = sns.barplot(
        data=df_exp2, 
        x="params.model_name", 
        y="metrics.test_f1", 
        hue="params.strategy", 
        errorbar="sd", 
        capsize=0.05
    )
    plt.title("Experimento 2: Impacto da Estratégia de Loss no F1-Score", pad=20)
    plt.xlabel("Modelo")
    plt.ylabel("F1-Score (Teste)")
    plt.legend(title="Estratégia")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("grafico_experimento_2.png", dpi=300)
    print("Gráfico do Experimento 2 salvo como 'grafico_experimento_2.png'")

# --- Experimento 3: Modelo Especialista ---
df_exp3 = df_valid[df_valid[TAG_COLUMN] == 3]

if not df_exp3.empty:
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=df_exp3, 
        x="params.model_name", 
        y="metrics.test_f1", 
        palette="crest"
    )
    plt.title("Experimento 3: Desempenho do Modelo Especialista", pad=20)
    plt.xlabel("Modelo")
    plt.ylabel("F1-Score (Teste)")
    
    # Adicionando o valor exato no topo da barra para destaque
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig("grafico_experimento_3.png", dpi=300)
    print("Gráfico do Experimento 3 salvo como 'grafico_experimento_3.png'")