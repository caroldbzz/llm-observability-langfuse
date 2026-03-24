# 🛠️ Configuração do Ambiente — Demo Langfuse 

Este guia mostra como configurar o ambiente para rodar a aplicação do curso, incluindo:

- Criação de ambiente virtual (Windows, Linux, Mac)
- Instalação de dependências
- Configuração de variáveis de ambiente (`.env`)

---

## 1. Criar e ativar ambiente virtual

**Windows**
```bash
python3.12 -m venv .venv
.\venv\Scripts\activate
```

**Mac/Linux**
```bash
python -m venv venv
source venv/bin/activate
```

## 2. Instalar dependências
Utilize o comando abaixo para instalar as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```
langfuse==4.0.1
openai==2.29.0
python-dotenv==1.2.2
pandas==3.0.1

## 3. Configuração das variáveis de ambiente
Crie ou edite o arquivo `.env`e adicione sua chave da OpenAI:
```bash
OPENAI_API_KEY=<sua_chave>
```
Adicione também as chaves do Langfuse:
```bash
LANGFUSE_SECRET_KEY=<secret_key>
LANGFUSE_PUBLIC_KEY=<public_key>
LANGFUSE_BASE_URL=<langfuse_url>
```

# Links úteis
- https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset
- https://us.cloud.langfuse.com