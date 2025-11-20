import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# --- Configuration de la page ---
st.set_page_config(page_title="App de Prédiction Boursière", layout="wide")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-d")

st.title("📈 Prédiction de Tendances Boursières")

# --- Sélection des Actions ---
# Tu peux ajouter autant de tickers que tu veux (ex: AAPL pour Apple, MSFT pour Microsoft)
stocks = ("AAPL", "GOOG", "MSFT", "GME", "TSLA", "BTC-USD", "ETH-USD")
selected_stock = st.selectbox("Sélectionnez l'action à analyser", stocks)

# --- Slider pour la durée de prédiction ---
n_years = st.slider("Années de prédiction:", 1, 4)
period = n_years * 365

# --- Chargement des données ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    # Yfinance met la date en index, on la remet en colonne pour Prophet
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Chargement des données...")
data = load_data(selected_stock)
data_load_state.text("Chargement des données... Terminé !")

# --- Affichage des données brutes ---
st.subheader(f'Données brutes pour {selected_stock}')
st.write(data.tail()) # Affiche les 5 dernières lignes

# Graphique des prix d'ouverture et de fermeture
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Prix d'ouverture"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Prix de fermeture"))
    fig.layout.update(title_text=f'Historique des prix pour {selected_stock}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# --- Préparation pour la Prédiction (Prophet) ---
# Prophet a besoin de deux colonnes spécifiques : 'ds' (date) et 'y' (valeur à prédire)
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# --- Entraînement du Modèle ---
st.subheader('Prédiction de la tendance')
m = Prophet()
m.fit(df_train)

# Création du dataframe futur
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# --- Affichage des Prédictions ---
st.write(f"Prédiction sur {n_years} ans")
st.write(forecast.tail())

# Graphique interactif de la prédiction
st.write(f"Graphique de prévision pour {selected_stock}")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Composantes de la prédiction (Tendances annuelles, hebdomadaires)
st.write("Composantes de la prédiction")
fig2 = m.plot_components(forecast)
st.write(fig2)

# --- Logique simple d'Achat / Vente (Bonus) ---
# On compare la dernière valeur réelle avec la valeur prédite dans le futur
last_price = df_train['y'].iloc[-1]
predicted_price_in_future = forecast['yhat'].iloc[-1]

st.subheader("💡 Recommandation (Basée sur la tendance)")
if predicted_price_in_future > last_price:
    st.success(f"Tendance HAUSSIÈRE 🚀. Le modèle prédit un prix de {predicted_price_in_future:.2f} contre {last_price:.2f} aujourd'hui.")
else:
    st.error(f"Tendance BAISSIÈRE 📉. Le modèle prédit un prix de {predicted_price_in_future:.2f} contre {last_price:.2f} aujourd'hui.")
