import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models
import tensorflow as tf

# --- Technical Indicators ---
def schaff_trend_cycle(close, fast_period=23, slow_period=50, cycle_period=10):
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    stoch_macd = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd)
    stoch_macd_smoothed1 = stoch_macd.ewm(span=3, adjust=False).mean()
    stoch_macd_smoothed2 = stoch_macd_smoothed1.ewm(span=3, adjust=False).mean()
    return stoch_macd_smoothed2

def chande_momentum_oscillator(close, window=14):
    delta = close.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    sum_gains = gains.rolling(window=window).sum()
    sum_losses = losses.rolling(window=window).sum()
    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    return cmo

# --- Capsule Network Components ---
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-7)
    return scale * vectors

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(
            shape=[1, self.input_num_capsules, self.num_capsules,
                   self.input_dim_capsule, self.dim_capsule],
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1, 1])
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat = tf.matmul(inputs_tiled, W_tiled)
        u_hat = tf.squeeze(u_hat, [3])
        b = tf.zeros([batch_size, self.input_num_capsules, self.num_capsules, 1])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            v = squash(s)
            if i < self.routings - 1:
                b += tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)
        return tf.squeeze(v, [1])

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))

# --- Models ---
def build_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(50, input_shape=input_shape),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_capsnet(input_shape, num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 5, strides=2, activation='relu')(x)
    x = layers.Reshape((-1, 8))(x)
    x = layers.Lambda(squash)(x)
    digitcaps = CapsuleLayer(num_capsules=num_classes, dim_capsule=8, routings=3)(x)
    out_caps = Length()(digitcaps)
    model = models.Model(inputs, out_caps)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Data Preparation ---
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['STC'] = schaff_trend_cycle(df['Close'])
    df['CMO'] = chande_momentum_oscillator(df['Close'])
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)
    features = ['Close','High','Low','Open','Volume','STC','CMO']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(scaled, columns=features)
    df_scaled['Target'] = df['Target'].values
    return df_scaled, features

def create_sequences(df, features, seq_len=20):
    X, y = [], []
    data = df[features].values
    target = df['Target'].values
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len])
    return np.array(X), np.array(y)

def train_test_split(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]

# --- Metrics ---
def evaluate(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

if __name__ == "__main__":
    df, features = load_data('MSFT_1986_2025-06-30.csv')
    X, y = create_sequences(df, features, seq_len=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lstm_model = build_lstm((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    lstm_pred = (lstm_model.predict(X_test) > 0.5).astype(int).flatten()
    lstm_metrics = evaluate(y_test, lstm_pred)

    y_train_ohe = tf.keras.utils.to_categorical(y_train, 2)
    caps_model = build_capsnet((X_train.shape[1], X_train.shape[2]))
    caps_model.fit(X_train, y_train_ohe, epochs=5, batch_size=32, verbose=0)
    caps_pred_prob = caps_model.predict(X_test)
    caps_pred = np.argmax(caps_pred_prob, axis=1)
    caps_metrics = evaluate(y_test, caps_pred)

    print("LSTM Metrics:", lstm_metrics)
    print("CapsNet Metrics:", caps_metrics)
    if caps_metrics['accuracy'] > lstm_metrics['accuracy']:
        print("CapsNet performed better based on accuracy")
    else:
        print("LSTM performed better based on accuracy")
