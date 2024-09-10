import numpy as np
import threading
import pdb 
from collections import Counter
import pandas as pd
import os
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import xgboost as xgb
from xgboost import XGBClassifier

# Function to fetch and process data
def fetch_and_process_data(period_in_months):
    engine_1m = create_engine('mysql+mysqlconnector://root:orenkats95@localhost/qqq_1m_data')
    query_1m = "SELECT datetime, open, high, low, close, volume FROM qqq_1m_data ORDER BY datetime ASC"
    df_1m = pd.read_sql(query_1m, engine_1m)

    df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
    df_1m.set_index('datetime', inplace=True)
    df_1m = df_1m[~df_1m.index.duplicated(keep='first')]

    # Filter to only the last n months
    df_1m = df_1m[df_1m.index >= (df_1m.index.max() - pd.DateOffset(months=period_in_months))]

    df_1m['total_range_pct'] = (df_1m['high'] - df_1m['low']) / 100
    df_1m['lrc_50'] = ta.linreg(df_1m['close'], length=20)
    df_1m['lrc_50_slope'] = df_1m['lrc_50'].diff()
    df_1m['dis_from_lrc_50'] = (df_1m['close'] - df_1m['lrc_50']) / 100
     
    # Calculate the trailing stop (TS)
    hhv_period = 5
    multiplier = 2.5
    df_1m['atr'] = ta.atr(df_1m['high'], df_1m['low'], df_1m['close'], length=5)
    df_1m['highest_hhv'] = df_1m['high'].rolling(window=hhv_period).max()
    df_1m['lowest_lhv'] = df_1m['low'].rolling(window=hhv_period).min()
    df_1m['trailing_stop_long'] = df_1m['highest_hhv'] - multiplier * df_1m['atr']
    df_1m['trailing_stop_short'] = df_1m['lowest_lhv'] + multiplier * df_1m['atr']

    lookback = 1
    for lag in range(1, lookback + 2):
        df_1m[f'lag_open_{lag}'] = df_1m['close'].shift(lag)
        df_1m[f'lag_high_{lag}'] = df_1m['high'].shift(lag)
        df_1m[f'lag_low_{lag}'] = df_1m['low'].shift(lag)
        df_1m[f'lag_close_{lag}'] = df_1m['close'].shift(lag)
        df_1m[f'lag_volume_{lag}'] = df_1m['volume'].shift(lag)
        df_1m[f'lag_total_range_pct_{lag}'] = df_1m['total_range_pct'].shift(lag)
        df_1m[f'lag_dis_from_lrc_50_{lag}'] = df_1m['dis_from_lrc_50'].shift(lag)

    df_1m.dropna(inplace=True)

    return df_1m

# Function to create technical features in a separate DataFrame
def create_features(df_1m):
    df_features = pd.DataFrame(index=df_1m.index)  # Initialize a new DataFrame with the same index as df_1m
    df_features['volume'] = df_1m['volume']
    df_features['lag_volume_1'] = df_1m['lag_volume_1']
    # df_features['lag_volume_2'] = df_1m['lag_volume_2']
    df_features['close_change_pct'] = df_1m['close'].pct_change()
    df_features['lag_1_close_change_pct'] = df_1m['lag_close_1'].pct_change()
    # df_features['lag_2_close_change_pct'] = df_1m['lag_close_2'].pct_change()
    # df_features['total_range_pct'] = df_1m['total_range_pct']
    # df_features['lag_total_range_pct_1'] = df_1m['lag_total_range_pct_1']
    # df_features['dis_from_lrc_50'] = df_1m['dis_from_lrc_50']
    # df_features['lrc_50_slope'] = df_1m['lrc_50'].diff()
    df_features['atr'] = ta.atr(df_1m['high'], df_1m['low'], df_1m['close'], length=3)

    df_features.dropna(inplace=True)
    return df_features

# Prepare data for the model using df_features
def prepare_data_for_model(df_1m, df_features):
    X = df_features.values

    # Define thresholds for the three classes: buy, sell, neutral
    upper_threshold = 0.0005  # Above this, it's a buy signal
    lower_threshold = -0.0005  # Below this, it's a sell signal


    # Calculate the price change percentage
    price_change_pct = (df_1m['close'].shift(-1) - df_1m['close']) / df_1m['close']
    price_change_2 = (df_1m['close'].shift(-2) - df_1m['close']) / df_1m['close']
    # Create the target variable Y with three classes
    Y = pd.Series(2, index=df_1m.index)  # Initialize with neutral signals (class 2)

    # Assign buy signals where price change is greater than upper threshold (class 1)
    Y[(price_change_pct >= upper_threshold)] = 1

    # Assign sell signals where price change is lower than lower threshold (class 0)
    Y[(price_change_pct <= lower_threshold)] = 0

    # Align the index of Y with features
    Y = Y.loc[df_features.index]
    Y = Y.values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y

# Function to simulate trades
def simulate_trades(df_1m_slice, predictions, df_features, stop_loss_pct=0.01, take_profit_pct=0.02):
    trades = []  # Track profits/losses
    buy_signals = []  # Track buy signals
    sell_signals = []  # Track sell signals
    closed_positions_long = []
    closed_positions_short = []
    position = None  # Track whether we're in a position ('long', 'short', or None)
    entry_price = None  # Track the price at which we entered a position


    for j in range(len(predictions)):
        current_price = df_1m_slice['close'].iloc[j]
        lrc_50 = df_1m_slice['lrc_lrc_50'] = df_1m_slice['lrc_50'].iloc[j]
        tsl_long = df_1m_slice['trailing_stop_long'].iloc[j]
        tsl_short = df_1m_slice['trailing_stop_short'].iloc[j]
        lrc_50_slope = df_1m_slice['lrc_50_slope'].iloc[j]
        timestamp = df_1m_slice.index[j]
        time_of_day = timestamp.strftime('%H:%M')

        # Exit long position if necessary
        if position == 'long':
            profit = current_price - entry_price
            if (predictions[j] == 0 and current_price < tsl_long) or time_of_day == '15:59':  # Exit long
                trades.append(profit)
                print(f"In Long, {time_of_day}, profit: {profit}")
                if time_of_day == '15:59':
                    print("Closed Long End Day")
                    closed_positions_long.append((timestamp, current_price))  # Track sell signal (exit)
                position = None

        # Exit short position if necessary
        elif position == 'short':
            profit = entry_price - current_price
            if (predictions[j] == 1 and current_price > tsl_short) or time_of_day == '15:59':  # Exit short
                trades.append(profit)
                print(f"In Short, {time_of_day}, profit: {profit}")
                if time_of_day == '15:59': 
                    print("Closed Short End Day")
                    closed_positions_short.append((timestamp, current_price))  # Track buy signal (exit)
                position = None

        # Enter long position
        if predictions[j] == 1 and position is None and time_of_day < '15:50' and (current_price > lrc_50 or lrc_50_slope > 0):
            position = 'long'
            print(f"Entry Long, {time_of_day}")
            entry_price = current_price
            buy_signals.append((timestamp, entry_price))

        # Enter short position
        if predictions[j] == 0 and position is None and time_of_day < '15:50' and (current_price < lrc_50 or lrc_50_slope < 0):
            position = 'short'
            print(f"Entry Short, {time_of_day}")
            entry_price = current_price
            sell_signals.append((timestamp, entry_price))

    return trades, buy_signals, sell_signals, closed_positions_long, closed_positions_short

# Function to calculate Win Rate, Sharpe Ratio, and Maximum Drawdown
def calculate_metrics(returns):
    returns = np.array(returns)
    win_rate = np.sum(returns > 0) / len(returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns)
    print(f'Win Rate: {win_rate:.2f}')
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    print(f'Max Drawdown: {max_drawdown:.2f}')
    return win_rate, sharpe_ratio, max_drawdown

# Function to plot signals
def plot_signals(df_1m, trades, buy_signals, sell_signals, closed_positions_long, closed_positions_short, output_dir):

    # Unzip buy and sell signals
    buy_times, buy_prices = zip(*buy_signals) if buy_signals else ([], [])
    sell_times, sell_prices = zip(*sell_signals) if sell_signals else ([], [])
    closed_long_times, closed_long_prices = zip(*closed_positions_long) if closed_positions_long else ([], [])
    closed_short_times, closed_short_prices = zip(*closed_positions_short) if closed_positions_short else ([], [])

    print(f'Total trades: {len(trades)}')
    calculate_metrics(trades)

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot the close price directly from df_1m
    plt.plot(df_1m.index, df_1m['close'], label='Close Price', color='blue', linewidth=0.5)
    plt.plot(df_1m.index, df_1m['lrc_50'], label='LRC', color='purple', linewidth=0.2)

    # Plot buy signals (green triangles)
    plt.scatter(buy_times, buy_prices, color='green', marker='^', label='Buy', s=50, edgecolors='black')
    plt.scatter(sell_times, sell_prices, color='red', marker='v', label='Sell', s=50, edgecolors='black')
    plt.scatter(closed_long_times, closed_long_prices, color='green', marker='_', label='Close (Long)', s=30)
    plt.scatter(closed_short_times, closed_short_prices, color='red', marker='_', label='Close (Short)', s=30)

    # Add labels, legend, and title
    plt.title('Close Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'Signals.png'))

    plt.show()

# Function to plot learning curve in a separate thread
def plot_learning_curve_thread(X_scaled, Y, output_dir):
    plot_learning_curve(X_scaled, Y, output_dir)

# Function to plot the learning curve
def plot_learning_curve(X_scaled, Y, output_dir):
    train_sizes, train_scores, test_scores = learning_curve(
        XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='multi:softmax', num_class=3, reg_lambda=1, gamma=0.2),
        X_scaled, Y, train_sizes=np.linspace(0.1, 1.0, 20), cv=5, scoring='accuracy', n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    print(f'Average accuracy: {np.mean(test_scores):.2f}')

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Cross-validation score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="g", label="Training accuracy")
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, 'Learning_Curve_XGBoost.png'))
    plt.show()

# Function to plot feature importance
def plot_feature_importance(model_xgb, features, output_dir, period_in_months):
    importance = model_xgb.get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])

    feature_map = {f'f{i}': feature for i, feature in enumerate(features)}
    importance_df['Feature'] = importance_df['Feature'].map(feature_map)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title(f'Feature Importance Based on XGBoost {period_in_months} Months')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(os.path.join(output_dir, 'Feature_Importance_Based_on_XGBoost.png'))
    plt.show()

# Main function
def main():
    output_dir = 'xgboost_project'
    os.makedirs(output_dir, exist_ok=True)

    period_in_months = 72
    df_1m = fetch_and_process_data(period_in_months)

    # Create features and store them in df_features
    df_features = create_features(df_1m)

    # Prepare data using df_features for the model
    X_scaled, Y = prepare_data_for_model(df_1m, df_features)
    # Create DMatrix for training and testing

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, shuffle=False)
    # Create DMatrix for training and testing
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test)

    param = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'multi:softmax',
        'num_class': 3,
        'lambda': 1,
        'gamma': 0.2
    }

    model_xgb = xgb.train(param, dtrain, num_boost_round=100, early_stopping_rounds=2, evals=[(dtrain, 'eval')], verbose_eval=False)

    # Get the predicted class labels directly
    predictions_test = model_xgb.predict(dtest).astype(int)

    # Use Counter to count occurrences of each class
    class_counts = Counter(predictions_test)

    # Now you can access the count of any class, for example:
    buy_signals_counter = class_counts[1]
    sell_signals_counter = class_counts[0]
    hold_signals_counter = class_counts[2]

    accuracy_xgb_test = accuracy_score(Y_test, predictions_test)

    df_1m_test = df_1m[-len(Y_test):]

    # Simulate trades based on predictions
    trades, buy_signals, sell_signals, closed_long_signals, closed_short_signals = simulate_trades(df_1m_test, predictions_test, df_features)

    print(f"Hold Signals: {hold_signals_counter}")
    print(f"Buy Signals: {buy_signals_counter}")
    print(f"Sell Signals: {sell_signals_counter}")
    print(f"Last Accuracy: {accuracy_xgb_test}")

    # Plot the buy and sell signals
    plot_signals(df_1m, trades, buy_signals, sell_signals, closed_long_signals, closed_short_signals, output_dir)

    # Plot feature importance
    plot_feature_importance(model_xgb, df_features.columns, output_dir, period_in_months)

    # Start the learning curve plot in a new thread
    plot_learning_curve(X_scaled, Y, output_dir)

    # Optionally save the model if needed
    model_xgb.save_model(os.path.join(output_dir, 'xgboost_model.json'))

if __name__ == "__main__":
    main()

