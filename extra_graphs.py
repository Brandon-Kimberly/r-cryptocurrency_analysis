import csv
import matplotlib.pyplot as plt


# This displays a graph of Bitcoin's absolute price over the last 52 weeks
def get_btc_price_graph():
    x_data = []
    y_data = []
    line_count = 0
    with open("BTC-price_data.csv") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            try:
                if line_count != 0 and line_count != 358:
                    x_data.append(str(row[1]))
                    y_data.append(int(float(row[2])))
                line_count += 1
            except:
                pass

    plt.plot(x_data, y_data)
    plt.title("Price of Bitcoin Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price of Bitcoin in USD")
    plt.show()


# Displays a graph of Bitcoin's price percentage change week over week over the last 52 weeks
# Note: This returns the y_data[] list because it is needed in the get_sentiment_to_btc_percent_diff_graph() function
def get_btc_price_percent_graph():
    line_count = 0
    btc_gain_xdata = ['05/05']
    btc_gain_ydata = [None]
    previous_week = None

    with open("BTC-price_data.csv") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if int(line_count % 7) == 1 and line_count != 1:
                try:
                    btc_gain_xdata.append(row[1][:len(row[1]) - 5])
                    if previous_week is not None:
                        last_week_price = int(float(previous_week[2]))
                        current_price = int(float(row[2]))
                        gain_percent = (current_price - last_week_price) / last_week_price
                        btc_gain_ydata.append(gain_percent)
                    else:
                        btc_gain_ydata.append(0.5)
                    previous_week = row
                except:
                    pass
            line_count += 1

    btc_gain_xdata.append("04/27")
    btc_gain_ydata.append(None)

    plt.close()
    plt.plot(btc_gain_xdata, btc_gain_ydata)

    return btc_gain_ydata


# Graphs the difference in sentiment score and Bitcoin percent price change week over week
# NOTE: This graph only works if the time period specified in the cmd line arg is <= 52, any more and it won't graph
#       the full diff.
def get_sentiment_to_btc_percent_diff_graph(sentiment_x_data, sentiment_y_data, btc_gain_ydata):
    diff_x_data = sentiment_x_data.copy()
    diff_y_data = []

    for i in range(len(sentiment_y_data)):
        try:
            diff_y_data.append(sentiment_y_data[i] - btc_gain_ydata[i])
        except TypeError:
            diff_y_data.append(None)

    plt.close()
    plt.plot(diff_x_data, diff_y_data)
    plt.title("Difference in Sentiment Score and Percent Increase in Bitcoin Price Week Over Week")
    plt.xlabel("Date")
    plt.ylabel("Difference")
    plt.show()
