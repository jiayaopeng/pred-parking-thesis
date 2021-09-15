import matplotlib.pyplot as plt
import pandas as pd


def area_split_boxplot(data, title):
    # loc the data separately for each algorithm
    df_lr = data.loc[(slice(None) , 'Logistic Regression'), :]
    df_rf = data.loc[(slice(None) , 'Random Forest'), :]
    df_cb = data.loc[(slice(None) , 'Catboost'), :]

    # anaylyze the metric for each algorithm for each area split
    color = {
      "boxes": "DarkGreen",
      "whiskers": "DarkOrange",
      "medians": "DarkBlue",
      "caps": "Gray",
     } 

    figure, axes = plt.subplots(1, 3, figsize=(20,8))

    df_lr.plot.box(ax=axes[0], color=color, sym="r+", grid = True, title = "Logistic Regression", ylim=(-1, 1))
    df_rf.plot.box(ax=axes[1], color=color, sym="r+",  grid = True, title = "Random Forest", ylim=(-1, 1))
    df_cb.plot.box(ax=axes[2], color=color, sym="r+", grid = True, title = "Catboost", ylim=(-1, 1))

    plt.suptitle(title)
    plt.show()
    

def area_split_mean_plot(data, title):
    df_lr = data.loc[(slice(None), 'Logistic Regression'), :]
    df_rf = data.loc[(slice(None), 'Random Forest'), :]
    df_cb = data.loc[(slice(None), 'Catboost'), :]
    
    # get the mean for each area split for each metric
    df_lr_mean = df_lr.mean(axis = 0)
    df_rf_mean = df_rf.mean(axis = 0)
    df_cb_mean = df_cb.mean(axis = 0)

    df_means = pd.DataFrame(
        {
        'Logistic Regression': df_lr_mean,
        'Random Forest': df_rf_mean,
        'Catboost': df_cb_mean
        }
    )

    df_means.plot(kind='bar', title=title)
    plt.ylim([0.0, 0.20])
    plt.grid()
    plt.show()
    