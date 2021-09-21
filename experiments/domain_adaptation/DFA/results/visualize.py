import matplotlib.pyplot as plt
import seaborn as sns


def plot_mean_of_mean(mean, figsize=None):
    mean.plot.bar(
        title="Average Matthew Across All Areas For Different Exepriment Setting",
        grid=True,
        xlabel="Experiment Setting",
        ylabel="Matthew Score",
        rot=45,
        figsize=None
    )


def plot_corr_matthew_datasize(matthew_size, highest_val_mean):
    """
    We plot the correlation between number of datapoints in the area and the matthew score of the area

    """
    # merge them together
    ax1 = matthew_size.plot(
        kind='scatter',
        x='data_size',
        y='val_matthew',
        color='r',
        label='val_matthew and size'
    )
    ax2 = matthew_size.plot(
        kind='scatter',
        x='data_size',
        y='test_matthew',
        color='g',
        label='test_matthew and size',
        ax=ax1
    )
    plt.title(f'{highest_val_mean} \n Correlation number of datapoints in the area and the average matthew score of that area')


def plot_matthew_per_area(data, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        ax=ax,
        x=data["target area"],
        y=data["matthew_score"],
        hue=data["metric"],
        data=data,
        palette="hls"
    )
    sns.despine(offset=10, trim=True)
    ax.set_title(title)
    sns.set_style("whitegrid")