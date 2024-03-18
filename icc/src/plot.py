import logging
import matplotlib.pyplot as plt
from seaborn import barplot, histplot, scatterplot
from icc.config import Config, Columns
from icc.src.api import read_data

logging.basicConfig(level=logging.INFO)

def plot_average_bmi_sex(df):
    logging.info("Plotting Average BMI by Sex.")
    fig, ax = plt.subplots() 

    barplot(data=df, x=Columns.SEX, y=Columns.BMI, ax=ax)
    ax.bar_label(ax.containers[0], fmt=f'%.2f')
    ax.set_ylabel('BMI (kg/m²)')
    ax.set_ylim((0, 50))
    ax.set_title("Average BMI for each Sex")
    
    fig.savefig(Config.PATH_TO_PLOT_FOLDER+"average_bmi_sex.png")
    plt.close()
    logging.info("Done.")

def plot_smoker_rate(df):
    logging.info("Plotting Smoker Rate.")
    df_smoker = df.groupby([Columns.SEX, Columns.SMOKER]).size().unstack()
    df_sex = df.groupby(Columns.SEX).size().to_frame(name="total")
    df_ratio = df_smoker.div(df_sex['total'], axis=0)
    
    ax = df_ratio.plot(kind='bar', stacked=True)
    ax.set_ylabel('% of Individuals')
    ax.set_title('Stacked Bar Chart of Smokers by Sex')

    for container in ax.containers:
        ax.bar_label(container, label_type='center', fmt='%.2f%%')

    ax.figure.savefig(Config.PATH_TO_PLOT_FOLDER+'smokers_sex.png')
    plt.close()
    logging.info("Done.")


def plot_charge_distribution_sex(df):
    logging.info("Plotting Charge Distribution by Sex.")
    _, axs = plt.subplots(1, 2, figsize=(8, 4))

    histplot(data=df[df[Columns.SEX] == 'male'], x=Columns.FACT, kde=True, ax=axs[0], color='purple')
    histplot(data=df[df[Columns.SEX] == 'female'], x=Columns.FACT, kde=True, ax=axs[1], color='red')

    axs[0].set_title("For Males")
    axs[1].set_title("For Females")

    plt.savefig(Config.PATH_TO_PLOT_FOLDER+'charges_dist_sex.png')
    plt.close()
    logging.info("Done.")


def plot_bmi_distribution(df):
    logging.info("Plotting BMI Distribution by Sex.")
    _, axs = plt.subplots(1, 2, figsize=(8, 4))

    histplot(data=df[df[Columns.SEX] == 'male'], x=Columns.BMI, kde=True, ax=axs[0], color='purple')
    histplot(data=df[df[Columns.SEX] == 'female'], x=Columns.BMI, kde=True, ax=axs[1], color='red')

    axs[0].set_title("For Males")
    axs[1].set_title("For Females")

    plt.savefig(Config.PATH_TO_PLOT_FOLDER+'bmi_dist_sex.png')
    plt.close()
    logging.info("Done.")
    

def plot_bmi_scatter_smoker(df):
    logging.info("Plotting BMI vs. Charge Rate by Smoker/Non-Smoker.")
    _, ax = plt.subplots()
    scatterplot(data=df, x=Columns.BMI, y=Columns.FACT, ax=ax, hue='smoker')
    ax.set_ylabel('Charges [USD]')
    ax.set_xlabel('BMI [kg/m²]')
    ax.set_title('Charges vs. BMI Plot for Smokers and Non-Smokers')
    plt.savefig(Config.PATH_TO_PLOT_FOLDER+"bmi_scatter_smoker.png")
    logging.info("Done.")


def plot_nchildren(df):
    logging.info("Plotting Number of Children.")
    df_children = df.groupby(Columns.KIDS).size().to_frame(name="total")
    ax = df_children.plot(kind='bar', stacked=False, legend=False)

    ax.set_ylabel('Number of Individuals')
    ax.set_title('Bar Chart for Number of Children')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0) # Force rotation to be zero, 90 default

    ax.figure.savefig(Config.PATH_TO_PLOT_FOLDER+"number_of_children.png")
    logging.info("Done.")
    
def plot_children_by_region(df):
    logging.info("Plotting Average Number of Children for Different Regions.")
    _, ax = plt.subplots()
    barplot(data=df, y=Columns.LOC, x=Columns.KIDS, ax=ax, palette='Set1', ci=None)
    ax.set_title("Average No. of Children for Different Regions")
    ax.set_xlim((0, 1.5))

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='%.1f')    

    ax.figure.savefig(Config.PATH_TO_PLOT_FOLDER+"avg_children_by_region.png")
    logging.info("Done.")

def plot_charge_by_region(df):
    logging.info("Plotting Average Charges by Region.")
    _, ax = plt.subplots()
    barplot(data=df, y=Columns.LOC, x=Columns.FACT, ax=ax, palette='Set1', ci=None)
    ax.set_title("Average Charge for Different Regions")
    ax.set_xlim((0, 19000))

    for container in ax.containers:
        ax.bar_label(container, label_type='edge', fmt='${:,.0f}')    

    ax.figure.savefig(Config.PATH_TO_PLOT_FOLDER+"avg_charge_by_region.png")
    logging.info("Done.")

class Plot:
    def __init__(self, conf=Config):
        self.file_path = conf.PATH_TO_DATA
        self.df = read_data(self.file_path)

    def run(self):
        logging.info("Starting Exploratory Data Analysis Task.")
        
        plot_average_bmi_sex(self.df)
        plot_smoker_rate(self.df)
        plot_charge_distribution_sex(self.df)
        plot_bmi_distribution(self.df)
        plot_bmi_scatter_smoker(self.df)
        plot_nchildren(self.df)
        plot_children_by_region(self.df)
        plot_charge_by_region(self.df)

        logging.info("Exploratory Data Analysis (EDA) completed.")
