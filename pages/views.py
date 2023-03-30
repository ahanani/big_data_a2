import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from django.urls import reverse
from io import BytesIO


def homePageView(request):
    return render(request, 'home.html')


def visualsPageView(request):
    return render(request, 'analytics.html')


def predictionPageView(request):
    return render(request, 'prediction.html')


def heatmap_view(request):
    df = pd.read_csv('./bike_buyers_clean.csv', sep=',')

    df = df.replace({'No': 0, 'Yes': 1})
    df = df.replace({'Male': 0, 'Female': 1})
    df = df.replace({'Married': 0, 'Single': 1})
    df = df.replace({'Bachelors': 0, 'Partial College': 1, 'High School': 2,
                    'Partial High School': 3, 'Graduate Degree': 4})
    df = df.replace({'Skilled Manual': 0, 'Clerical': 1, 'Professional': 2, 'Manual': 3, 'Management': 4})
    df = df.replace({'0-1 Miles': 0, '1-2 Miles': 1, '2-5 Miles': 2, '5-10 Miles': 3, '10+ Miles': 4})
    df = df.replace({'Europe': 0, 'Pacific': 1, 'North America': 2})

    X = df.drop(['ID'], axis=1)

    fig, ax = plt.subplots()
    corr = X.corr()
    sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="YlGnBu")

    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='image/png')
    return response


def hist_of_target(request):
    df = pd.read_csv('./bike_buyers_clean.csv', sep=',')

    plt.hist(df["Purchased Bike"], bins=10)
    plt.xlabel("Purchased Bike")
    plt.ylabel("Frequency")
    plt.title('Cx Purchased Bike')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='image/png')
    return response


def get_numeric_summary(reqesut):
    df = pd.read_csv('./bike_buyers_clean.csv', sep=',')
    df = df.drop(['Purchased Bike', 'ID'], axis=1)
    summary_df = pd.DataFrame(df.describe().transpose())
    df_html = summary_df.to_html(border=0, justify='center')
    df_html = df_html.replace('<table class="dataframe">', """<table style="font-family: Arial, sans-serif; color: #40513b">""")
    return HttpResponse(df_html)

def get_non_num_summary(reqesut):
    df = pd.read_csv('./bike_buyers_clean.csv', sep=',')
    df = df.drop(['Purchased Bike', 'ID'], axis=1)
    summary_df = pd.DataFrame(df.describe(include=['object']).transpose())
    df_html = summary_df.to_html(border=0, justify='center')
    df_html = df_html.replace('<table class="dataframe">', """<table style="font-family: Arial, sans-serif; color: #40513b">""")
    return HttpResponse(df_html)


def homePost(request):
    Gender = request.POST.get('Gender')
    Age = request.POST.get('Age')
    Income = request.POST.get('Income')
    Education = request.POST.get('Education')
    Commute_Distance = request.POST.get('Commute_Distance')
    Occupation = request.POST.get('Occupation')
    Cars = request.POST.get('Cars')

    with open('model_pkl', 'rb') as f:
        loadedModel = pickle.load(f)

        singleSampleDf = pd.DataFrame(columns=['Gender', 'Income', 'Education', 'Occupation', 'Commute Distance', 'Age'])

        singleSampleDf = singleSampleDf.append({'Gender': Gender, 'Income': Income, 'Education': Education,
                                            'Occupation': Occupation, 'Commute Distance': Commute_Distance, 'Age': Age},
                                            ignore_index=True)

        singlePrediction = loadedModel.predict(singleSampleDf)

    return HttpResponseRedirect(reverse('result', kwargs={'answer': singlePrediction}))


def result(request, answer):
    if answer[1] == '0':
        return render(request, 'result.html',
                      {'answer': "Rev Up Your Ride with 20% Off: You've Qualified for Our Exclusive Bike Discount!"})
    else:
        return render(request, 'result.html', {'answer': "Sorry, You Didn't Qualify: Keep an Eye Out for Future Deals!"})
