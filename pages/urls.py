from django.urls import path
from .views import homePageView, visualsPageView, predictionPageView, homePost, result, heatmap_view, hist_of_target, get_numeric_summary, get_non_num_summary

urlpatterns = [
    path('', homePageView, name='home'),
    path('visuals/', visualsPageView, name='visuals'),
    path('prediction/', predictionPageView, name='prediction'),
    path('homePost/', homePost, name='homePost'),
    path('result/<str:answer>/', result, name='result'),
    path('heatmap/', heatmap_view, name='heatmap_view'),
    path('hist_view/', hist_of_target, name='hist_view'),
    path('numeric_df/', get_numeric_summary, name='numeric_df'),
    path('non_numeric_df/', get_non_num_summary, name='non_numeric_df'),
]
